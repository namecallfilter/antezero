//! `RLBalatro`'s fast poker-hand classifier (`sel_eval`) and its hand-first
//! `core-then-free` per-pick rule (`CoreCtx`, `core_rule`).

use std::ops::Range;
use std::sync::OnceLock;

use crate::card::{HandSort, SuitTest};
use crate::data::{BlindKind, Joker, Suit};
use crate::game::{Game, State};
use crate::num;

/// The hand cards an observation has rows for; cards past them cannot be
/// picked.
pub(super) const MAX_HAND: usize = 24;
pub(super) const N_HANDS: usize = 12;
/// The declared "hand" of a plain play: any one to five cards.
pub(super) const HS_ANY: usize = N_HANDS;
/// The length of the hand context that [`BalEnv::write_all_obs`] writes, from
/// which a policy computes the pick masks of the declared-hand plays itself.
///
/// While a hand is being selected, entry `i < 24` describes hand card `i`:
/// its rank id (`2..=14`, `0` for a card without a rank) in bits 0 to 3, the
/// suits it counts as for a flush (Spades, Hearts, Clubs, Diamonds) in bits
/// 4 to 7, bit 8 if it has no rank, bit 9 if it is forced into the selection
/// and bit 10 for every card present. Entry 24 is the number of hand cards,
/// entry 25 is bit 0 for Four Fingers and bit 1 for Shortcut, and entries 28
/// and 29 are the fewest and the most cards the boss lets a play have. A
/// face-down card has no rank and no suit when face-down cards are hidden.
/// In every other state all entries are 0.
///
/// Entries 26, 27, 30 and 31 are unused, as in `RLBalatro`. Bit 3 of entry 25
/// is its "play blocked" flag, which is only set between the events of a
/// draw and so is always 0 here.
///
/// [`BalEnv::write_all_obs`]: super::BalEnv::write_all_obs
pub const HS_CTX: usize = 32;
const CORE_INF: i32 = 64;

/// The rank sets of one [`straight_lut`] mode: bit `r - 2` for rank id `r`.
const RANK_SETS: usize = 1 << 13;

fn build_straight_lut() -> Vec<[u16; RANK_SETS]> {
	let mut lut = vec![[0u16; RANK_SETS]; 4];
	for (mode, table) in lut.iter_mut().enumerate() {
		let need = if (mode & 2) != 0 { 4 } else { 5 };
		let can_skip = (mode & 1) != 0;
		for (mask, cell) in table.iter_mut().enumerate() {
			let mut len = 0;
			let mut skipped = false;
			let mut straight = false;
			let mut t: u16 = 0;
			for j in 1..=14usize {
				let slot = if j == 1 { 14 } else { j };
				if ((mask >> (slot - 2)) & 1) != 0 {
					len += 1;
					skipped = false;
					t |= 1u16 << (slot - 2);
				} else if can_skip && !skipped && j != 14 {
					skipped = true;
				} else {
					len = 0;
					skipped = false;
					if !straight {
						t = 0;
					}
					if straight {
						break;
					}
				}
				if len >= need {
					straight = true;
				}
			}
			*cell = if straight { t | 0x8000 } else { 0 };
		}
	}
	lut
}

/// Which sets of ranks hold a straight: four tables, one for each mode
/// `2 * four_fingers + shortcut`, indexed by a rank set with bit `r - 2` for
/// rank id `r` (an Ace is bit 12).
///
/// An entry is 0 if the set holds no straight. Otherwise bit 15 is set and
/// the low 13 bits are the ranks of the straight.
#[inline]
#[must_use]
pub fn straight_lut() -> &'static [[u16; RANK_SETS]] {
	static LUT: OnceLock<Vec<[u16; RANK_SETS]>> = OnceLock::new();
	LUT.get_or_init(build_straight_lut)
}

fn build_straight_mins() -> [Vec<u16>; 4] {
	let lut = straight_lut();
	let mut mins = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
	for mode in 0..4 {
		for x in 0..RANK_SETS {
			if (lut[mode][x] & 0x8000) == 0 {
				continue;
			}
			let mut minimal = true;
			for b in 0..13 {
				if ((x >> b) & 1) != 0 && (lut[mode][x & !(1usize << b)] & 0x8000) != 0 {
					minimal = false;
					break;
				}
			}
			if minimal {
				mins[mode].push(u16::try_from(x).expect("a set of the 13 ranks fits 13 bits"));
			}
		}
	}
	mins
}

/// For each [`straight_lut`] mode, the rank sets that hold a straight and no
/// smaller one.
#[inline]
fn straight_mins() -> &'static [Vec<u16>; 4] {
	static MINS: OnceLock<[Vec<u16>; 4]> = OnceLock::new();
	MINS.get_or_init(build_straight_mins)
}

#[derive(Clone, Copy)]
pub(super) struct SelCtx {
	pub(super) nh: usize,
	rbit: [u32; MAX_HAND],
	nominal: [f64; MAX_HAND],
	suit: [u32; 4],
	stone: u32,
	ff: bool,
	shortcut: bool,
	splash: bool,
}

pub(super) fn sel_ctx(g: &Game) -> SelCtx {
	let hc = &g.areas.hand.cards;
	let nh = hc.len().min(MAX_HAND);
	let smeared = g.areas.smeared();
	let mut x = SelCtx {
		nh,
		rbit: [0; MAX_HAND],
		nominal: [0.0; MAX_HAND],
		suit: [0; 4],
		stone: 0,
		ff: g.areas.has_joker(Joker::FourFingers),
		shortcut: g.areas.has_joker(Joker::Shortcut),
		splash: g.areas.has_joker(Joker::Splash),
	};
	for (i, cd) in hc.iter().take(nh).enumerate() {
		let stone = cd.is_stone();
		let id = if stone && !cd.vampired {
			0
		} else {
			cd.rank.id()
		};
		x.rbit[i] = if (1..=14).contains(&id) {
			1u32 << id
		} else {
			0
		};
		for (q, s) in Suit::ALL.into_iter().enumerate() {
			if cd.is_suit(s, SuitTest::Flush, smeared) {
				x.suit[q] |= 1u32 << i;
			}
		}
		x.nominal[i] = cd.nominal(HandSort::Rank);
		if stone {
			x.stone |= 1u32 << i;
		}
	}
	x
}

fn cards_with_ranks(ctx: &SelCtx, pos: &[usize], rm: u32) -> u32 {
	let mut out = 0u32;
	for &idx in pos {
		if (ctx.rbit[idx] & rm) != 0 {
			out |= 1u32 << idx;
		}
	}
	out
}

/// The cards of the hand cards `bits` that score when those `count` cards
/// are played, one bit per hand position.
pub(super) fn sel_eval(ctx: &SelCtx, bits: u32, count: usize) -> u32 {
	if count == 0 {
		return 0;
	}
	let mut pos = [0usize; 5];
	let mut num_pos = 0usize;
	let mut cur = bits;
	while cur != 0 && num_pos < 5 {
		pos[num_pos] = num::index(cur.trailing_zeros());
		num_pos += 1;
		cur &= cur - 1;
	}
	let pos_slice = &pos[..num_pos];
	let (mut s1, mut s2, mut s3, mut s4, mut s5) = (0u32, 0u32, 0u32, 0u32, 0u32);
	for &idx in pos_slice {
		let rb = ctx.rbit[idx];
		s5 |= s4 & rb;
		s4 |= s3 & rb;
		s3 |= s2 & rb;
		s2 |= s1 & rb;
		s1 |= rb;
	}
	let (e2, e3, e4, e5) = (s2 & !s3, s3 & !s4, s4 & !s5, s5);
	let n2 = e2.count_ones();
	let (p5, p4, p3, p2) = (e5 != 0, e4 != 0, e3 != 0, e2 != 0);
	let high = |mask: u32| -> u32 { if mask != 0 { 1u32 << mask.ilog2() } else { 0 } };
	let need = if ctx.ff { 4 } else { 5 };
	let (mut pf, mut ps) = (0u32, 0u32);
	let (mut flush, mut straight) = (false, false);
	if num_pos >= need {
		for &sm in &ctx.suit {
			if num::index((bits & sm).count_ones()) >= need {
				flush = true;
				pf = bits & sm;
				break;
			}
		}
		let mode = (usize::from(ctx.ff) << 1) | usize::from(ctx.shortcut);
		let st_entry = straight_lut()[mode][num::index((s1 >> 2) & 0x1fff)];
		if (st_entry & 0x8000) != 0 {
			straight = true;
			ps = cards_with_ranks(ctx, pos_slice, u32::from(st_entry & 0x1fff) << 2);
		}
	}
	let r_flags = [
		p5 && flush,
		p3 && p2 && flush,
		p5,
		flush && straight,
		p4 || p5,
		p3 && p2,
		flush,
		straight,
		p3 || p4 || p5,
		n2 == 2 || (p3 && n2 == 1),
		p2 || p3 || p4 || p5,
	];
	let sc = match r_flags.iter().position(|&flag| flag) {
		Some(0 | 2) => cards_with_ranks(ctx, pos_slice, e5),
		Some(1 | 5) => cards_with_ranks(ctx, pos_slice, e3 | high(e2)),
		Some(3) => pf | ps,
		Some(4) => cards_with_ranks(ctx, pos_slice, e4),
		Some(6) => pf,
		Some(7) => ps,
		Some(8) => cards_with_ranks(ctx, pos_slice, e3),
		Some(9) => cards_with_ranks(ctx, pos_slice, e2),
		Some(10) => cards_with_ranks(ctx, pos_slice, high(e2)),
		_ => {
			let mut best = pos_slice[0];
			for &idx in &pos_slice[1..] {
				if ctx.nominal[idx] > ctx.nominal[best] {
					best = idx;
				}
			}
			1u32 << best
		}
	};
	sc | if ctx.splash { bits } else { bits & ctx.stone }
}

/// What the per-pick rule reads of the hand.
#[derive(Clone, Copy)]
pub(super) struct CoreCtx {
	pub(super) nh: usize,
	/// A card's rank id less 2, or -1 for a card without a rank.
	pub(super) rank: [i8; MAX_HAND],
	/// The suits a card counts as for a flush, bit `q` for `Suit::ALL[q]`.
	pub(super) suit: [u8; MAX_HAND],
	pub(super) forced: u32,
	pub(super) ff: bool,
	pub(super) sc: bool,
	/// The fewest and the most cards the boss lets a play have.
	pub(super) lo: usize,
	pub(super) hi: usize,
	/// The hand's cards of each rank: `cnt[0]` of any suit, `cnt[1 + q]` of
	/// suit `q`.
	cnt: [[u8; 13]; 5],
	scnt: [u8; 4],
}

impl Default for CoreCtx {
	fn default() -> Self {
		CoreCtx {
			nh: 0,
			rank: [-1; MAX_HAND],
			suit: [0; MAX_HAND],
			forced: 0,
			ff: false,
			sc: false,
			lo: 1,
			hi: 5,
			cnt: [[0; 13]; 5],
			scnt: [0; 4],
		}
	}
}

/// The context of the hand being selected in `g`. With `hide`, a face-down
/// card has no rank and no suit.
pub(super) fn core_ctx(g: &Game, hide: bool) -> CoreCtx {
	let mut x = CoreCtx::default();
	if g.flow.state != State::SelectingHand || g.is_terminal() {
		return x;
	}
	let hc = &g.areas.hand.cards;
	x.nh = hc.len().min(MAX_HAND);
	let smeared = g.areas.smeared();
	for (i, cd) in hc.iter().take(x.nh).enumerate() {
		let stone = cd.is_stone() && !cd.vampired;
		let id = cd.rank.id();
		x.rank[i] = if !stone && (2..=14).contains(&id) {
			(id - 2).cast_signed()
		} else {
			-1
		};
		x.suit[i] = 0;
		for (q, s) in Suit::ALL.into_iter().enumerate() {
			if cd.is_suit(s, SuitTest::Flush, smeared) {
				x.suit[i] |= 1u8 << q;
			}
		}
		if hide && cd.core.face_down {
			x.rank[i] = -1;
			x.suit[i] = 0;
		}
		if cd.forced.is_locked() {
			x.forced |= 1u32 << i;
		}
		for (q, sc) in x.scnt.iter_mut().enumerate() {
			*sc += (x.suit[i] >> q) & 1;
		}
		if let Ok(r) = usize::try_from(x.rank[i]) {
			x.cnt[0][r] += 1;
			for q in 0..4 {
				x.cnt[1 + q][r] += (x.suit[i] >> q) & 1;
			}
		}
	}
	x.ff = g.areas.has_joker(Joker::FourFingers);
	x.sc = g.areas.has_joker(Joker::Shortcut);
	let lo = if g.blind.active(BlindKind::Psychic) {
		5usize
	} else {
		1usize
	};
	let hi = 5usize;
	let forced_cnt = num::index(x.forced.count_ones());
	if lo.max(forced_cnt).max(1) <= hi.min(x.nh).min(5) {
		x.lo = lo.max(1);
		x.hi = hi.min(5);
	}
	x
}

struct RuleScratch {
	cp: [[u8; 13]; 5],
	sp: [u8; 4],
	rem_budget: i32,
	cost: i32,
	use_mask: [u16; 5],
	use_suit: u8,
}

impl RuleScratch {
	fn eval_kind(&mut self, x: &CoreCtx, chans: Range<usize>, k: u8) {
		for c in chans {
			for v in 0..13 {
				if x.cnt[c][v] < k {
					continue;
				}
				let d = i32::from(k.saturating_sub(self.cp[c][v]));
				self.cost = self.cost.min(d);
				if d >= 1 && d <= self.rem_budget {
					self.use_mask[c] |= 1u16 << v;
				}
			}
		}
	}

	fn eval_groups(&mut self, x: &CoreCtx, chans: Range<usize>, ka: u8, kb: u8) {
		for c in chans {
			let mut da = [CORE_INF; 13];
			let mut db = [CORE_INF; 13];
			let (mut a1, mut a2, mut b1, mut b2) = (CORE_INF, CORE_INF, CORE_INF, CORE_INF);
			let (mut ia, mut ib) = (usize::MAX, usize::MAX);
			for v in 0..13 {
				if x.cnt[c][v] >= ka {
					da[v] = i32::from(ka.saturating_sub(self.cp[c][v]));
				}
				if x.cnt[c][v] >= kb {
					db[v] = i32::from(kb.saturating_sub(self.cp[c][v]));
				}
				if da[v] < a1 {
					a2 = a1;
					a1 = da[v];
					ia = v;
				} else if da[v] < a2 {
					a2 = da[v];
				}
				if db[v] < b1 {
					b2 = b1;
					b1 = db[v];
					ib = v;
				} else if db[v] < b2 {
					b2 = db[v];
				}
			}
			for v in 0..13 {
				let ob = if v == ib { b2 } else { b1 };
				let oa = if v == ia { a2 } else { a1 };
				self.cost = self.cost.min(da[v] + ob);
				let w = ((da[v] - 1).max(0) + ob).min((db[v] - 1).max(0) + oa);
				if w < self.rem_budget {
					self.use_mask[c] |= 1u16 << v;
				}
			}
		}
	}

	fn eval_straight(&mut self, x: &CoreCtx) {
		let (pres, have) = (ranks(&self.cp[0]), ranks(&x.cnt[0]));
		for &m in straight_sets(x) {
			if (m & !have) != 0 {
				continue;
			}
			let d = count(m & !pres);
			self.cost = self.cost.min(d);
			if d <= self.rem_budget {
				self.use_mask[0] |= m & !pres;
			}
		}
	}

	/// A straight and a flush among the cards played, as the game finds them:
	/// each on its own. With Four Fingers both take four cards, so one card
	/// of a five-card play may be off suit and one may be off the straight.
	///
	/// For a suit and a straight, each rank of the straight that the
	/// selection lacks costs a card. A card of the suit also counts towards
	/// the flush, and the cards the flush still lacks after those cost one
	/// more each. So a card lowers the cost if it has a missing rank and is
	/// of the suit, or no card of the suit has that rank, or the flush has
	/// cards to spare; and any card of the suit does while the flush lacks
	/// more cards than the missing ranks can give it.
	fn eval_straight_flush(&mut self, x: &CoreCtx) {
		let need: u8 = if x.ff { 4 } else { 5 };
		let (pres, have) = (ranks(&self.cp[0]), ranks(&x.cnt[0]));
		for q in 0..4 {
			if x.scnt[q] < need {
				continue;
			}
			let suited = ranks(&x.cnt[1 + q]);
			let lacking = i32::from(need.saturating_sub(self.sp[q]));
			for &m in straight_sets(x) {
				if (m & !have) != 0 {
					continue;
				}
				let missing = m & !pres;
				let spare = count(missing & suited) - lacking;
				let d = count(missing) + (-spare).max(0);
				self.cost = self.cost.min(d);
				if d > self.rem_budget {
					continue;
				}
				self.use_mask[1 + q] |= missing;
				self.use_mask[0] |= if spare > 0 {
					missing
				} else {
					missing & !suited
				};
				if spare < 0 {
					self.use_suit |= 1u8 << q;
				}
			}
		}
	}
}

/// The ranks `counts` has a card of, as a [`straight_lut`] rank set.
fn ranks(counts: &[u8; 13]) -> u16 {
	let mut set = 0u16;
	for (v, &n) in counts.iter().enumerate() {
		if n > 0 {
			set |= 1u16 << v;
		}
	}
	set
}

/// The smallest rank sets that hold a straight under the hand's jokers.
fn straight_sets(x: &CoreCtx) -> &'static [u16] {
	&straight_mins()[(usize::from(x.ff) << 1) | usize::from(x.sc)]
}

fn count(set: u16) -> i32 {
	set.count_ones().cast_signed()
}

/// Returns `(legal_next_cards_mask, can_stop)` for declared hand `hand` (`0..=11`,
/// or `HS_ANY = 12`) and current selection bitmask `prefix` (forced cards included).
pub(super) fn core_rule(ctx: &CoreCtx, hand: usize, prefix: u32) -> (u32, bool) {
	let budget = num::id(ctx.hi);
	let lo = num::id(ctx.lo);
	let num_sel = prefix.count_ones().cast_signed();
	let rem_budget = budget - num_sel;
	let all_hand = if ctx.nh >= 32 {
		u32::MAX
	} else {
		(1u32 << ctx.nh) - 1
	};
	let avail = all_hand & !prefix;
	if rem_budget < 0 || ctx.nh == 0 {
		return (0, false);
	}
	let mut sc = RuleScratch {
		cp: [[0u8; 13]; 5],
		sp: [0u8; 4],
		rem_budget,
		cost: CORE_INF,
		use_mask: [0u16; 5],
		use_suit: 0u8,
	};
	let mut cur = prefix;
	while cur != 0 {
		let idx = num::index(cur.trailing_zeros());
		cur &= cur - 1;
		for (q, s_cnt) in sc.sp.iter_mut().enumerate() {
			*s_cnt += (ctx.suit[idx] >> q) & 1;
		}
		if let Ok(rk) = usize::try_from(ctx.rank[idx]) {
			sc.cp[0][rk] += 1;
			for q in 0..4 {
				sc.cp[1 + q][rk] += (ctx.suit[idx] >> q) & 1;
			}
		}
	}
	match hand {
		0 => sc.eval_kind(ctx, 1..5, 5),
		1 => sc.eval_groups(ctx, 1..5, 3, 2),
		2 => sc.eval_kind(ctx, 0..1, 5),
		3 => sc.eval_straight_flush(ctx),
		4 => sc.eval_kind(ctx, 0..1, 4),
		5 => sc.eval_groups(ctx, 0..1, 3, 2),
		6 => {
			let need: u8 = if ctx.ff { 4 } else { 5 };
			for q in 0..4 {
				if ctx.scnt[q] < need {
					continue;
				}
				let diff = i32::from(need.saturating_sub(sc.sp[q]));
				sc.cost = sc.cost.min(diff);
				if diff >= 1 && diff <= rem_budget {
					sc.use_suit |= 1u8 << q;
				}
			}
		}
		7 => sc.eval_straight(ctx),
		8 => sc.eval_kind(ctx, 0..1, 3),
		9 => {
			sc.eval_groups(ctx, 0..1, 2, 2);
			sc.eval_kind(ctx, 0..1, 4);
		}
		10 => sc.eval_kind(ctx, 0..1, 2),
		_ => {
			sc.cost = i32::from(num_sel < 1);
		}
	}
	sc.cost = sc.cost.min(CORE_INF);
	let stop = sc.cost == 0 && num_sel >= lo;
	if rem_budget < 1 || sc.cost > rem_budget {
		return (0, stop);
	}
	if sc.cost < rem_budget || hand >= 11 {
		return (avail, stop);
	}
	let mut out_mask = 0u32;
	let mut rem = avail;
	while rem != 0 {
		let idx = num::index(rem.trailing_zeros());
		rem &= rem - 1;
		let mut ok = (ctx.suit[idx] & sc.use_suit) != 0;
		if let Ok(rk) = usize::try_from(ctx.rank[idx]) {
			ok |= ((sc.use_mask[0] >> rk) & 1) != 0;
			for q in 0..4 {
				if ok {
					break;
				}
				ok |= ((ctx.suit[idx] >> q) & 1) != 0 && ((sc.use_mask[1 + q] >> rk) & 1) != 0;
			}
		}
		if ok {
			out_mask |= 1u32 << idx;
		}
	}
	(out_mask, stop)
}
