//! Fast poker-hand classifier (`sel_eval`) and the hand-first `core-then-free`
//! per-pick rule (`CoreCtx`, `core_rule`) ported directly from
//! `RLBalatro/env/src/hand_eval.hpp`.

use std::ops::Range;
use std::sync::OnceLock;

use crate::areas::HandSort;
use crate::data::{BlindKind, Joker, Suit};
use crate::game::{Game, State};
use crate::num;
use crate::playing_card::SuitTest;

pub const MAX_HAND: usize = 24;
pub const N_HANDS: usize = 12;
pub const HS_ANY: usize = N_HANDS;
pub const HS_CTX: usize = 32;
pub const CORE_INF: i32 = 64;

pub struct StraightLut {
	pub run: Vec<[u16; 1 << 13]>,
}

pub struct StraightMins {
	pub m: [Vec<u16>; 4],
}

fn build_straight_lut() -> StraightLut {
	let mut lut = StraightLut {
		run: vec![[0u16; 1 << 13]; 4],
	};
	for (mode, table) in lut.run.iter_mut().enumerate() {
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

#[inline]
#[must_use]
pub fn straight_lut() -> &'static StraightLut {
	static LUT: OnceLock<StraightLut> = OnceLock::new();
	LUT.get_or_init(build_straight_lut)
}

fn build_straight_mins() -> StraightMins {
	let lut = straight_lut();
	let mut mins = StraightMins {
		m: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
	};
	for mode in 0..4 {
		for x in 0..(1usize << 13) {
			if (lut.run[mode][x] & 0x8000) == 0 {
				continue;
			}
			let mut minimal = true;
			for b in 0..13 {
				if ((x >> b) & 1) != 0 && (lut.run[mode][x & !(1usize << b)] & 0x8000) != 0 {
					minimal = false;
					break;
				}
			}
			if minimal {
				mins.m[mode].push(u16::try_from(x).unwrap_or(0));
			}
		}
	}
	mins
}

#[inline]
#[must_use]
pub fn straight_mins() -> &'static StraightMins {
	static MINS: OnceLock<StraightMins> = OnceLock::new();
	MINS.get_or_init(build_straight_mins)
}

#[derive(Clone, Copy)]
pub struct SelCtx {
	pub nh: usize,
	pub rbit: [u32; MAX_HAND],
	pub nominal: [f64; MAX_HAND],
	pub suit: [u32; 4],
	pub stone: u32,
	pub forced: u32,
	pub ff: bool,
	pub shortcut: bool,
	pub splash: bool,
}

#[derive(Clone, Copy)]
pub struct SelEval {
	pub hand_type: i32,
	pub scoring: u32,
	pub contains: u16,
}

#[must_use]
pub fn sel_ctx(g: &Game) -> SelCtx {
	let hc = &g.areas.hand.cards;
	let nh = hc.len().min(MAX_HAND);
	let smeared = g.areas.smeared();
	let mut x = SelCtx {
		nh,
		rbit: [0; MAX_HAND],
		nominal: [0.0; MAX_HAND],
		suit: [0; 4],
		stone: 0,
		forced: 0,
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
		if cd.forced_selection {
			x.forced |= 1u32 << i;
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

#[must_use]
pub fn sel_eval(ctx: &SelCtx, bits: u32, count: usize) -> SelEval {
	if count == 0 {
		return SelEval {
			hand_type: -1,
			scoring: 0,
			contains: 0,
		};
	}
	let mut pos = [0usize; 5];
	let mut num_pos = 0usize;
	let mut cur = bits;
	while cur != 0 && num_pos < 5 {
		pos[num_pos] = usize::try_from(cur.trailing_zeros()).unwrap_or(0);
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
			if usize::try_from((bits & sm).count_ones()).unwrap_or(0) >= need {
				flush = true;
				pf = bits & sm;
				break;
			}
		}
		let mode = (usize::from(ctx.ff) << 1) | usize::from(ctx.shortcut);
		let st_entry = straight_lut().run[mode][usize::try_from((s1 >> 2) & 0x1fff).unwrap_or(0)];
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
		true,
	];
	let mut contains = 0u16;
	for (h, &flag) in r_flags.iter().enumerate() {
		if flag {
			contains |= 1u16 << h;
		}
	}
	let hand_type = num::id(usize::try_from(contains.trailing_zeros()).unwrap_or(11));
	let sc = match hand_type {
		0 | 2 => cards_with_ranks(ctx, pos_slice, e5),
		1 | 5 => cards_with_ranks(ctx, pos_slice, e3 | high(e2)),
		3 => pf | ps,
		4 => cards_with_ranks(ctx, pos_slice, e4),
		6 => pf,
		7 => ps,
		8 => cards_with_ranks(ctx, pos_slice, e3),
		9 => cards_with_ranks(ctx, pos_slice, e2),
		10 => cards_with_ranks(ctx, pos_slice, high(e2)),
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
	let scoring = sc | if ctx.splash { bits } else { bits & ctx.stone };
	SelEval {
		hand_type,
		scoring,
		contains,
	}
}

#[derive(Clone, Copy)]
pub struct CoreCtx {
	pub nh: usize,
	pub rank: [i8; MAX_HAND],
	pub suit: [u8; MAX_HAND],
	pub forced: u32,
	pub ff: bool,
	pub sc: bool,
	pub block: bool,
	pub lo: usize,
	pub hi: usize,
	pub cnt: [[u8; 13]; 5],
	pub scnt: [u8; 4],
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
			block: false,
			lo: 1,
			hi: 5,
			cnt: [[0; 13]; 5],
			scnt: [0; 4],
		}
	}
}

#[must_use]
pub fn core_ctx(g: &Game, hide: bool) -> CoreCtx {
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
			i8::try_from(id - 2).unwrap_or(-1)
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
		if cd.forced_selection {
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
	x.block = false;
	let lo = if g.blind.active(BlindKind::Psychic) {
		5usize
	} else {
		1usize
	};
	let hi = 5usize;
	let forced_cnt = usize::try_from(x.forced.count_ones()).unwrap_or(0);
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

	fn eval_straight(&mut self, x: &CoreCtx, chans: Range<usize>) {
		let mode = (usize::from(x.ff) << 1) | usize::from(x.sc);
		let mins = &straight_mins().m[mode];
		for c in chans {
			let (mut pres, mut have) = (0u16, 0u16);
			for v in 0..13 {
				if self.cp[c][v] > 0 {
					pres |= 1u16 << v;
				}
				if x.cnt[c][v] > 0 {
					have |= 1u16 << v;
				}
			}
			for &m in mins {
				if (m & !have) != 0 {
					continue;
				}
				let d = i32::try_from((m & !pres).count_ones()).unwrap_or(CORE_INF);
				self.cost = self.cost.min(d);
				if d <= self.rem_budget {
					self.use_mask[c] |= m & !pres;
				}
			}
		}
	}
}

/// Returns `(legal_next_cards_mask, can_stop)` for declared hand `hand` (`0..=11`,
/// or `HS_ANY = 12`) and current selection bitmask `prefix` (forced cards included).
#[must_use]
pub fn core_rule(ctx: &CoreCtx, hand: usize, boss: bool, prefix: u32) -> (u32, bool) {
	let budget = num::id(if boss { ctx.hi } else { 5 });
	let lo = num::id(if boss { ctx.lo } else { 1 });
	let num_sel = i32::try_from(prefix.count_ones()).unwrap_or(0);
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
		let idx = usize::try_from(cur.trailing_zeros()).unwrap_or(0);
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
		3 => sc.eval_straight(ctx, 1..5),
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
		7 => sc.eval_straight(ctx, 0..1),
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
		let idx = usize::try_from(rem.trailing_zeros()).unwrap_or(0);
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
