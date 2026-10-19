//! Emulation of `G.E_MANAGER` and of `Game:update`'s per-state handlers.
//!
//! The game's flows are chains of events on a FIFO queue; events created while
//! an event runs are appended at the tail, and the per-state `update_*`
//! handlers run after the event that changed the state. The queue is stepped
//! frame by frame (`dt = 1/60`, `G.TIMERS.TOTAL` rebased to 0 per action)
//! with the game's `trigger` / `delay` / `blocking` / `blockable` semantics,
//! so event order and frame timing match the real game. Purely visual events
//! are kept as timing-only placeholders (`Ev::Wait`).

use crate::card::Kind;
use crate::data::*;
use crate::game::{Area, CardSet, Game};

/// `G.STATE`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum St {
    SelectingHand,
    HandPlayed,
    DrawToHand,
    GameOver,
    Shop,
    PlayTarot,
    BlindSelect,
    RoundEval,
    TarotPack,
    PlanetPack,
    SpectralPack,
    StandardPack,
    BuffoonPack,
    NewRound,
}

impl St {
    pub fn is_pack(self) -> bool {
        matches!(self, St::TarotPack | St::PlanetPack | St::SpectralPack | St::StandardPack | St::BuffoonPack)
    }
    pub fn name(self) -> &'static str {
        match self {
            St::SelectingHand => "SELECTING_HAND",
            St::HandPlayed => "HAND_PLAYED",
            St::DrawToHand => "DRAW_TO_HAND",
            St::GameOver => "GAME_OVER",
            St::Shop => "SHOP",
            St::PlayTarot => "PLAY_TAROT",
            St::BlindSelect => "BLIND_SELECT",
            St::RoundEval => "ROUND_EVAL",
            St::TarotPack => "TAROT_PACK",
            St::PlanetPack => "PLANET_PACK",
            St::SpectralPack => "SPECTRAL_PACK",
            St::StandardPack => "STANDARD_PACK",
            St::BuffoonPack => "BUFFOON_PACK",
            St::NewRound => "NEW_ROUND",
        }
    }
}

/// Card piles addressable by events.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Pile {
    Deck,
    Hand,
    Play,
    Discard,
}

impl Pile {
    /// Timeline spelling (`trace-keys` builds).
    #[cfg(feature = "trace-keys")]
    pub(crate) fn name(self) -> &'static str {
        match self {
            Pile::Deck => "deck",
            Pile::Hand => "hand",
            Pile::Play => "play",
            Pile::Discard => "discard",
        }
    }
}

/// Game speed (`G.SETTINGS.GAMESPEED`) the timing model assumes (the
/// game's default).
pub(crate) const GAMESPEED: f64 = 1.0;
/// Real seconds per frame (60 fps).
pub(crate) const DT: f64 = 1.0 / 60.0;

/// `G.TIMERS.TOTAL` after `n` frames at speed factor 1, accumulated frame by
/// frame exactly like the game does.
fn totals() -> &'static [f64] {
    static T: std::sync::OnceLock<Box<[f64]>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut v = Vec::with_capacity(1 << 16);
        let mut t = 0.0f64;
        v.push(t);
        for _ in 1..(1 << 16) {
            t += DT;
            v.push(t);
        }
        v.into_boxed_slice()
    })
}

/// An event's `trigger` and `delay` (in `G.TIMERS.TOTAL` seconds).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Trig {
    /// `'immediate'`.
    Imm,
    /// `'after'`: runs once `delay` has elapsed since first handled.
    After(f64),
    /// `'before'`: runs when first handled, completes after `delay`.
    Before(f64),
    /// `'ease'`: completes once strictly more than `delay` has elapsed.
    Ease(f64),
    /// An immediate event polling `TOTAL - start > d` (`Card:explode`).
    Until(f64, f64),
    /// An immediate event polling a UI condition that holds `n` frames
    /// after its first call (measured; UI motion is not simulated).
    Frames(u32),
}

/// An entry of `G.E_MANAGER.queues.base`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Queued {
    pub ev: Ev,
    pub trig: Trig,
    pub blocking: bool,
    pub blockable: bool,
    /// `Event.time` (set when first handled; NaN before).
    pub time: f64,
    /// `Event.complete`.
    pub done: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Ev {
    /// A purely visual event (kept for its timing).
    Wait,
    /// An event whose body queues `update_hand_text({immediate = true, delay = 0})`.
    PushHandTextImm,
    /// An event whose body calls `start_materialize` (timing only).
    Materialize,
    /// `G.GAME.chips` reaching its eased target.
    SetChips(f64),
    /// end_round's non-blocking poller calling `win_game()` in ROUND_EVAL.
    WinPoll,
    /// `win_game()`'s event: pauses the game behind the win overlay (the
    /// oracle, like a player choosing Endless, dismisses it at once).
    WinPause,
    /// `ease_chips(0)`'s outer event.
    EaseChips0,
    /// `G.FUNCS.reroll_shop`'s 'after' event queueing the joker pass.
    RerollOuter,
    /// The pack UI being removed (with the unchosen pack cards).
    PackRemove,
    /// An event whose body calls `card_eval_status_text` (base delay).
    StatusLater(f64),
    /// An event whose body calls `ease_dollars(d)` + a status text.
    DollarsStatus { d: f64, delay: f64 },
    // --- generic -----------------------------------------------------
    EaseDollars(f64),
    EaseDiscard(f64),
    EaseHands(f64),
    EaseRound,
    EaseAnte(i32),
    DollarBufferReset,
    HandSize(f64),
    RefreshDebuffs,
    SetCostAll,
    /// `create_card` + `add_to_deck` + `emplace` (+ buffer reset).
    Create { set: CardSet, area: Area, forced: Option<Kind>, append: &'static str, rarity: Option<f64>, legendary: bool },
    /// Joker eats/melts itself: an event that queues the removal.
    EatenOuter { id: u32 },
    RemoveJokerLater { id: u32 },
    /// An event whose body calls `start_dissolve` on a joker.
    DissolveJoker { id: u32, fac: f64 },
    /// The event that destroys a scored card (`shatter`/`start_dissolve`).
    DissolvePlaying { id: u32 },
    /// Removal at the end of a dissolve/shatter/explode animation. Those
    /// animations queue blocking-but-unblockable events that hold back every
    /// event queued after them, so removal is modelled as an event queued at
    /// the moment the animation starts.
    RemoveCard { id: u32 },
    /// `set_blind(nil)` at the end of `Blind:defeat`'s animation.
    BlindReset,
    DrawCard { from: Pile, to: Pile, card: Option<u32>, sort: bool, discarded_only: bool },
    AddTag { kind: TagKind, orbital: Option<HandType> },

    // --- blind select / round ----------------------------------------
    BlindSelectInit,
    BlindSelectUi,
    SelectBlind { blind: BlindKind },
    NewRound,
    NewRoundBody,
    NewRoundDraw,
    /// `Blind:set_blind`'s HUD event (non-blockable, queues a second one).
    BlindSetHud,
    DrawToHand,
    EnterSelecting,
    PlayStart,
    PlayChain,
    EvaluatePlay,
    AfterPlay,
    StateCompleteFalse,
    HandPlayedCheck,
    HookDiscard,
    ToothOuter,
    EndRound,
    EndRoundFinal,
    RoundEvalInit,
    RoundEvalRun,
    CashOutFinish,
    BlindDisableCheck,
    AcornOuter,
    AcornShuffle,
    SkipBlindRun,
    RerollBossRun,

    // --- jokers --------------------------------------------------------
    ChicotOuter,
    ChicotDisable,
    Burglar,
    RiffRaff { n: u32 },
    CartomancerOuter,
    DaggerSlice { dagger: u32, victim: u32 },
    MarbleStone,
    CertificateCard,
    GlassJokerOuter { id: u32, n: u32 },
    GlassJokerInner { id: u32, n: u32 },
    PerkeoCopy,
    /// Vampire's juice event, which clears the card's `vampired` mark.
    Unvampire { id: u32 },

    // --- tags ----------------------------------------------------------
    TagFunc { which: TagFn },
    TagRemove { tag_id: u32 },

    // --- shop ----------------------------------------------------------
    ShopOuter,
    ShopPopulate { shop_exists: bool },
    StoreModify { card: u32 },
    RerollShopRun,
    RerollShopJokers,
    BuyRun { card: u32, buy_and_use: bool },
    SellRun { card: u32, area: Area },
    VerdantDisable,
    ToggleShopRun,

    // --- packs / consumables ---------------------------------------------
    OpenPack { card: u32 },
    PackCreate,
    PackStateUi,
    PackDrawHand,
    UseFinishOuter { prev: St, area: Option<Area>, card: u32, dissolve: bool, delay_fac: f64 },
    UseFinishInner { prev: St, area: Option<Area>, delay_fac: f64 },
    EndConsumeable { df: f64 },
    EndConsumeableInner,
    Consumable(crate::consumables::CEv),
    LastTarotPlanetOuter { kind: Kind },
    LastTarotPlanetInner { kind: Kind },
}

/// The body of a tag's `yep(..., func)` callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TagFn {
    TopUp,
    Charm,
    Meteor,
    Ethereal,
    Standard,
    Buffoon,
    Boss,
    Voucher,
    Double { kind: TagKind, orbital: Option<HandType> },
    RareCard { card: u32 },
    UncommonCard { card: u32 },
    Edition { card: u32, edition: Edition },
    D6,
    Coupon,
    Economy,
    Noop,
}

impl Game {
    /// `G.E_MANAGER:add_event(Event({func = ...}))` (immediate, blocking).
    #[inline]
    pub(crate) fn push(&mut self, e: Ev) {
        self.push_ev(e, Trig::Imm, true, true);
    }

    #[inline]
    pub(crate) fn push_ev(&mut self, ev: Ev, trig: Trig, blocking: bool, blockable: bool) {
        if !blockable {
            self.unblockable += 1;
        }
        self.queue.push_back(Queued { ev, trig, blocking, blockable, time: f64::NAN, done: false });
    }

    /// `draw_card(from, to, ..., delay)`: a `'before'` event.
    #[inline]
    pub(crate) fn draw_ev(&mut self, from: Pile, to: Pile, card: Option<u32>, sort: bool, discarded_only: bool, delay: f64) {
        self.push_before(delay, Ev::DrawCard { from, to, card, sort, discarded_only });
    }

    #[inline]
    pub(crate) fn push_after(&mut self, d: f64, e: Ev) {
        self.push_ev(e, Trig::After(d), true, true);
    }

    #[inline]
    pub(crate) fn push_before(&mut self, d: f64, e: Ev) {
        self.push_ev(e, Trig::Before(d), true, true);
    }

    /// `delay(t)`.
    #[inline]
    pub(crate) fn delay(&mut self, d: f64) {
        self.push_ev(Ev::Wait, Trig::After(d), true, true);
    }

    /// A blocking event with no game effect (`trigger`, `delay`).
    #[inline]
    pub(crate) fn wait(&mut self, t: Trig) {
        self.push_ev(Ev::Wait, t, true, true);
    }

    /// A non-blockable (but blocking) event with no game effect: holds back
    /// every event queued after it.
    #[inline]
    pub(crate) fn barrier(&mut self, t: Trig) {
        self.push_ev(Ev::Wait, t, true, false);
    }

    /// `update_hand_text({delay = d}, ...)`.
    #[inline]
    pub(crate) fn hand_text(&mut self, d: f64) {
        self.wait(Trig::Before(d));
    }

    /// `update_hand_text({immediate = true, delay = d}, ...)` (non-blockable).
    #[inline]
    pub(crate) fn hand_text_imm(&mut self, d: f64) {
        self.barrier(Trig::Before(d));
    }

    /// `highlight_card(card, ...)`.
    #[inline]
    pub(crate) fn highlight_ev(&mut self) {
        self.wait(Trig::Before(0.1));
    }

    /// The events `Card:set_edition(edition, immediate)` queues (not silent).
    pub(crate) fn edition_fx(&mut self, e: Edition, immediate: bool) {
        if e == Edition::None {
            return;
        }
        if immediate {
            self.push_ev(Ev::Wait, Trig::After(0.0), true, false);
        } else {
            self.push_after(0.2, Ev::Wait);
        }
        self.push_after(0.1, Ev::Wait);
    }

    /// `level_up_hand(card, hand, instant, amount)` including its events.
    pub(crate) fn level_up_hand_ev(&mut self, h: HandType, amount: f64, instant: bool) {
        self.level_up_hand(h, amount);
        if !instant {
            self.push_after(0.2, Ev::Wait);
            self.hand_text(0.0);
            self.push_after(0.9, Ev::Wait);
            self.hand_text(0.0);
            self.push_after(0.9, Ev::Wait);
            self.hand_text(0.0);
            self.delay(1.3);
        }
        self.push(Ev::Wait);
    }

    /// `card_eval_status_text(card, ...)` with an explicit base `delay`: a
    /// `'before'` event of `delay*1.25`.
    #[inline]
    pub(crate) fn status_text(&mut self, delay: f64) {
        self.wait(Trig::Before(delay * 1.25));
    }

    /// `card_eval_status_text(card, 'extra', ...)` without `extra.delay`.
    #[inline]
    pub(crate) fn status_extra(&mut self) {
        self.status_text(0.75);
    }

    /// `Card:start_dissolve(colours, silent, dissolve_time_fac)`.
    pub(crate) fn start_dissolve_f(&mut self, id: u32, fac: f64) {
        let dt = 0.7 * fac;
        self.push_ev(Ev::RemoveCard { id }, Trig::After(1.05 * dt), true, false);
        self.barrier(Trig::After(1.051 * dt));
    }

    #[inline]
    pub(crate) fn start_dissolve(&mut self, id: u32) {
        self.start_dissolve_f(id, 1.0);
    }

    /// `Card:shatter()`.
    pub(crate) fn shatter(&mut self, id: u32) {
        let dt = 0.7;
        self.push_ev(Ev::RemoveCard { id }, Trig::After(0.55 * dt), true, false);
        self.barrier(Trig::After(0.51 * dt));
    }

    /// `Card:explode()`.
    pub(crate) fn explode(&mut self, id: u32) {
        let et = 1.3 * GAMESPEED.sqrt();
        self.barrier(Trig::Until(self.total, 1.5 * et));
        self.push_ev(Ev::RemoveCard { id }, Trig::After(1.5 * et), true, false);
    }

    /// `Card:start_materialize(colours, silent, timefac)`: only its timing.
    pub(crate) fn materialize(&mut self, timefac: f64) {
        let dt = 0.6 * timefac;
        self.barrier(Trig::After(0.5 * dt));
        self.barrier(Trig::Ease(dt));
        self.barrier(Trig::After(1.05 * dt));
    }

    /// Glass cards shatter, everything else dissolves.
    pub(crate) fn destroy_playing(&mut self, id: u32) {
        let glass = self.playing_cards().any(|c| c.id == id && c.enhancement == Enhancement::Glass);
        if glass { self.shatter(id) } else { self.start_dissolve(id) }
    }

    /// One frame of `Game:update` after the timers advanced:
    /// `G.E_MANAGER:update`, then the state handlers. Returns whether
    /// anything happened.
    fn pass(&mut self) -> bool {
        let mut changed = false;
        let mut blocked = false;
        let mut seen_unblockable = 0u32;
        let mut i = 0;
        while i < self.queue.len() {
            let q = self.queue[i];
            if blocked && q.blockable {
                // Only non-blockable events can still run this frame.
                if seen_unblockable == self.unblockable {
                    break;
                }
                i += 1;
                continue;
            }
            if !q.blockable {
                seen_unblockable += 1;
            }
            let time = if q.time.is_nan() {
                changed = true;
                let t = if matches!(q.trig, Trig::Frames(_)) { self.frames as f64 } else { self.total };
                self.queue[i].time = t;
                t
            } else {
                q.time
            };
            let (completed, time_done) = match q.trig {
                Trig::Imm => {
                    changed = true;
                    (self.run_event(q.ev), true)
                }
                Trig::After(d) => {
                    if time + d <= self.total {
                        changed = true;
                        (self.run_event(q.ev), true)
                    } else {
                        (false, false)
                    }
                }
                Trig::Before(d) => {
                    let c = if q.done {
                        true
                    } else {
                        changed = true;
                        self.run_event(q.ev)
                    };
                    (c, time + d <= self.total)
                }
                Trig::Ease(d) => {
                    if time + d >= self.total {
                        (false, false)
                    } else {
                        changed = true;
                        (self.run_event(q.ev), true)
                    }
                }
                Trig::Until(start, d) => {
                    if self.total - start > d {
                        changed = true;
                        (self.run_event(q.ev), true)
                    } else {
                        (false, true)
                    }
                }
                Trig::Frames(n) => {
                    changed = true;
                    if self.frames as f64 - time >= n as f64 { (self.run_event(q.ev), true) } else { (false, true) }
                }
            };
            if q.blocking {
                blocked = true;
            }
            if self.pause_rest {
                // `G.SETTINGS.paused`: the rest of this pass is skipped.
                self.pause_rest = false;
                if completed && time_done {
                    self.queue.remove(i);
                    if !q.blockable {
                        self.unblockable -= 1;
                    }
                }
                break;
            }
            if completed && time_done {
                #[cfg(feature = "trace-keys")]
                if let Some(log) = self.rng.trace.evlog.as_mut() {
                    log.push(format!("{} {:?} {:?}{}{}", self.frames, q.ev, q.trig, if q.blocking { "" } else { " nb" }, if q.blockable { "" } else { " nbl" }));
                }
                self.queue.remove(i);
                if !q.blockable {
                    self.unblockable -= 1;
                    seen_unblockable -= 1;
                }
                changed = true;
            } else {
                if completed {
                    self.queue[i].done = true;
                }
                i += 1;
            }
        }
        let before = (self.state, self.state_complete, self.queue.len());
        self.tick_state();
        changed || before != (self.state, self.state_complete, self.queue.len())
    }

    /// `G.TIMERS.TOTAL` advance (with the `G.ACC` speed-up of long hands).
    #[inline]
    fn advance_time(&mut self) {
        self.frames += 1;
        #[cfg(feature = "trace-keys")]
        {
            self.rng.trace.frame = self.frames;
        }
        if self.state != self.acc_state {
            self.acc = 0.0;
        }
        self.acc_state = self.state;
        if matches!(self.state, St::HandPlayed | St::NewRound) {
            self.acc = (self.acc + DT * 0.2 * GAMESPEED).min(16.0);
        } else {
            self.acc = 0.0;
        }
        let sf = GAMESPEED + (self.acc.abs() - 2.0).max(0.0);
        if sf != 1.0 {
            self.on_table = false;
        }
        self.total += DT * sf;
    }

    /// Advances idle frames until `TOTAL >= due`. While every frame of the
    /// action advanced `TOTAL` by exactly `DT` (no `G.ACC` speed-up), `TOTAL`
    /// after `n` frames is `totals()[n]`, so the skip is a table lookup with
    /// bit-identical results.
    fn skip_until(&mut self, due: f64) {
        if self.on_table && !matches!(self.state, St::HandPlayed | St::NewRound) {
            let t = totals();
            let cur = self.frames as usize;
            if cur < t.len() && self.total == t[cur] {
                // First frame n > cur with t[n] >= due.
                let mut n = ((due / DT) as usize).max(cur + 1);
                while n > cur + 1 && t.get(n - 1).is_some_and(|&v| v >= due) {
                    n -= 1;
                }
                while n < t.len() && t[n] < due {
                    n += 1;
                }
                if n < t.len() {
                    self.frames = n as u32;
                    self.total = t[n];
                    self.acc = 0.0;
                    self.acc_state = self.state;
                    #[cfg(feature = "trace-keys")]
                    {
                        self.rng.trace.frame = self.frames;
                    }
                    return;
                }
            }
        }
        while self.total < due {
            self.advance_time();
        }
    }

    /// Earliest `TOTAL` at which a waiting event can progress.
    fn next_due(&self) -> f64 {
        let mut due = f64::INFINITY;
        for q in &self.queue {
            if q.time.is_nan() {
                continue;
            }
            let t = match q.trig {
                Trig::Imm => continue,
                Trig::After(d) | Trig::Before(d) | Trig::Ease(d) => q.time + d,
                Trig::Until(s, d) => s + d,
                Trig::Frames(_) => self.total,
            };
            due = due.min(t);
        }
        due
    }

    /// Every action starts at `G.TIMERS.TOTAL = 0` (the oracle rebases its
    /// timers the same way), so timer arithmetic is reproducible.
    pub(crate) fn begin_action(&mut self) {
        self.total = 0.0;
        self.frames = 0;
        self.on_table = true;
        #[cfg(feature = "trace-keys")]
        {
            let t = &mut self.rng.trace;
            t.frame = 0;
            t.entries.clear();
            if let Some(l) = t.evlog.as_mut() {
                l.clear();
            }
        }
    }

    /// Turns timeline recording on or off (a no-op without `trace-keys`).
    pub fn set_tracing(&mut self, on: bool) {
        #[cfg(feature = "trace-keys")]
        {
            self.rng.trace.enabled = on;
            if !on {
                self.rng.trace.entries = Vec::new();
            }
        }
        #[cfg(not(feature = "trace-keys"))]
        let _ = on;
    }

    /// Enables the completed-event log (`trace-keys` builds).
    #[cfg(feature = "trace-keys")]
    pub fn enable_evlog(&mut self) {
        self.rng.trace.evlog = Some(Vec::new());
    }

    /// Completed events of the current action (`trace-keys` builds).
    #[cfg(feature = "trace-keys")]
    pub fn evlog(&self) -> &[String] {
        self.rng.trace.evlog.as_deref().unwrap_or(&[])
    }

    /// The current action's timeline (empty without `trace-keys`).
    pub fn timeline(&self) -> &[String] {
        #[cfg(feature = "trace-keys")]
        {
            &self.rng.trace.entries
        }
        #[cfg(not(feature = "trace-keys"))]
        {
            &[]
        }
    }

    /// Runs frames until `evaluate_play` has run (the rest of the chain is
    /// abandoned: the game is left mid-animation and must be discarded).
    pub(crate) fn run_until_scored(&mut self) {
        self.scored = false;
        let mut guard = 0u32;
        while !self.scored {
            self.advance_time();
            if !self.pass() && !self.scored {
                if self.queue.is_empty() {
                    break;
                }
                let due = self.next_due();
                if !due.is_finite() {
                    break;
                }
                self.skip_until(due);
                self.pass();
            }
            guard += 1;
            if guard > 1_000_000 {
                break;
            }
        }
    }

    /// Runs frames until the game waits for input.
    pub(crate) fn run_queue(&mut self) {
        let mut guard = 0u32;
        loop {
            self.advance_time();
            if !self.pass() {
                if self.queue.is_empty() {
                    break;
                }
                // Nothing can happen before a timer expires: skip idle
                // frames (their passes and state handlers are no-ops).
                let due = self.next_due();
                assert!(due.is_finite(), "event queue stalled");
                self.skip_until(due);
                self.pass();
            }
            guard += 1;
            assert!(guard < 100_000_000, "event loop did not terminate");
        }
        self.sync_phase();
    }

    /// Runs an event's `func`; returns its result (false = retry next frame).
    pub(crate) fn run_event(&mut self, e: Ev) -> bool {
        match e {
            Ev::EaseDollars(d) => self.ease_dollars(d),
            Ev::EaseDiscard(d) => self.ease_discard(d),
            Ev::EaseHands(d) => self.ease_hands_played(d),
            Ev::EaseRound => self.round += 1,
            Ev::EaseAnte(d) => self.ante += d,
            Ev::DollarBufferReset => self.dollar_buffer = 0.0,
            Ev::HandSize(d) => self.apply_hand_size(d),
            Ev::RefreshDebuffs => self.refresh_playing_debuffs(),
            Ev::SetCostAll => self.set_cost_all(),
            Ev::Create { set, area, forced, append, rarity, legendary } => {
                let mut c = self.create_card(set, area, legendary, rarity, false, forced, append);
                self.add_to_deck(&mut c, false);
                match area {
                    Area::Jokers => {
                        self.jokers.push(c);
                        self.joker_buffer = 0.0;
                    }
                    _ => {
                        self.consumables.push(c);
                        self.consumeable_buffer = 0.0;
                    }
                }
            }
            Ev::Wait => {}
            Ev::PushHandTextImm => self.hand_text_imm(0.0),
            Ev::Materialize => self.materialize(1.0),
            Ev::SetChips(c) => self.chips = c,
            Ev::WinPoll => {
                if self.state != St::RoundEval {
                    return false;
                }
                self.won = true;
                self.push(Ev::WinPause);
                // round_text = 'Endless Round ' once unpaused
                self.push(Ev::Wait);
            }
            Ev::WinPause => self.pause_rest = true,
            Ev::EaseChips0 => self.push_ev(Ev::SetChips(0.0), Trig::Ease(0.3), true, false),
            Ev::RerollOuter => self.push(Ev::RerollShopJokers),
            Ev::StatusLater(d) => self.status_text(d),
            Ev::DollarsStatus { d, delay } => {
                self.push(Ev::EaseDollars(d));
                self.status_text(delay);
            }
            Ev::PackRemove => {
                let cards = std::mem::take(&mut self.pack_cards);
                for c in cards {
                    self.drop_unowned(c);
                }
            }
            Ev::EatenOuter { id } => self.push_ev(Ev::RemoveJokerLater { id }, Trig::After(0.3), true, false),
            Ev::RemoveCard { id } => self.remove_card_by_id(id),
            Ev::DissolvePlaying { id } => self.destroy_playing(id),
            Ev::BlindReset => self.set_blind(None),
            Ev::RemoveJokerLater { id } => {
                if let Some(i) = self.jokers.iter().position(|c| c.id == id) {
                    let c = self.jokers.remove(i);
                    self.remove_owned(c);
                }
            }
            Ev::DissolveJoker { id, fac } => self.start_dissolve_f(id, fac),
            Ev::DrawCard { from, to, card, sort, discarded_only } => self.draw_card(from, to, card, sort, discarded_only),
            Ev::AddTag { kind, orbital } => self.add_tag(kind, orbital),
            Ev::Consumable(c) => self.run_consumable_event(c),
            Ev::LastTarotPlanetOuter { kind } => self.push(Ev::LastTarotPlanetInner { kind }),
            Ev::LastTarotPlanetInner { kind } => self.last_tarot_planet = Some(kind),
            Ev::Unvampire { id } => {
                if let Some(c) = self.playing_cards_mut().find(|c| c.id == id) {
                    c.vampired = false;
                }
            }
            Ev::TagFunc { .. } | Ev::TagRemove { .. } => self.run_tag_event(e),
            Ev::BlindSelectInit
            | Ev::BlindSelectUi
            | Ev::SelectBlind { .. }
            | Ev::NewRound
            | Ev::NewRoundBody
            | Ev::NewRoundDraw
            | Ev::BlindSetHud
            | Ev::DrawToHand
            | Ev::EnterSelecting
            | Ev::PlayStart
            | Ev::PlayChain
            | Ev::EvaluatePlay
            | Ev::AfterPlay
            | Ev::StateCompleteFalse
            | Ev::HandPlayedCheck
            | Ev::HookDiscard
            | Ev::ToothOuter
            | Ev::EndRound
            | Ev::EndRoundFinal
            | Ev::RoundEvalInit
            | Ev::RoundEvalRun
            | Ev::CashOutFinish
            | Ev::BlindDisableCheck
            | Ev::AcornOuter
            | Ev::AcornShuffle
            | Ev::SkipBlindRun
            | Ev::RerollBossRun
            | Ev::ChicotOuter
            | Ev::ChicotDisable
            | Ev::Burglar
            | Ev::RiffRaff { .. }
            | Ev::CartomancerOuter
            | Ev::DaggerSlice { .. }
            | Ev::MarbleStone
            | Ev::CertificateCard
            | Ev::GlassJokerOuter { .. }
            | Ev::GlassJokerInner { .. }
            | Ev::PerkeoCopy => self.run_flow_event(e),
            Ev::ShopOuter
            | Ev::ShopPopulate { .. }
            | Ev::StoreModify { .. }
            | Ev::RerollShopRun
            | Ev::RerollShopJokers
            | Ev::BuyRun { .. }
            | Ev::SellRun { .. }
            | Ev::VerdantDisable
            | Ev::ToggleShopRun
            | Ev::OpenPack { .. }
            | Ev::PackCreate
            | Ev::PackStateUi
            | Ev::PackDrawHand
            | Ev::UseFinishOuter { .. }
            | Ev::UseFinishInner { .. }
            | Ev::EndConsumeable { .. }
            | Ev::EndConsumeableInner => self.run_shop_event(e),
        }
        true
    }

    /// `Game:update`'s state dispatch (after each event-queue pass).
    pub(crate) fn tick_state(&mut self) {
        // A handler can move to another state; keep dispatching until stable
        // (no transition chain is longer than a few states).
        for _ in 0..8 {
            let before = (self.state, self.state_complete);
            match self.state {
                St::SelectingHand => {
                    if self.hand.is_empty() && !self.deck.is_empty() {
                        self.state = St::DrawToHand;
                        self.state_complete = false;
                    } else {
                        self.update_selecting_hand();
                    }
                }
                St::Shop => self.update_shop(),
                St::HandPlayed => self.update_hand_played(),
                St::DrawToHand => self.update_draw_to_hand(),
                St::NewRound => self.update_new_round(),
                St::BlindSelect => self.update_blind_select(),
                St::RoundEval => self.update_round_eval(),
                St::TarotPack | St::PlanetPack | St::SpectralPack | St::StandardPack | St::BuffoonPack => self.update_pack(),
                St::GameOver => self.state_complete = true,
                St::PlayTarot => {}
            }
            if (self.state, self.state_complete) == before {
                break;
            }
        }
    }

    /// Body of the `change_size` event.
    pub(crate) fn apply_hand_size(&mut self, delta: f64) {
        self.hand_limit.change(delta);
        if delta > 0.0
            && self.hand_limit.real > 1.0
            && !self.hand.is_empty()
            && matches!(self.state, St::DrawToHand | St::SelectingHand)
        {
            for _ in 0..(delta.abs() as usize) {
                self.draw_ev(Pile::Deck, Pile::Hand, None, false, false, 0.07);
                self.push(Ev::Consumable(crate::consumables::CEv::SortHand));
            }
        }
    }

}
