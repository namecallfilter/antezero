//! Joker contexts of hand scoring: before, individual cards, repetitions,
//! the joker pass (`main` / other jokers), after, and card destruction.

use super::*;
use crate::rng::Key;

impl Game {
    pub(super) fn joker_scoring(&mut self, ji: usize, j: Joker, ctx: &Ctx, sc: Option<&Scoring>, blueprinted: bool) -> Option<Effect> {
        use Joker as J;
        let ab = self.jokers[ji].ab;
        let prob = self.probabilities;
        match *ctx {
            Ctx::DestroyingCard { play_idx, full_len } => {
                if blueprinted {
                    return None;
                }
                if j == J::SixthSense
                    && full_len == 1
                    && self.play[play_idx].id_is(6)
                    && self.hands_played_round == 0.0
                {
                    if self.consumable_room() {
                        self.consumeable_buffer += 1.0;
                        self.push_before(0.0, Ev::Create { set: CardSet::Spectral, area: Area::Consumables, forced: None, append: "sixth", rarity: None, legendary: false });
                        self.status_extra();
                    }
                    ret!(remove: true);
                }
                None
            }
            Ctx::DebuffedHand => {
                if j == J::Matador && self.blind.triggered {
                    self.push(Ev::EaseDollars(8.0));
                    self.buffer_dollars(8.0);
                    ret!(dollars: Some(8.0));
                }
                None
            }
            Ctx::Individual { area: CardArea::Play, other } => {
                let sc = sc.expect("individual context without scoring");
                let par = self.pareidolia();
                let smeared = self.smeared();
                let oc = &self.play[other];
                if j == J::Hiker {
                    self.play[other].ab.perma_bonus += 5.0;
                    ret!(extra: true);
                }
                if j == J::LuckyCat && oc.lucky_trigger && !blueprinted {
                    self.jab(ji).x_mult += 0.25;
                    ret!(extra: true);
                }
                if j == J::Wee && oc.id_is(2) && !blueprinted {
                    self.jab(ji).chips += 8.0;
                    ret!(extra: true);
                }
                if j == J::Photograph {
                    let first_face = sc.scoring().iter().map(|&i| i as usize).find(|&i| self.play[i].is_face(false, par));
                    if Some(other) == first_face {
                        ret!(x_mult: Some(2.0));
                    }
                }
                if j == J::J8Ball && self.consumable_room() && self.play[other].id_is(8) {
                    let k = Key::from("8ball");
                    if self.rng.random(&k) < prob / 4.0 {
                        self.consumeable_buffer += 1.0;
                        ret!(extra: true, eight_ball: true);
                    }
                }
                let oc = &self.play[other];
                if j == J::Idol && oc.get_id() == Some(self.idol_card.0) && oc.is_suit(self.idol_card.1, false, false, smeared) {
                    ret!(x_mult: Some(2.0));
                }
                if j == J::ScaryFace && oc.is_face(false, par) {
                    ret!(chips: Some(30.0));
                }
                if j == J::Smiley && oc.is_face(false, par) {
                    ret!(mult: Some(5.0));
                }
                if j == J::Ticket && oc.enhancement == Enhancement::Gold {
                    self.buffer_dollars(4.0);
                    ret!(dollars: Some(4.0));
                }
                if j == J::Scholar && oc.id_is(ACE) {
                    ret!(chips: Some(20.0), mult: Some(4.0));
                }
                if j == J::WalkieTalkie && (oc.id_is(10) || oc.id_is(4)) {
                    ret!(chips: Some(10.0), mult: Some(4.0));
                }
                if j == J::Business && oc.is_face(false, par) {
                    let k = Key::from("business");
                    if self.rng.random(&k) < prob / 2.0 {
                        self.buffer_dollars(2.0);
                        ret!(dollars: Some(2.0));
                    }
                }
                let oc = &self.play[other];
                if j == J::Fibonacci && matches!(oc.get_id(), Some(2 | 3 | 5 | 8 | ACE)) {
                    ret!(mult: Some(8.0));
                }
                if j == J::EvenSteven && matches!(oc.get_id(), Some(r) if r <= 10 && r % 2 == 0) {
                    ret!(mult: Some(4.0));
                }
                if j == J::OddTodd && matches!(oc.get_id(), Some(r) if (r <= 10 && r % 2 == 1) || r == ACE) {
                    ret!(chips: Some(31.0));
                }
                let suit_mult = match j {
                    J::GreedyJoker => Some(Suit::Diamonds),
                    J::LustyJoker => Some(Suit::Hearts),
                    J::WrathfulJoker => Some(Suit::Spades),
                    J::GluttenousJoker => Some(Suit::Clubs),
                    _ => None,
                };
                if let Some(s) = suit_mult && oc.is_suit(s, false, false, smeared) {
                    ret!(mult: Some(3.0));
                }
                if j == J::RoughGem && oc.is_suit(Suit::Diamonds, false, false, smeared) {
                    self.buffer_dollars(1.0);
                    ret!(dollars: Some(1.0));
                }
                let oc = &self.play[other];
                if j == J::OnyxAgate && oc.is_suit(Suit::Clubs, false, false, smeared) {
                    ret!(mult: Some(7.0));
                }
                if j == J::Arrowhead && oc.is_suit(Suit::Spades, false, false, smeared) {
                    ret!(chips: Some(50.0));
                }
                if j == J::Bloodstone && oc.is_suit(Suit::Hearts, false, false, smeared) {
                    let k = Key::from("bloodstone");
                    if self.rng.random(&k) < prob / 2.0 {
                        ret!(x_mult: Some(1.5));
                    }
                }
                let oc = &self.play[other];
                if j == J::Ancient && self.ancient_suit.is_some_and(|s| oc.is_suit(s, false, false, smeared)) {
                    ret!(x_mult: Some(1.5));
                }
                if j == J::Triboulet && (oc.id_is(QUEEN) || oc.id_is(KING)) {
                    ret!(x_mult: Some(2.0));
                }
                None
            }
            Ctx::Individual { area: CardArea::Hand, other } => {
                let par = self.pareidolia();
                let oc = &self.hand[other];
                if j == J::ShootTheMoon && oc.id_is(QUEEN) {
                    if oc.debuff {
                        ret!(extra: true);
                    } else {
                        ret!(h_mult: Some(13.0));
                    }
                }
                if j == J::Baron && oc.id_is(KING) {
                    if oc.debuff {
                        ret!(extra: true);
                    } else {
                        ret!(x_mult: Some(1.5));
                    }
                }
                if j == J::ReservedParking && oc.is_face(false, par) {
                    let k = Key::from("parking");
                    if self.rng.random(&k) < prob / 2.0 {
                        if self.hand[other].debuff {
                            ret!(extra: true);
                        } else {
                            self.buffer_dollars(1.0);
                            ret!(dollars: Some(1.0));
                        }
                    }
                }
                if j == J::RaisedFist {
                    let mut temp_mult = 15.0;
                    let mut temp_id: Rank = 15;
                    let mut raised = None;
                    for (i, c) in self.hand.iter().enumerate() {
                        if temp_id >= c.rank && !c.is_stone() {
                            temp_mult = rank_nominal(c.rank);
                            temp_id = c.rank;
                            raised = Some(i);
                        }
                    }
                    if raised == Some(other) {
                        if self.hand[other].debuff {
                            ret!(extra: true);
                        } else {
                            ret!(h_mult: Some(2.0 * temp_mult));
                        }
                    }
                }
                None
            }
            Ctx::Repetition { area: CardArea::Play, other, .. } => {
                let sc = sc.expect("repetition without scoring");
                let par = self.pareidolia();
                let oc = &self.play[other];
                if j == J::SockAndBuskin && oc.is_face(false, par) {
                    ret!(repetitions: 1);
                }
                if j == J::HangingChad && sc.scoring().first().map(|&i| i as usize) == Some(other) {
                    ret!(repetitions: 2);
                }
                if j == J::Dusk && self.hands_left == 0.0 {
                    ret!(repetitions: 1);
                }
                if j == J::Selzer {
                    ret!(repetitions: 1);
                }
                if j == J::Hack && matches!(oc.get_id(), Some(2..=5)) {
                    ret!(repetitions: 1);
                }
                None
            }
            Ctx::Repetition { area: CardArea::Hand, effects, .. } => {
                if j == J::Mime && effects {
                    ret!(repetitions: 1);
                }
                None
            }
            Ctx::OtherJoker { other } => {
                if let (J::Baseball, Some(other)) = (j, other)
                    && other != ji
                    && let Some(oj) = self.jokers[other].joker()
                    && oj.rarity() == 2
                {
                    self.push(Ev::Wait);
                    ret!(xmult_mod: Some(1.5));
                }
                None
            }
            Ctx::Before => {
                let sc = sc.expect("before without scoring");
                self.joker_before(ji, j, sc, blueprinted)
            }
            Ctx::After => {
                if j == J::IceCream && !blueprinted {
                    if ab.chips - 5.0 <= 0.0 {
                        self.schedule_self_destruct(ji);
                        ret!(extra: true);
                    } else {
                        self.jab(ji).chips -= 5.0;
                        ret!(extra: true);
                    }
                }
                if j == J::Selzer && !blueprinted {
                    if ab.remaining - 1.0 <= 0.0 {
                        self.schedule_self_destruct(ji);
                        ret!(extra: true);
                    } else {
                        self.jab(ji).remaining -= 1.0;
                        ret!(extra: true);
                    }
                }
                None
            }
            Ctx::Main => {
                let sc = sc.expect("main without scoring");
                self.joker_main(ji, j, sc)
            }
            _ => unreachable!("lifecycle context {ctx:?}"),
        }
    }

    fn joker_before(&mut self, ji: usize, j: Joker, sc: &Scoring, blueprinted: bool) -> Option<Effect> {
        use Joker as J;
        let prob = self.probabilities;
        let ab = self.jokers[ji].ab;
        let hands = sc.hands;
        if j == J::Trousers && (hands.has(HandType::TwoPair) || hands.has(HandType::FullHouse)) && !blueprinted {
            self.jab(ji).mult += 2.0;
            ret!(extra: true);
        }
        if j == J::Space {
            let k = Key::from("space");
            if self.rng.random(&k) < prob / 4.0 {
                ret!(level_up: true);
            }
        }
        if j == J::Square && self.play.len() == 4 && !blueprinted {
            self.jab(ji).chips += 4.0;
            ret!(extra: true);
        }
        if j == J::Runner && hands.has(HandType::Straight) && !blueprinted {
            self.jab(ji).chips += 15.0;
            ret!(extra: true);
        }
        if j == J::MidasMask && !blueprinted {
            let par = self.pareidolia();
            let mut any = false;
            for &i in sc.scoring() {
                let i = i as usize;
                if self.play[i].is_face(false, par) {
                    any = true;
                    let mut c = self.play[i].clone();
                    self.set_enhancement(&mut c, Enhancement::Gold);
                    self.play[i] = c;
                    self.push(Ev::Wait);
                }
            }
            if any {
                ret!(extra: true);
            }
        }
        if j == J::Vampire && !blueprinted {
            let mut n = 0;
            for &i in sc.scoring() {
                let i = i as usize;
                let c = &self.play[i];
                if c.enhancement != Enhancement::Base && !c.debuff && !c.vampired {
                    n += 1;
                    let mut c = self.play[i].clone();
                    c.vampired = true;
                    self.set_enhancement(&mut c, Enhancement::Base);
                    let id = c.id;
                    self.play[i] = c;
                    self.push(Ev::Unvampire { id });
                }
            }
            if n > 0 {
                self.jab(ji).x_mult += 0.1 * n as f64;
                ret!(extra: true);
            }
        }
        if j == J::TodoList && Some(sc.name) == ab.to_do {
            self.push(Ev::EaseDollars(4.0));
            self.buffer_dollars(4.0);
            ret!(dollars: Some(4.0));
        }
        if j == J::Dna && self.hands_played_round == 0.0 && self.play.len() == 1 {
            let src = self.play[0].clone();
            let mut c = self.copy_card(&src, false);
            self.add_to_deck(&mut c, false);
            self.hand.push(c);
            self.push(Ev::Materialize);
            ret!(playing_cards_created: true, extra: true);
        }
        if j == J::RideTheBus && !blueprinted {
            let par = self.pareidolia();
            let faces = sc.scoring().iter().any(|&i| self.play[i as usize].is_face(false, par));
            if faces {
                let last = ab.mult;
                self.jab(ji).mult = 0.0;
                if last > 0.0 {
                    ret!(extra: true);
                }
            } else {
                self.jab(ji).mult += 1.0;
            }
        }
        if j == J::Obelisk && !blueprinted {
            let mut reset = true;
            let more_than = self.hands[sc.name as usize].played;
            for h in HandType::PAIRS_ORDER {
                let hl = &self.hands[h as usize];
                if h != sc.name && hl.played >= more_than && hl.visible {
                    reset = false;
                }
            }
            if reset {
                if ab.x_mult > 1.0 {
                    self.jab(ji).x_mult = 1.0;
                    ret!(extra: true);
                }
            } else {
                self.jab(ji).x_mult += 0.2;
            }
        }
        if j == J::GreenJoker && !blueprinted {
            self.jab(ji).mult += 1.0;
            ret!(extra: true);
        }
        None
    }

    fn joker_main(&mut self, ji: usize, j: Joker, sc: &Scoring) -> Option<Effect> {
        use Joker as J;
        let ab = self.jokers[ji].ab;
        let hands = sc.hands;
        let has_type = |h: Option<HandType>| -> bool {
            match h {
                None => true,
                Some(h) => hands.has(h),
            }
        };
        let jtype = joker_hand_type(j);
        if j == J::LoyaltyCard {
            let every = 5.0;
            let rem = crate::rng::lua_mod(every - 1.0 - (self.hands_played - ab.hands_played_at_create), every + 1.0);
            self.jab(ji).loyalty_remaining = rem;
            if rem == every {
                ret!(xmult_mod: Some(4.0));
            }
        }
        let ab = self.jokers[ji].ab;
        if j != J::SeeingDouble && ab.x_mult > 1.0 && has_type(jtype) {
            ret!(xmult_mod: Some(ab.x_mult));
        }
        let (t_mult, t_chips) = joker_t_values(j);
        if t_mult > 0.0 && has_type(jtype) {
            ret!(mult_mod: Some(t_mult));
        }
        if t_chips > 0.0 && has_type(jtype) {
            ret!(chip_mod: Some(t_chips));
        }
        if j == J::Half && self.play.len() <= 3 {
            ret!(mult_mod: Some(20.0));
        }
        if j == J::Abstract {
            let x = self.jokers.iter().filter(|c| matches!(c.kind, Kind::Joker(_))).count() as f64;
            ret!(mult_mod: Some(x * 3.0));
        }
        if j == J::Acrobat && self.hands_left == 0.0 {
            ret!(xmult_mod: Some(3.0));
        }
        if j == J::MysticSummit && self.discards_left == 0.0 {
            ret!(mult_mod: Some(15.0));
        }
        if j == J::Misprint {
            let k = Key::from("misprint");
            let m = self.rng.random_range(&k, 0.0, 23.0);
            ret!(mult_mod: Some(m));
        }
        if j == J::Banner && self.discards_left > 0.0 {
            ret!(chip_mod: Some(self.discards_left * 30.0));
        }
        if j == J::Stuntman {
            ret!(chip_mod: Some(250.0));
        }
        if j == J::Matador && self.blind.triggered {
            self.push(Ev::EaseDollars(8.0));
            self.buffer_dollars(8.0);
            ret!(dollars: Some(8.0));
        }
        if j == J::Supernova {
            ret!(mult_mod: Some(self.hands[sc.name as usize].played));
        }
        if j == J::Ceremonial && ab.mult > 0.0 {
            ret!(mult_mod: Some(ab.mult));
        }
        if j == J::Vagabond && self.consumable_room() && self.dollars <= 4.0 {
            self.consumeable_buffer += 1.0;
            self.push_before(0.0, Ev::Create { set: CardSet::Tarot, area: Area::Consumables, forced: None, append: "vag", rarity: None, legendary: false });
            ret!(extra: true);
        }
        if j == J::Superposition && self.consumable_room() {
            let aces = sc.scoring().iter().filter(|&&i| self.play[i as usize].id_is(ACE)).count();
            if aces >= 1 && hands.has(HandType::Straight) {
                self.consumeable_buffer += 1.0;
                self.push_before(0.0, Ev::Create { set: CardSet::Tarot, area: Area::Consumables, forced: None, append: "sup", rarity: None, legendary: false });
                ret!(extra: true);
            }
        }
        if j == J::Seance && self.consumable_room() && hands.has(HandType::StraightFlush) {
            self.consumeable_buffer += 1.0;
            self.push_before(0.0, Ev::Create { set: CardSet::Spectral, area: Area::Consumables, forced: None, append: "sea", rarity: None, legendary: false });
            ret!(extra: true);
        }
        if j == J::FlowerPot {
            let smeared = self.smeared();
            let mut suits = [0u8; 4]; // H, D, S, C
            let order = [Suit::Hearts, Suit::Diamonds, Suit::Spades, Suit::Clubs];
            for &i in sc.scoring() {
                let c = &self.play[i as usize];
                if c.enhancement != Enhancement::Wild {
                    for (si, &s) in order.iter().enumerate() {
                        if c.is_suit(s, true, false, smeared) && suits[si] == 0 {
                            suits[si] += 1;
                            break;
                        }
                    }
                }
            }
            for &i in sc.scoring() {
                let c = &self.play[i as usize];
                if c.enhancement == Enhancement::Wild {
                    for (si, &s) in order.iter().enumerate() {
                        if c.is_suit(s, false, false, smeared) && suits[si] == 0 {
                            suits[si] += 1;
                            break;
                        }
                    }
                }
            }
            if suits.iter().all(|&x| x > 0) {
                ret!(xmult_mod: Some(3.0));
            }
        }
        if j == J::SeeingDouble {
            let smeared = self.smeared();
            // Hearts, Diamonds, Spades, Clubs
            let mut suits = [0u32; 4];
            for &i in sc.scoring() {
                let c = &self.play[i as usize];
                if c.enhancement != Enhancement::Wild {
                    if c.is_suit(Suit::Hearts, false, false, smeared) {
                        suits[0] += 1;
                    }
                    if c.is_suit(Suit::Diamonds, false, false, smeared) {
                        suits[1] += 1;
                    }
                    if c.is_suit(Suit::Spades, false, false, smeared) {
                        suits[2] += 1;
                    }
                    if c.is_suit(Suit::Clubs, false, false, smeared) {
                        suits[3] += 1;
                    }
                }
            }
            for &i in sc.scoring() {
                let c = &self.play[i as usize];
                if c.enhancement == Enhancement::Wild {
                    if c.is_suit(Suit::Clubs, false, false, smeared) && suits[3] == 0 {
                        suits[3] += 1;
                    } else if c.is_suit(Suit::Diamonds, false, false, smeared) && suits[1] == 0 {
                        suits[1] += 1;
                    } else if c.is_suit(Suit::Spades, false, false, smeared) && suits[2] == 0 {
                        suits[2] += 1;
                    } else if c.is_suit(Suit::Hearts, false, false, smeared) && suits[0] == 0 {
                        suits[0] += 1;
                    }
                }
            }
            if (suits[0] > 0 || suits[1] > 0 || suits[2] > 0) && suits[3] > 0 {
                ret!(xmult_mod: Some(2.0));
            }
        }
        if j == J::Wee {
            ret!(chip_mod: Some(ab.chips));
        }
        if j == J::Castle && ab.chips > 0.0 {
            ret!(chip_mod: Some(ab.chips));
        }
        if j == J::BlueJoker && !self.deck.is_empty() {
            ret!(chip_mod: Some(2.0 * self.deck.len() as f64));
        }
        if j == J::Erosion && (self.starting_deck_size - self.playing_card_count() as f64) > 0.0 {
            ret!(mult_mod: Some(4.0 * (self.starting_deck_size - self.playing_card_count() as f64)));
        }
        if j == J::Square {
            ret!(chip_mod: Some(ab.chips));
        }
        if j == J::Runner {
            ret!(chip_mod: Some(ab.chips));
        }
        if j == J::IceCream {
            ret!(chip_mod: Some(ab.chips));
        }
        if j == J::Stone && ab.stone_tally > 0.0 {
            ret!(chip_mod: Some(25.0 * ab.stone_tally));
        }
        if j == J::SteelJoker && ab.steel_tally > 0.0 {
            ret!(xmult_mod: Some(1.0 + 0.2 * ab.steel_tally));
        }
        if j == J::Bull && (self.dollars + self.dollar_buffer) > 0.0 {
            ret!(chip_mod: Some(2.0 * (self.dollars + self.dollar_buffer).max(0.0)));
        }
        if j == J::DriversLicense && ab.driver_tally >= 16.0 {
            ret!(xmult_mod: Some(3.0));
        }
        if j == J::Blackboard {
            let smeared = self.smeared();
            if self.hand.iter().all(|c| c.is_suit(Suit::Clubs, false, true, smeared) || c.is_suit(Suit::Spades, false, true, smeared)) {
                ret!(xmult_mod: Some(3.0));
            }
        }
        if j == J::Stencil && (self.joker_limit.limit - self.jokers.len() as f64) > 0.0 {
            ret!(xmult_mod: Some(ab.x_mult));
        }
        if j == J::Swashbuckler && ab.mult > 0.0 {
            ret!(mult_mod: Some(ab.mult));
        }
        if j == J::Joker {
            ret!(mult_mod: Some(ab.mult));
        }
        if j == J::Trousers && ab.mult > 0.0 {
            ret!(mult_mod: Some(ab.mult));
        }
        if j == J::RideTheBus && ab.mult > 0.0 {
            ret!(mult_mod: Some(ab.mult));
        }
        if j == J::Flash && ab.mult > 0.0 {
            ret!(mult_mod: Some(ab.mult));
        }
        if j == J::Popcorn && ab.mult > 0.0 {
            ret!(mult_mod: Some(ab.mult));
        }
        if j == J::GreenJoker && ab.mult > 0.0 {
            ret!(mult_mod: Some(ab.mult));
        }
        if j == J::FortuneTeller && self.consumable_usage_tarot > 0.0 {
            ret!(mult_mod: Some(self.consumable_usage_tarot));
        }
        if j == J::GrosMichel {
            ret!(mult_mod: Some(15.0));
        }
        if j == J::Cavendish {
            ret!(xmult_mod: Some(3.0));
        }
        if j == J::RedCard && ab.mult > 0.0 {
            ret!(mult_mod: Some(ab.mult));
        }
        if j == J::CardSharp && self.hands[sc.name as usize].played_this_round > 1.0 {
            ret!(xmult_mod: Some(3.0));
        }
        if j == J::Bootstraps && ((self.dollars + self.dollar_buffer) / 5.0).floor() >= 1.0 {
            ret!(mult_mod: Some(2.0 * ((self.dollars + self.dollar_buffer) / 5.0).floor()));
        }
        if j == J::Caino && ab.caino_xmult > 1.0 {
            ret!(xmult_mod: Some(ab.caino_xmult));
        }
        None
    }
}

/// `ability.type` (poker hand requirement) for a joker.
fn joker_hand_type(j: Joker) -> Option<HandType> {
    use Joker as J;
    Some(match j {
        J::Jolly | J::Sly | J::Duo => HandType::Pair,
        J::Zany | J::Wily | J::Trio => HandType::ThreeOfAKind,
        J::Mad | J::Clever => HandType::TwoPair,
        J::Crazy | J::Devious | J::Order => HandType::Straight,
        J::Droll | J::Crafty | J::Tribe => HandType::Flush,
        J::Family => HandType::FourOfAKind,
        _ => return None,
    })
}

/// `(t_mult, t_chips)`.
fn joker_t_values(j: Joker) -> (f64, f64) {
    use Joker as J;
    match j {
        J::Jolly => (8.0, 0.0),
        J::Zany => (12.0, 0.0),
        J::Mad => (10.0, 0.0),
        J::Crazy => (12.0, 0.0),
        J::Droll => (10.0, 0.0),
        J::Sly => (0.0, 50.0),
        J::Wily => (0.0, 100.0),
        J::Clever => (0.0, 80.0),
        J::Devious => (0.0, 100.0),
        J::Crafty => (0.0, 80.0),
        _ => (0.0, 0.0),
    }
}
