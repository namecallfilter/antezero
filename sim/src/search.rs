//! Agent support: determinization of hidden information and play previews
//! (used by search, candidate scoring and snapshot restarts).

use crate::actions::{Action, ActionError, CardMask, mask_indices};
use crate::card::Card;
use crate::game::Game;
use crate::rng::{Key, PseudoRandom, shuffle};

impl Game {
    /// Replaces everything a player cannot know with a fresh sample, keeping
    /// the visible state: every future random outcome (shops, packs, bosses,
    /// tags, chance effects), the draw order of the deck, the identities of
    /// face-down hand cards (drawn again from the unseen cards), the order of
    /// face-down jokers, and the ante's voucher if no shop has shown it yet.
    /// For determinized search and for restarting from snapshots.
    pub fn resample_hidden(&mut self, seed: &str) {
        #[cfg(feature = "trace-keys")]
        let trace = std::mem::take(&mut self.rng.trace);
        self.rng = PseudoRandom::new(seed);
        #[cfg(feature = "trace-keys")]
        {
            self.rng.trace = trace;
        }
        let key = |s: &str| Key::from(s);
        // Face-down hand cards: shuffle them into the unseen pool and deal
        // the same slots again; their face-down state stays with the slot.
        let down: Vec<usize> = (0..self.hand.len()).filter(|&i| self.hand[i].face_down).collect();
        if !down.is_empty() {
            // Flags of a card lying in the deck, for hand cards moved there.
            let deck_flags = self.deck.first().map(|c| (c.face_down, c.ab.wheel_flipped)).unwrap_or((false, false));
            let mut pool: Vec<(Card, bool)> = down.iter().map(|&i| (self.hand[i].clone(), true)).collect();
            pool.extend(self.deck.drain(..).map(|c| (c, false)));
            let s = self.rng.seed_for(&key("resample_hand"));
            shuffle(&mut pool, s);
            let rest = pool.split_off(down.len());
            for ((mut c, _), &i) in pool.into_iter().zip(&down) {
                c.face_down = self.hand[i].face_down;
                c.ab.wheel_flipped = self.hand[i].ab.wheel_flipped;
                self.hand[i] = c;
            }
            self.deck = rest
                .into_iter()
                .map(|(mut c, from_hand)| {
                    if from_hand {
                        (c.face_down, c.ab.wheel_flipped) = deck_flags;
                    }
                    c
                })
                .collect();
        }
        let s = self.rng.seed_for(&key("resample"));
        shuffle(&mut self.deck, s);
        // Face-down jokers: permute among their slots.
        let jd: Vec<usize> = (0..self.jokers.len()).filter(|&i| self.jokers[i].face_down).collect();
        if jd.len() > 1 {
            let mut js: Vec<Card> = jd.iter().map(|&i| self.jokers[i].clone()).collect();
            let s = self.rng.seed_for(&key("resample_jokers"));
            shuffle(&mut js, s);
            for (c, &i) in js.into_iter().zip(&jd) {
                self.jokers[i] = c;
            }
        }
        if !self.voucher_seen && self.current_voucher.is_some() && !self.shop_open {
            self.current_voucher = Some(self.get_next_voucher(false));
        }
    }

    /// Score `Play(mask)` would make, computed on `self`, which is left
    /// mid-hand and must be discarded (call it on a clone).
    pub fn preview_play(&mut self, mask: CardMask) -> Result<f64, ActionError> {
        let a = Action::Play(mask);
        self.check(&a)?;
        self.begin_action();
        self.do_play(mask_indices(mask));
        self.run_until_scored();
        assert!(self.scored, "preview_play: the hand was never scored");
        Ok(self.last_hand_score)
    }
}
