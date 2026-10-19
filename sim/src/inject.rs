//! Scenario construction helpers: inject cards, money and vouchers into a
//! running game (mirroring the game's own `add_joker` /
//! `create_playing_card`). Useful for curricula, unit tests and the oracle
//! differential tests, which apply identical injections to the real game.

use crate::actions::{Operands, ParseError};
use crate::card::Kind;
use crate::data::*;
use crate::game::{Area, Game};
use std::fmt;
use std::str::FromStr;

impl Game {
    /// `add_joker(key, edition, silent, eternal)` for a joker.
    pub fn inject_joker(&mut self, j: Joker, edition: Edition, eternal: bool) {
        self.begin_action();
        let mut c = self.make_card(Kind::Joker(j));
        self.materialize(1.0);
        self.add_to_deck(&mut c, false);
        if edition != Edition::None {
            self.set_edition(&mut c, edition, Some(Area::Jokers));
            self.edition_fx(edition, false);
        }
        if eternal {
            c.set_eternal(true);
        }
        self.jokers.push(c);
        self.run_queue();
    }

    /// `add_joker(key, edition)` for a tarot, planet or spectral card.
    pub fn inject_consumable(&mut self, kind: Kind, edition: Edition) {
        assert!(kind.is_consumable(), "inject_consumable needs a consumable");
        self.begin_action();
        let mut c = self.make_card(kind);
        self.materialize(1.0);
        self.add_to_deck(&mut c, false);
        if edition != Edition::None {
            self.set_edition(&mut c, edition, Some(Area::Consumables));
            self.edition_fx(edition, false);
        }
        self.consumables.push(c);
        self.run_queue();
    }

    /// `create_playing_card({front, center}, G.deck)` + edition/seal.
    pub fn inject_card(&mut self, suit: Suit, rank: Rank, enh: Enhancement, edition: Edition, seal: Option<Seal>) {
        self.begin_action();
        let mut c = self.make_playing_card(suit, rank, enh);
        c.face_down = true;
        if edition != Edition::None {
            self.set_edition(&mut c, edition, Some(Area::Deck));
        }
        if seal.is_some() {
            c.seal = seal;
            self.set_cost(&mut c, Some(Area::Deck));
        }
        self.deck.insert(0, c);
        self.run_queue();
    }

    pub fn inject_dollars(&mut self, d: f64) {
        self.begin_action();
        self.dollars += d;
    }

    /// Marks a voucher redeemed and applies it (like a starting deck voucher).
    pub fn inject_voucher(&mut self, v: Voucher) {
        self.begin_action();
        self.set_voucher_used(v);
        self.apply_voucher(v, true);
        self.run_queue();
    }
}

/// A scenario injection applied identically to both engines.
#[derive(Clone, Copy, Debug)]
pub enum Inject {
    Joker(Joker, Edition, bool),
    Consumable(Kind, Edition),
    Card(Suit, Rank, Enhancement, Edition, Option<Seal>),
    Dollars(f64),
    Voucher(Voucher),
}

impl fmt::Display for Inject {
    /// The text code used in histories (enum discriminants as numbers).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Inject::Joker(j, e, et) => write!(f, "inject_joker {} {} {}", j as u8, e as u8, et as u8),
            Inject::Consumable(k, e) => {
                let (t, i) = match k {
                    Kind::Tarot(t) => (0, t as u8),
                    Kind::Planet(p) => (1, p as u8),
                    Kind::Spectral(s) => (2, s as u8),
                    _ => return Err(fmt::Error),
                };
                write!(f, "inject_cons {t} {i} {}", e as u8)
            }
            Inject::Card(s, r, enh, e, seal) => {
                write!(f, "inject_card {} {} {} {} {}", s as u8, r, enh as u8, e as u8, seal.map_or(0, |x| x as u8 + 1))
            }
            Inject::Dollars(d) => write!(f, "inject_dollars {d}"),
            Inject::Voucher(v) => write!(f, "inject_voucher {}", v as u8),
        }
    }
}

impl FromStr for Inject {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Inject, ParseError> {
        let (op, mut n) = Operands::new(s);
        fn pick<T: Copy>(table: &[T], i: usize, n: &Operands) -> Result<T, ParseError> {
            table.get(i).copied().ok_or_else(|| n.err())
        }
        let i = match op.ok_or_else(|| n.err())? {
            "inject_joker" => {
                let j = pick(&ALL_JOKERS, n.next()?, &n)?;
                let e = pick(&Edition::ALL, n.next()?, &n)?;
                Inject::Joker(j, e, n.next::<u8>()? != 0)
            }
            "inject_cons" => {
                let k = match n.next::<u8>()? {
                    0 => Kind::Tarot(pick(&ALL_TAROTS, n.next()?, &n)?),
                    1 => Kind::Planet(pick(&ALL_PLANETS, n.next()?, &n)?),
                    2 => Kind::Spectral(pick(&ALL_SPECTRALS, n.next()?, &n)?),
                    _ => return Err(n.err()),
                };
                Inject::Consumable(k, pick(&Edition::ALL, n.next()?, &n)?)
            }
            "inject_card" => {
                let suit = pick(&Suit::ALL, n.next()?, &n)?;
                let rank: Rank = n.next()?;
                if !(2..=ACE).contains(&rank) {
                    return Err(n.err());
                }
                let enh = pick(&Enhancement::ALL, n.next()?, &n)?;
                let ed = pick(&Edition::ALL, n.next()?, &n)?;
                let seal = match n.next::<usize>()? {
                    0 => None,
                    s => Some(pick(&Seal::ALL, s - 1, &n)?),
                };
                Inject::Card(suit, rank, enh, ed, seal)
            }
            "inject_dollars" => {
                let d: f64 = n.next()?;
                if !d.is_finite() {
                    return Err(n.err());
                }
                Inject::Dollars(d)
            }
            "inject_voucher" => Inject::Voucher(pick(&ALL_VOUCHERS, n.next()?, &n)?),
            _ => return Err(n.err()),
        };
        n.end(i)
    }
}

impl Game {
    /// Applies an [`Inject`].
    pub fn inject(&mut self, i: &Inject) {
        match *i {
            Inject::Joker(j, e, et) => self.inject_joker(j, e, et),
            Inject::Consumable(k, e) => self.inject_consumable(k, e),
            Inject::Card(s, r, enh, e, seal) => self.inject_card(s, r, enh, e, seal),
            Inject::Dollars(d) => self.inject_dollars(d),
            Inject::Voucher(v) => self.inject_voucher(v),
        }
    }
}
