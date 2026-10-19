//! Text codes of actions and scenario injections (histories, bindings).

use sim::{Action, Kind};
use sim::data::*;
use sim::game::{Game, Phase};
use sim::inject::Inject;

#[test]
fn every_legal_action_round_trips() {
    let mut g = Game::new_run("CODES", Back::Red, 1);
    let mut acts = Vec::new();
    let mut n = 0;
    for step in 0..400 {
        if g.is_terminal() {
            break;
        }
        g.legal_actions(&mut acts);
        for a in &acts {
            let code = a.to_string();
            assert_eq!(code.parse::<Action>(), Ok(*a), "{code}");
            n += 1;
        }
        let a = if g.phase == Phase::Round {
            *acts.iter().find(|a| matches!(a, Action::Play(_))).unwrap()
        } else {
            acts[step % acts.len()]
        };
        g.step(a).unwrap();
    }
    for a in [Action::MoveJoker { from: 0, to: 3 }, Action::MoveHand { from: 7, to: 1 }, Action::SortHand { by_suit: true }] {
        assert_eq!(a.to_string().parse::<Action>(), Ok(a));
    }
    assert!(n > 1000, "only {n} codes checked");
}

#[test]
fn malformed_action_codes_are_rejected() {
    for bad in ["", "play", "play x", "select now", "play 3 4", "use 0", "sell_joker 256", "move_joker 1 999", "sort 2", "buy -1", "jump"] {
        assert!(bad.parse::<Action>().is_err(), "{bad:?} parsed");
    }
    assert_eq!("  play   30 ".parse::<Action>(), Ok(Action::Play(30)));
}

#[test]
fn inject_codes_round_trip_and_reject_malformed_input() {
    let samples = [
        Inject::Joker(Joker::Blueprint, Edition::Negative, true),
        Inject::Consumable(Kind::Planet(Planet::Eris), Edition::None),
        Inject::Consumable(Kind::Spectral(Spectral::Ankh), Edition::Polychrome),
        Inject::Card(Suit::Clubs, 12, Enhancement::Lucky, Edition::Holo, Some(Seal::Purple)),
        Inject::Card(Suit::Hearts, ACE, Enhancement::Base, Edition::None, None),
        Inject::Dollars(17.5),
        Inject::Voucher(Voucher::Palette),
    ];
    for i in samples {
        let code = i.to_string();
        let back: Inject = code.parse().unwrap();
        assert_eq!(back.to_string(), code);
    }
    for bad in [
        "inject_joker",
        "inject_joker 150 0 0",
        "inject_joker 1 5 0",
        "inject_joker 1 0 0 9",
        "inject_cons 3 0 0",
        "inject_cons 0 22 0",
        "inject_card 4 2 0 0 0",
        "inject_card 0 15 0 0 0",
        "inject_card 0 2 9 0 0",
        "inject_card 0 2 0 0 5",
        "inject_dollars junk",
        "inject_dollars inf",
        "inject_voucher 32",
        "inject_tag 1",
    ] {
        assert!(bad.parse::<Inject>().is_err(), "{bad:?} parsed");
    }
}

