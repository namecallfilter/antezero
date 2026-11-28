//! The real-game bridge's legal-action port (`mod/src/legal.lua`)
//! run on the oracle's (vanilla) game, compared with `Game::legal_actions`
//! after every step of differential games. Needs the game's `lua51.dll`:
//! `cargo test -p oracle --test bridge_legal -- --include-ignored`.

use std::path::Path;

use oracle::command::{Command, Start};
use oracle::difftest::{Checks, Comparison, compare};
use oracle::policy::Policy;
use oracle::{Oracle, lua_path};
use serde_json::Value;
use sim::actions::Action;
use sim::data::{Back, BlindKind, Edition, Spectral, TagKind, Tarot};
use sim::game::{Game, Phase};
use sim::inject::{Consumable, Inject};
use sim::{Forced, Stake, num};

/// Loads legal.lua with stand-ins for the two helpers it takes from the
/// mod's core.lua (vanilla has no SMODS booster state).
fn load_legal(o: &mut Oracle) {
	let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../mod/src/legal.lua");
	o.exec(&format!(
        r"
        AZB = {{}}
        local S = G.STATES
        local KIND = {{[S.TAROT_PACK] = 'Arcana', [S.PLANET_PACK] = 'Celestial', [S.SPECTRAL_PACK] = 'Spectral',
                      [S.STANDARD_PACK] = 'Standard', [S.BUFFOON_PACK] = 'Buffoon'}}
        function AZB.is_pack_state(s) return KIND[s or G.STATE] ~= nil end
        function AZB.opened_booster_kind() return KIND[G.STATE] end
        dofile({})
        function AZ_LEGAL()
          local codes = AZB.legal_codes()
          codes.__array = true
          return JSON.encode(codes)
        end
        ",
        lua_path(&path)
    ))
    .expect("legal.lua loads in the vanilla game");
}

fn sim_codes(g: &Game) -> Vec<String> {
	let mut v = Vec::new();
	g.legal_actions(&mut v);
	v.iter().map(ToString::to_string).collect()
}

fn lua_codes(o: &mut Oracle) -> Vec<String> {
	use sim::json::ValueExt;

	let codes = o.eval("return AZ_LEGAL()").expect("AZ_LEGAL runs");
	let j: Value = serde_json::from_str(&codes).expect("AZ_LEGAL returns JSON");
	j.list()
		.iter()
		.map(|c| c.as_str().expect("a legal code is a string").to_string())
		.collect()
}

#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn bridge_legal_actions_match_the_sim() {
	assert!(
		oracle::available(),
		"the oracle needs the game's lua51.dll and scripts"
	);
	let mut o = Oracle::new().unwrap();
	load_legal(&mut o);
	let backs = [
		Back::Red,
		Back::Blue,
		Back::Yellow,
		Back::Magic,
		Back::Ghost,
		Back::Abandoned,
		Back::Plasma,
	];
	let mut checked = 0;
	for n in 0..24u64 {
		let seed = format!("LEGAL{n:03}");
		let start = Start {
			seed: &seed,
			back: backs[num::pick(n, backs.len())],
			stake: Stake::ALL[num::pick(n, Stake::ALL.len())],
		};
		let mut g = start.new_game();
		let mut lua = o.command(&start.to_lua()).unwrap();
		let mut policy = Policy::for_game(n, 0.15);
		let mut history = vec![start.to_string()];
		for _ in 0..250 {
			let j: Value = serde_json::from_str(&lua).unwrap();
			if let Comparison::Diverged(m) = compare(&mut g, &j, Checks::StateOnly) {
				panic!("engines diverged after {}: {m}", history.join(" / "));
			}
			if g.flow.phase == Phase::GameOver {
				break;
			}
			let (want, got) = (sim_codes(&g), lua_codes(&mut o));
			if want != got {
				let missing: Vec<_> = want.iter().filter(|c| !got.contains(c)).take(8).collect();
				let extra: Vec<_> = got.iter().filter(|c| !want.contains(c)).take(8).collect();
				let how = if missing.is_empty() && extra.is_empty() {
					"the same codes in another order".to_string()
				} else {
					format!("missing {missing:?}, extra {extra:?}")
				};
				panic!(
					"legal.lua disagrees in {:?} after {}: {how}",
					g.flow.phase,
					history[history.len().saturating_sub(8)..].join(" / "),
				);
			}
			checked += 1;
			let Some(command) = policy.next(&g) else {
				break;
			};
			history.push(command.to_string());
			command.apply(&mut g).unwrap();
			lua = o.command(&command.to_lua()).unwrap();
		}
	}
	assert!(checked > 1000, "only {checked} states checked");
}

/// Under Cerulean Bell, after the held card is copied, a copy is used and
/// the held cards are destroyed: the mod asks for the cards the simulator
/// asks for.
#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn bridge_legal_actions_match_the_sim_with_copies_of_the_forced_card() {
	assert!(
		oracle::available(),
		"the oracle needs the game's lua51.dll and scripts"
	);
	let quiet = |tag| {
		matches!(
			tag,
			Some(TagKind::Economy | TagKind::Handy | TagKind::Garbage | TagKind::Skip)
		)
	};
	let seed = (0..1000)
		.map(|i| format!("LEGALBELL{i:03}"))
		.find(|seed| {
			let tags = Game::new_run(seed, Back::Red, Stake::White)
				.blind_select
				.skip_tags;
			quiet(tags.small) && quiet(tags.big)
		})
		.expect("some run has skip tags that open nothing");
	let start = Start {
		seed: &seed,
		back: Back::Red,
		stake: Stake::White,
	};
	let mut o = Oracle::new().unwrap();
	load_legal(&mut o);
	let mut g = start.new_game();
	o.command(&start.to_lua()).unwrap();
	g.blind_select.boss = BlindKind::FinalBell;
	o.exec("G.GAME.round_resets.blind_choices.Boss = 'bl_final_bell'")
		.unwrap();
	let held = |g: &Game, how: Forced| -> Vec<usize> {
		let hand = g.areas.hand.cards.iter().enumerate();
		hand.filter(|(_, c)| c.forced == how)
			.map(|(i, _)| i)
			.collect()
	};
	let mut run = |g: &mut Game, command: Command| {
		command.apply(g).unwrap();
		let lua = o.command(&command.to_lua()).unwrap();
		let state: Value = serde_json::from_str(&lua).unwrap();
		if let Comparison::Diverged(m) = compare(g, &state, Checks::StateOnly) {
			panic!("engines diverged after {command}: {m}");
		}
		assert_eq!(sim_codes(g), lua_codes(&mut o), "after {command}");
	};
	for card in [
		Consumable::Spectral(Spectral::Cryptid),
		Consumable::Tarot(Tarot::Magician),
		Consumable::Tarot(Tarot::HangedMan),
	] {
		let edition = Some(Edition::Negative);
		run(
			&mut g,
			Command::Inject(Inject::Consumable { card, edition }),
		);
	}
	for action in [Action::SkipBlind, Action::SkipBlind, Action::SelectBlind] {
		run(&mut g, Command::Act(action));
	}
	let use_on = |cards: &[usize]| {
		Command::Act(Action::UseConsumable {
			index: 0,
			targets: cards.iter().copied().collect(),
		})
	};
	let bell = held(&g, Forced::Locked)[0];
	run(&mut g, use_on(&[bell]));
	let copy = held(&g, Forced::Flagged)[0];
	run(&mut g, use_on(&[bell, copy]));
	assert_eq!(held(&g, Forced::Locked), [bell, copy]);
	run(&mut g, use_on(&[bell, copy]));
	assert_eq!(
		held(&g, Forced::Locked).len(),
		1,
		"the other copy took over"
	);
}
