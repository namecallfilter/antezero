//! The real-game bridge's legal-action port (`mod/src/legal.lua`)
//! run on the oracle's (vanilla) game, compared with `Game::legal_actions`
//! after every step of differential games. Needs the game's `lua51.dll`:
//! `cargo test -p oracle --test bridge_legal -- --include-ignored`.

use std::collections::HashSet;
use std::path::Path;

use oracle::command::{Command, Start};
use oracle::difftest::{Checks, Comparison, compare};
use oracle::policy::Policy;
use oracle::{Oracle, lua_path};
use sim::actions::Action;
use sim::data::{Back, BlindKind, Edition, Spectral, TagKind, Tarot};
use sim::game::{Game, Phase};
use sim::inject::{Consumable, Inject};
use sim::{CardMask, Forced, Stake, json, num};

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
	let codes = o.eval("return AZ_LEGAL()").expect("AZ_LEGAL runs");
	let j = json::parse(&codes).expect("AZ_LEGAL returns JSON");
	j.as_list()
		.iter()
		.map(|c| c.as_str().expect("a legal code is a string").to_string())
		.collect()
}

/// Asserts that the mod lists the simulator's legal actions, in its order,
/// after the commands `history`.
fn assert_same_codes(g: &Game, o: &mut Oracle, history: &str) {
	let (want, got) = (sim_codes(g), lua_codes(o));
	if want == got {
		return;
	}
	let (in_sim, in_lua): (HashSet<_>, HashSet<_>) = (want.iter().collect(), got.iter().collect());
	let missing: Vec<_> = want
		.iter()
		.filter(|c| !in_lua.contains(c))
		.take(8)
		.collect();
	let extra: Vec<_> = got.iter().filter(|c| !in_sim.contains(c)).take(8).collect();
	let how = if missing.is_empty() && extra.is_empty() {
		"the same codes in another order".to_string()
	} else {
		format!("missing {missing:?}, extra {extra:?}")
	};
	panic!(
		"legal.lua disagrees in {:?} after {history}: {how}",
		g.flow.phase
	);
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
			let j = json::parse(&lua).unwrap();
			if let Comparison::Diverged(m) = compare(&mut g, &j, Checks::StateOnly) {
				panic!("engines diverged after {}: {m}", history.join(" / "));
			}
			if g.flow.phase == Phase::GameOver {
				break;
			}
			let recent = history[history.len().saturating_sub(8)..].join(" / ");
			assert_same_codes(&g, &mut o, &recent);
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

/// Both engines in a first round against Cerulean Bell, with `legal.lua`
/// loaded.
struct BellRound {
	o: Oracle,
	g: Game,
}

impl BellRound {
	/// With `consumables` in this order.
	fn start(consumables: &[Consumable]) -> BellRound {
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
		let mut o = Oracle::new().expect("the harness loads (`oracle::available()` holds)");
		load_legal(&mut o);
		let mut g = start.new_game();
		o.command(&start.to_lua())
			.expect("the oracle starts the run");
		g.blind_select.boss = BlindKind::FinalBell;
		o.exec("G.GAME.round_resets.blind_choices.Boss = 'bl_final_bell'")
			.expect("the game takes the boss");
		let mut round = BellRound { o, g };
		for &card in consumables {
			let edition = Some(Edition::Negative);
			round.step(Command::Inject(Inject::Consumable { card, edition }));
		}
		for action in [Action::SkipBlind, Action::SkipBlind, Action::SelectBlind] {
			round.step(Command::Act(action));
		}
		round
	}

	/// Runs `command` in both engines, which must agree on the state after
	/// it.
	fn run(&mut self, command: Command) {
		command
			.apply(&mut self.g)
			.unwrap_or_else(|e| panic!("sim rejected {command:?}: {e}"));
		let lua = self
			.o
			.command(&command.to_lua())
			.expect("the oracle runs the command");
		let state = json::parse(&lua).expect("the oracle returns JSON");
		if let Comparison::Diverged(m) = compare(&mut self.g, &state, Checks::StateOnly) {
			panic!("engines diverged after {command}: {m}");
		}
	}

	/// [`run`](Self::run), after which the mod must list the simulator's
	/// legal actions.
	fn step(&mut self, command: Command) {
		self.run(command);
		assert_same_codes(&self.g, &mut self.o, &command.to_string());
	}

	/// What the mod's action handler (`mod/src/act.lua`) does with `action`
	/// (a Lua table, its cards counted from 1) before the button is
	/// pressed: why it refuses, or the hand positions (from 1) of the
	/// game's highlight list.
	fn prepare(&mut self, action: &str) -> String {
		let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../mod/src/act.lua");
		let prepare = format!(
			"AZB.mark_action = function() end
			dofile({})
			local ok, why = pcall(AZB.act_prepare, {action})
			if not ok then return 'refused: ' .. (type(why) == 'table' and why.msg or tostring(why)) end
			local at = {{}}
			for _, c in ipairs(G.hand.highlighted) do
				for i, h in ipairs(G.hand.cards) do
					if h == c then at[#at + 1] = i end
				end
			end
			return 'highlighted: ' .. table.concat(at, ' ')",
			lua_path(&path),
		);
		self.o.eval(&prepare).expect("the mod's handler runs")
	}

	/// The hand positions of the cards the Bell holds in a given way.
	fn held(&self, how: fn(Forced) -> bool) -> Vec<usize> {
		let hand = self.g.areas.hand.cards.iter().enumerate();
		hand.filter(|(_, c)| how(c.forced))
			.map(|(i, _)| i)
			.collect()
	}
}

/// Using the first consumable on the hand cards `cards`.
fn use_on(cards: &[usize]) -> Command {
	Command::Act(Action::UseConsumable {
		index: 0,
		targets: cards.iter().copied().collect(),
	})
}

/// Under Cerulean Bell, after the held card is copied, a copy is used and
/// the held cards are destroyed: the mod asks for the cards the simulator
/// asks for.
#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn bridge_legal_actions_match_the_sim_with_copies_of_the_forced_card() {
	let mut round = BellRound::start(&[
		Consumable::Spectral(Spectral::Cryptid),
		Consumable::Tarot(Tarot::Magician),
		Consumable::Tarot(Tarot::HangedMan),
	]);
	let bell = round.held(Forced::is_locked)[0];
	// The handler highlights by hand position, whatever the cards' order.
	let free: Vec<usize> = (1..=8).filter(|&i| i != bell + 1).take(2).collect();
	let play = format!(
		"{{type = 'play', cards = {{{}, {}, {}}}}}",
		free[1],
		bell + 1,
		free[0]
	);
	let listed = format!("highlighted: {} {} {}", bell + 1, free[0], free[1]);
	assert_eq!(round.prepare(&play), listed);
	round.step(use_on(&[bell]));
	let copy = round.held(|f| f == Forced::Flagged)[0];
	round.step(use_on(&[bell, copy]));
	assert_eq!(round.held(Forced::is_locked), [bell, copy]);
	round.step(use_on(&[bell, copy]));
	assert_eq!(
		round.held(Forced::is_locked).len(),
		1,
		"the other copy took over"
	);
}

/// A hand of more than 32 cards whose held card is past the positions a
/// selection can name: neither the simulator nor the mod lists a selection,
/// and both do again once the card is in reach.
#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn bridge_legal_actions_match_the_sim_with_the_held_card_out_of_reach() {
	let mut cards = [Consumable::Spectral(Spectral::Cryptid); 15];
	cards[14] = Consumable::Tarot(Tarot::Hermit);
	let mut round = BellRound::start(&cards);
	let bell = round.held(Forced::is_locked)[0];
	for _ in 0..13 {
		round.run(use_on(&[bell]));
	}
	let last = round.g.areas.hand.cards.len() - 1;
	assert!(
		last >= CardMask::CAPACITY,
		"the hand has {} cards",
		last + 1
	);
	let move_hand = |from, to| {
		let [from, to] = [from, to].map(num::byte);
		Command::Act(Action::MoveHand { from, to })
	};
	round.step(move_hand(bell, last));
	let mut legal = Vec::new();
	round.g.legal_actions(&mut legal);
	let hermit = Action::UseConsumable {
		index: 1,
		targets: CardMask::EMPTY,
	};
	let sells = [Action::SellConsumable(0), Action::SellConsumable(1)];
	assert_eq!(legal, [hermit, sells[0], sells[1]], "what takes no cards");
	// The mod's handler refuses what would act on a card unasked: a play
	// without the held card, and Cryptid with no cards or none named.
	let refused = "refused: the cards leave out one the blind holds";
	assert_eq!(round.prepare("{type = 'play', cards = {1}}"), refused);
	for action in [
		"{type = 'use', index = 1, cards = {}}",
		"{type = 'use', index = 1}",
	] {
		assert_eq!(round.prepare(action), "refused: no cards", "{action}");
	}
	let held_card = format!("highlighted: {}", last + 1);
	assert_eq!(round.prepare("{type = 'use', index = 2}"), held_card);

	round.step(move_hand(last, CardMask::CAPACITY - 1));
	round.g.legal_actions(&mut legal);
	let held = CardMask::from_iter([CardMask::CAPACITY - 1]);
	assert!(legal.contains(&Action::Play(held)), "the held card alone");
	assert_eq!(round.prepare("{type = 'play', cards = {1}}"), refused);
	assert_eq!(
		round.prepare("{type = 'play', cards = {32, 1}}"),
		"highlighted: 32 1"
	);
}
