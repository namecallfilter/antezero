//! The real-game bridge's legal-action port (`mod/src/legal.lua`)
//! run on the oracle's (vanilla) game, compared with `Game::legal_actions`
//! after every step of differential games. Needs the game's `lua51.dll`:
//! `cargo test -p oracle --test bridge_legal -- --include-ignored`.

use oracle::diff::{
	Mismatch, Xs, action_to_lua, choose, compare, inject_to_lua, random_inject, start_to_lua,
};
use oracle::{Oracle, json, lua_str};
use sim::data::Back;
use sim::game::{Game, Phase};
use sim::num;

/// Loads legal.lua with stand-ins for the two helpers it takes from the
/// mod's core.lua (vanilla has no SMODS booster state).
fn load_legal(o: &Oracle) {
	let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../mod/src/legal.lua");
	let path = path.display().to_string().replace('\\', "/");
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
        lua_str(&path)
    ))
    .expect("legal.lua loads in the vanilla game");
}

fn sim_codes(g: &Game) -> Vec<String> {
	let mut v = Vec::new();
	g.legal_actions(&mut v);
	v.iter().map(std::string::ToString::to_string).collect()
}

fn lua_codes(o: &Oracle) -> Vec<String> {
	let codes = o.eval("return AZ_LEGAL()").expect("AZ_LEGAL runs");
	let j = json::parse(&codes).expect("AZ_LEGAL returns JSON");
	j.arr()
		.iter()
		.map(|c| c.str().expect("a legal code is a string").to_string())
		.collect()
}

#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn bridge_legal_actions_match_the_sim() {
	assert!(
		oracle::available(),
		"the oracle needs the game's lua51.dll and scripts"
	);
	let o = Oracle::new().unwrap();
	load_legal(&o);
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
		let (seed, back, stake) = (
			format!("LEGAL{n:03}"),
			backs[num::pick(n, backs.len())],
			num::byte(num::pick(n, 8)) + 1,
		);
		let mut g = Game::new_run(&seed, back, stake);
		let mut lua = o.call("oracle", &start_to_lua(&seed, back, stake)).unwrap();
		let mut rng = Xs(n.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
		let mut rerolls = 0;
		let mut history = vec![format!("start {seed} {} {stake}", back.key())];
		for _ in 0..250 {
			let j = json::parse(&lua).unwrap();
			if let Some(Mismatch::State(d)) = compare(&mut g, &j, false) {
				panic!(
					"engines diverged ({:?}) after {}",
					&d[..d.len().min(5)],
					history.join(" / ")
				);
			}
			if g.phase == Phase::GameOver {
				break;
			}
			let (want, got) = (sim_codes(&g), lua_codes(&o));
			if want != got {
				let missing: Vec<_> = want.iter().filter(|c| !got.contains(c)).take(8).collect();
				let extra: Vec<_> = got.iter().filter(|c| !want.contains(c)).take(8).collect();
				panic!(
					"legal.lua disagrees in {:?} after {}: missing {missing:?}, extra {extra:?} (order equal: {})",
					g.phase,
					history[history.len().saturating_sub(8)..].join(" / "),
					missing.is_empty() && extra.is_empty()
				);
			}
			checked += 1;
			if g.phase != Phase::Shop {
				rerolls = 0;
			}
			if rng.chance(0.15)
				&& let Some(inj) = random_inject(&g, &mut rng)
			{
				history.push(inj.to_string());
				g.inject(&inj);
				lua = o.call("oracle", &inject_to_lua(&inj)).unwrap();
				continue;
			}
			let Some(a) = choose(&g, &mut rng, &mut rerolls) else {
				break;
			};
			history.push(a.to_string());
			g.step(a).unwrap();
			lua = o.call("oracle", &action_to_lua(&a)).unwrap();
		}
	}
	assert!(checked > 1000, "only {checked} states checked");
}
