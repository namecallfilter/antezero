//! Differential tests of the RNG layer against the game's own LuaJIT. They
//! need the game's `lua51.dll` (and scripts), so they only run when asked
//! for: `cargo test -p oracle --test rng -- --include-ignored`.

use oracle::Oracle;
use sim::rng::{LuaRandom, PseudoRandom, pseudohash, round13};

fn parse(s: &str) -> Vec<f64> {
    s.split(',').filter(|x| !x.is_empty()).map(|x| x.parse::<f64>().unwrap()).collect()
}

fn raw() -> Oracle {
    assert!(oracle::dll_available(), "the RNG tests need the game's lua51.dll (see oracle/src/lib.rs)");
    Oracle::raw().unwrap()
}

#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn math_random_matches_luajit() {
    let o = raw();
    // Seeds spanning the ranges the game uses (pseudoseed outputs are in [0,1)).
    let out = o
        .eval(
            r#"
            local t = {}
            local s = 0.123456789
            for i = 1, 3000 do
                s = (s * 7.31 + 0.137) % 1
                local seed = (i % 7 == 0) and (s * 1000 - 500) or s
                math.randomseed(seed)
                t[#t+1] = string.format('%.17g', seed)
                t[#t+1] = string.format('%.17g', math.random())
                t[#t+1] = string.format('%.17g', math.random(52))
                t[#t+1] = string.format('%.17g', math.random(0, 23))
                t[#t+1] = string.format('%.17g', math.random())
            end
            return table.concat(t, ',')
        "#,
        )
        .unwrap();
    let v = parse(&out);
    assert_eq!(v.len(), 3000 * 5);
    for c in v.chunks(5) {
        let mut r = LuaRandom::new(c[0]);
        assert_eq!(r.random(), c[1], "seed {}", c[0]);
        assert_eq!(r.random_int(52.0), c[2], "seed {}", c[0]);
        assert_eq!(r.random_range(0.0, 23.0), c[3], "seed {}", c[0]);
        assert_eq!(r.random(), c[4], "seed {}", c[0]);
    }
}

#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn round13_matches_string_format() {
    let o = raw();
    let out = o
        .eval(
            r#"
            local t = {}
            math.randomseed(42)
            for i = 1, 200000 do
                local x = math.random()
                if i % 3 == 0 then x = (2.134453429141 + x*1.72431234)%1 end
                if i % 11 == 0 then x = math.floor(x*1e13)/1e13 + 0.5e-13 end
                t[#t+1] = string.format('%.17g', x)
                t[#t+1] = string.format('%.17g', math.abs(tonumber(string.format("%.13f", x))))
            end
            return table.concat(t, ',')
        "#,
        )
        .unwrap();
    let v = parse(&out);
    let mut n = 0;
    for c in v.chunks(2) {
        assert_eq!(round13(c[0]), c[1], "x = {:e}", c[0]);
        n += 1;
    }
    assert_eq!(n, 200000);
}

#[test]
#[ignore = "needs the game's lua51.dll; run with --include-ignored"]
fn pseudohash_and_pseudoseed_match_game_code() {
    let o = raw();
    let src = oracle::default_src().join("functions/misc_functions.lua");
    assert!(src.exists(), "the game scripts are missing ({})", src.display());
    o.exec(&format!("dofile({})", oracle::lua_str(&src.display().to_string().replace('\\', "/")))).unwrap();
    let keys = [
        "boss", "shuffle", "Joker1sho1", "rarity1sho", "cdt1", "edisho1", "Tag1", "Voucher1", "soul_Tarot1",
        "front", "nr1", "cashout2", "lucky_mult", "Joker2_resample2", "misprint", "a", "", "orbital", "to_do",
    ];
    let seeds = ["ABCDEFGH", "1", "TUTORIAL", "ZZZZZZZZ", "8NJHJ5M2", "seedwith spaces"];
    for seed in seeds {
        let code = format!(
            r#"
            G = {{GAME = {{pseudorandom = {{seed = '{seed}'}}}}}}
            G.GAME.pseudorandom.hashed_seed = pseudohash('{seed}')
            local keys = {{{}}}
            local t = {{string.format('%.17g', G.GAME.pseudorandom.hashed_seed)}}
            for rep = 1, 40 do
                for _, k in ipairs(keys) do
                    t[#t+1] = string.format('%.17g', pseudoseed(k))
                end
            end
            for _, k in ipairs(keys) do t[#t+1] = string.format('%.17g', pseudohash(k)) end
            return table.concat(t, ',')
        "#,
            keys.iter().map(|k| format!("'{k}'")).collect::<Vec<_>>().join(",")
        );
        let v = parse(&o.eval(&code).unwrap());
        let mut pr = PseudoRandom::new(seed);
        assert_eq!(pr.hashed_seed(), v[0], "hashed seed {seed}");
        let mut i = 1;
        for _ in 0..40 {
            for k in keys {
                assert_eq!(pr.seed_str(k), v[i], "seed {seed} key {k}");
                i += 1;
            }
        }
        for k in keys {
            assert_eq!(pseudohash(k.as_bytes()), v[i], "pseudohash {k}");
            i += 1;
        }
    }
}
