# AnteZero's simulator in BalatroEngine's video pipeline: phase 1

2026-11-27. Everything here was run on CPU with at most 4 threads. Upstream code is
BalatroEngine at `a3fea4c`, imported under `cv/`.

## Answers

**Can AnteZero's simulator replace their engine?** Yes for the game rules. One gap is left:
their ground-truth log comes from an old Steamodded build that starts a blind differently
with Marble Joker or Certificate (section 2). The Steamodded of today does not, so the
simulator does not model it, and a video recorded on such a build stops at that point.

- On a real video it was exact. For `youtube.com/watch?v=wInI1W_EaEw` (Magic Deck, Gold
  Stake, seed `LNDW2KK8` from the description) the simulator, given only the seed, deck and
  stake, predicted the first hand, both redraws, the score, the first shop and the jokers
  Riff-Raff created. The prototype below then followed the first 10 minutes (87 actions,
  into ante 3): all 532 HUD numbers read along the way equal the simulator's, and the deck at
  the end equals the deck view on the video's last frame (47 cards, same count for every suit
  and every rank).
- On their one ground-truth log (299 entries, a stress run with 89 jokers) the simulator
  reproduces 273 of 299 logged states exactly, RNG streams included, when each transition is
  tested on its own; 7 more differ only because the log's snapshot was taken mid-action. The
  other 19 are two ways that game's old Steamodded build departs from vanilla, and from
  the Steamodded of today (below). Given only the actions, it diverges at the first blind
  (entry 3) for the same reason.
- Their engine passes its own parity test on that log (0 discrepancies), but that test copies
  the hand, shop, pack, consumables, jokers, blind and score from the log before every step
  and checks a few counters. Their engine cannot deal a hand or fill a shop from a seed.

**What "97%" is.** The share of trajectory steps the pipeline did not flag as `corrupted`
(`clean_pct`). It is the pipeline's own consistency check, not accuracy against a known game.

**What "6 minutes" is.** Almost all stage 1 (finding button clicks in every other frame).
The figure itself is not in the repo; their measured stage 1 speed and the batch runner's
defaults give 5 to 6 minutes for a 50 to 60 minute video on their Ryzen 9800X3D + RTX 5090.

**The seed** comes from the YouTube description, not from OCR. Doc writes it there ("I put
the seeds in the video descriptions"); the seed OCR model is an optional fallback.

## 1. Their pipeline

Stage 1 (`main.py`, CPU): template matching finds which button is pressed in each kept
frame. Output: `{frame: button, box}`.

Stage 2 (`post_processing.py`): removes held-button duplicates, finds the play that ended
each round with a video classifier, decodes the frames it needs in a second pass, runs the
detectors and classifiers on them (hand, joker bar, consumables, shop, pack, HUD OCR), and
steps their Python engine, overwriting engine state with what the classifiers and the OCR
read ("drift recovery").

### "97% data retention"

- Definition: `_summarize_trajectory_cleanliness` and `_aggregate_cleanliness_summary` in
  `vod_pipeline/runner.py`: `clean_steps / total_steps`, where a step is clean unless its
  action has `metadata["corrupted"]`.
- Flags are set by `_flag_corrupted_action` and `_scan_and_quarantine_corrupted_segments`
  (`post_processing.py:3226`, `:3281`). Reasons, by reach:
  - one step: `unexplained_joker_drift`, `transient_inventory_glitch`, `large_ocr_override`
    (the HUD's ante disagrees with the engine, the round is off by 2 or more, or money is off
    by $15 or more), `location_mismatch`, `unexpected_deck_addition`,
    `deck_drift_before_full_deck_resync`;
  - to the end of the round: `hand_size_over_15`, `empty_hand_cv`, `missing_shop_cv`,
    `cashout_without_round_won`;
  - to the end of the run: `downstream_cascade_quarantine`, when several of the above occur
    together.
- Not in the number: videos skipped before processing (no seed or deck in the description,
  shorter than 30 minutes), play after ante 8 (dropped), and steps that are wrong without
  tripping a check. Their batch reports are not in the repo, so the real mix of reasons
  behind the 3% is not known.
- On my 10-minute clip their stage 2 flagged 3 of 95 steps (96.8% clean): one
  `unexplained_joker_drift` (stage 1 missed a Sell click) and two `large_ocr_override` (their
  engine's ante was wrong, 2 against the HUD's 3). It also ended on round 8; the HUD says 7.

### Where the time goes

Their numbers (`notes/template_matching_optimization_README.md`, 9800X3D + RTX 5090):
stage 1 runs at 165.7 kept frames/s (decode on the GPU's NVDEC 42%, template matching 45%).
The batch runner keeps every other frame (`--skip-frames 1`, 30 per second of a 60 fps video)
and runs two stage 1 processes at once; stage 2 is estimated at 10 to 15 s on the GPU. A 55
minute video is 99,000 kept frames: 10 minutes for one process, 5 with two in parallel.

Measured here on the first 10 minutes of the video above (1080p60 H.264, 36,001 frames):

| step | wall time | notes |
|---|---|---|
| stage 1 | 445 s | 3 matching threads, 40 kept frames/s; template matching 80% |
| their stage 2 | 70 s | round-end classifier 19.8, second decode pass 21.7, detectors + classifiers + OCR on 247 frames 21.7, engine 1.9 |
| decode alone | 48.6 s | ffmpeg, 4 threads; keyframes only: 2.8 s |

So on CPU alone a 58 minute video costs about 43 minutes of stage 1 and 7 of stage 2.
Needs a GPU in their setup: video decode in stage 1 (there is a CPU fallback) and the
classifiers in stage 2 (they run on CPU; see above).

### Seed, deck, stake

`vod_pipeline/metadata.py` reads them from the title and description: a labelled seed
(`Magic Deck:` then `LNDW2KK8`), the deck name, and the stake (Gold unless stated). A video
without both is skipped. Fallbacks: the deck by template matching on a few frames
(`vod_pipeline/deck_matcher.py`), and the seed by OCR (`classifiers/seed_ocr/pipeline.py`,
off unless `--allow-ocr-seed-fallback`), which scans the last 7 minutes backwards for the
end-of-run screen and votes over 12 frames.

## 2. Parity

### The log

`testing/telemetry_data/` holds one log: `run_DMF1SJ5Y_299steps.jsonl`, Ghost Deck, White
Stake, 8 antes, 13 hands played with scores up to 1.9e19. It is not a normal game: the mod's
"Setup Test Suite" gave the run 89 negative jokers and 33 negative consumables and made half
the deck Lucky and half Glass. The log has the seed, the full state before every action
(hand, draw pile in order, discard pile, jokers with their counters, consumables, shop, pack,
money, score, every RNG stream) and the action with its target. Nothing needed is missing.

`antezero_bridge/telemetry.py` rebuilds the edited start (injecting the cards in the mod's
order, then two quirks of how the mod made them: under an open menu the game does not mark a
card as seen, and the mod raises the slot limits before the negative editions add theirs).
After that the simulator equals the log's first settled state in every compared field,
including all 13 RNG streams, the draw pile order and To Do List's hand.

Compared at every entry: money, ante, round, hands and discards left, score, blind target,
reroll cost, free rerolls, skips, slot limits, location, blind, boss, tags, the rotating
joker targets, vouchers, hand levels and play counts, the cards of the hand, draw pile,
discard pile, jokers (with cost, sell value, stickers and counters), consumables, shop,
vouchers, packs and open pack, and every RNG stream to 11 digits.

### Results

| | AnteZero simulator | their engine |
|---|---|---|
| given only seed, deck, stake and the actions | start state exact; diverges at entry 3 (the first blind); an action becomes illegal at entry 43 | cannot run this way |
| each transition tested alone (state restored from the log after a miss) | 273 of 299 exact, 7 snapshot timing, 19 Steamodded; all 298 actions legal | 0 discrepancies, on money, hands, discards, reroll cost, deck and discard counts and joker counters only, with every card on screen copied from the log first |

The 7 timing entries: the setup snapshot itself, a $1 payout still easing, and five entries
that are the game's own doing logged as actions (four packs opened by skip tags, one Boss
Tag reroll).

The 19 misses have two causes. Both are ways the logged game, which ran under an old
Steamodded build, departs from vanilla; the log does not say which version it was.

1. **Blind start with Marble Joker and Certificate (13 entries, every blind of the run).**
   The game shuffled before Marble Joker's Stone card joined the deck (the card is the
   bottom card of the logged draw pile at every blind); vanilla, and the simulator, shuffle
   after. Checked directly: with Marble Joker removed the simulator deals the logged hand
   and draw pile exactly. Certificate advanced the `stdseal<ante>` stream, which vanilla
   keeps for Standard Packs.

   **Neither is what Steamodded does today.** Checked in the real game on this PC
   (Steamodded 26.1002, seed `MARBCERT`, both jokers held, six blinds): the vanilla
   simulator matched all 24 transitions in every compared field, the deck's order and every
   RNG stream included. The Stone card is shuffled in, and no `stdseal` stream is rolled.
   The installed code agrees: `SMODS.poll_seal` is called with `guaranteed = true`, which
   skips the `stdseal` roll (`if guaranteed or pseudorandom(...)`), and Marble Joker's events
   are queued before the shuffle. Upstream's history has "make marble joker timing work like
   vanilla (#1386)" on 2026-05-20, and its current `poll_seal` (26.1121) is the same as the
   installed one. So the simulator's `steamodded` rules leave both jokers as in vanilla
   (`sim/tests/steamodded.rs` holds the real game's deal), and the 13 entries stay misses of
   this log.
2. **Sigil, Grim, Familiar, Incantation (6 entries).** Same RNG stream values, different
   suit or rank. The picks fit a suit list in the order Spades, Hearts, Clubs, Diamonds
   (Steamodded's suit table) where vanilla has Spades, Hearts, Diamonds, Clubs. This one
   depends on the version: the copy installed here uses the vanilla lists for these cards.
   Ouija, used once, matched.

I took the simulator to be right about vanilla in both cases (its oracle tests are for
that); I did not rerun the oracle on these cards.

Two more differences are harmless and ignored by the comparison: Steamodded keeps extra
`soul_smods_*` streams, and the art of the first shop's Buffoon Pack comes from
`math.random`.

The first divergence the brief asks for is therefore entry 3: the old build's blind start. With
5 ordinary jokers it would not happen; with Marble Joker in the lineup it happens every round.

The simulator's Steamodded rounding flag made no difference on this log.

### The real video

Doc's mod list (linked from every description) includes Steamodded, so the question is how
often these differences bite in real runs. In the 10 minutes tested: never. Marble Joker,
Certificate and the four spectral cards did not appear, and everything else matched: eight
packs (four Arcana, a Spectral with Immolate and Medium, a Standard whose Gold Seal ace is in
the final deck, two Buffoon), fourteen consumables used (The Fool, Strength, Judgement, The
Emperor, The High Priestess and others), Riff-Raff's four jokers, and The Hook's random
discards. That is one video; see the open points.

## 3. The combined pipeline

Stage 1 stays as it is. Stage 2 becomes a walk over the clicks with the simulator holding
the game:

1. **Start** `Game(seed, deck, stake)` with the Steamodded rules on (`--rules steamodded`).
   The description gives all three before any frame is decoded.
2. **For each click**, ask the simulator for the legal actions of that button.
   - One legal action (Cash Out, Next Round, Select Blind, Reroll, Skip Pack, a shop with one
     card): take it. No classifier runs. 33 of 90 clicks in the test clip.
   - Several: the hand detector's boxes say which cards are raised (their geometric test, no
     card classifier), and the item detector's boxes say which slot the button sits at.
3. **Check against the next screen** (`reconcile.settle`). Before the next click the HUD
   (money, round score, hands, discards, ante, round) is read and compared with the result
   of every legal action of that button, and of "nothing happened", each with the sales the
   joker bar shows (step 4). A result fits if no other has more of the HUD's numbers and the
   next click's button is legal after it: a Next Round click after which the shop's buttons
   are still clicked did nothing. If the fits differ in the joker bar, the consumables or
   the hand, those are read too. This is where wrong reads are repaired (3 sell slots in the
   clip) and clicks that did nothing are rejected (3 in the clip).
   - The read is recorded if it is one of the fits. If results other than its own fit as
     well, its line gets the comment `# unconfirmed: N other results fit the next screen`:
     it rests on the classifiers alone (12 of the clip's 87 lines, the first of them the
     fifth action, so cutting at the first one would leave nothing). A Skip Blind click is
     always one of these: no number or card read tells it from a click that did nothing.
   - A read that does not fit, or a click the classifiers read nothing of, is replaced only
     if all the fits have one result, and on the cards' word only if they are exactly the
     cards read: a partly matching hand never replaces a read.
   - The walk stops if the fits have several results, if they differ in a HUD number the
     OCR did not read (the round score tells a 324-chip straight from a 308-chip flush that
     both win the blind; without it nothing does), or if the next click's button is legal
     after none of them.
4. **Missed clicks.** A click that is not legal now may be legal after a forced step (Cash
   Out, Next Round, Skip Pack, Select Blind); that step is inserted. A joker missing from the
   bar while the money is up by its sell value is a missed sale (1 in the clip, the one their
   stage 2 flagged); none is inferred of a joker held twice, since the bar does not say which
   copy went. A click whose button is legal neither way stops the walk.
5. **Hold the result to the screen.** After the action (and any missed sale) every HUD
   number the OCR read must be the simulator's, and at least 4 of the 6 must have been read
   (the OCR leaves the ante and the round score unread on 4 of the clip's 90 screens). If
   not, the walk stops and the click is not recorded. The first screen is held to the new
   run the same way. The last click has no screen after it and is not recorded (the clip's
   last, a reroll).
6. **Write** the header and one line per action as a `.bal` file and replay it through
   `antezero_sim.StartPool` as the last check. The header ends in `rules steamodded` for a
   run played by Steamodded's rules, and a replay follows it (`Game.from_recording`); a
   header without it is a vanilla run. Recordings this walk wrote before the header had the
   rules were Steamodded runs with no mark and have to be relabelled, as the fixture was.
   The screen cannot tell the two rule sets apart (on the clip both give the same lines), so
   `--rules` has to be right. A walk that stopped ends in a `# stopped at` comment with the
   reason. Comments are not part of a recording: a start point's script does not keep them.

What happens to their models:

| model | today | with the simulator |
|---|---|---|
| card detector (hand) | every hand action | still needed: which cards are raised |
| card classifier (rank, suit, enhancement, edition, seal) | every hand action | check only, when the HUD leaves a tie |
| shop / joker / pack detector | every action | slot of a button; joker bar for missed sales |
| shop item classifier (which joker, which tarot) | every action | check only |
| HUD OCR | every action, overrides the engine | every action, as the check; never overrides |
| round-end video classifier | every round | not needed: the simulator knows which play wins |
| deck view detector + classifier | when the deck view is open | not needed (optional end-of-run check) |
| seed OCR | fallback | fallback |
| deck matcher | fallback | fallback |

**Seed unknown until the end.** Stage 1 does not use the seed, so it runs first either way.
Before stage 2, read the seed from the end screen (seeking to the tail costs seconds). A
misread seed shows in the first hand, which is not the one the simulator deals, so the few
look-alike readings (8/B, 5/S, 1/I, 2/Z) can be tried and the one whose first hand matches
kept. The walk does not make that comparison yet: with a wrong seed it stops at the first
play, whose score is not the simulator's, after recording the blind's selection and any
discards before it. A video with no readable seed cannot use the simulator and stays on
their stage 2; it also cannot become a `.bal` file, whose header is the seed.

**When the simulator and the video part ways** (an old Steamodded build's difference, a
drag not yet read, a classifier failure nothing repairs): the next screen is not the
simulator's game after any legal action. The walk stops there (steps 3 and 5) and keeps what
came before. This replaces their quarantine rules: a step is either reproduced by the
simulator or not written. What it cannot rule out is a wrong read that the next screen fits
as well as the right one; those lines are the ones marked `unconfirmed`, and a wrong one
among them stays in the recording when the walk stops later.

### Estimates

Accuracy. In the clip, 100% of 532 HUD checks (446 before the round score was read too)
and the final deck. On 2,859 actions of 8
human Gold Stake games (`antezero_bridge/measure.py`):

| what is read after the action | actions it pins to one legal action |
|---|---|
| HUD numbers only | 58.0% (plays 12.9%, discards 0%) |
| + the hand's cards | 81.7% (plays 68.4%, discards 96.9%) |
| + jokers, consumables, shop, pack | 86.6% |

So the screen cannot replace reading which cards are raised, but it checks the read: with
3.2% of actions misread by one card or one slot, the check repaired 75 of 91 and changed no
correct read, leaving 0.56% wrong. Most of those are a kicker in a round's last play, which
leaves the same game. (`measure.py` still breaks ties by the action nearest the read; the
walk now stops instead, so its share of wrong lines is lower and its share of stops is not
measured.) How often the walk reaches the end of a full run is the number to measure.

Speed. Stage 2 is no longer a cost: the prototype took 58 s for the clip on CPU (48 s with
the checks above; frame
decode 21.6, item detector 18.7, hand detector 9.1, simulator 6.7, OCR 1.8), unbatched and
with model loading. Total time stays stage 1. The simulator opens three savings there:
fewer frames (see the experiment below), no second decode pass if stage 1 hands the frames
on, and matching only the buttons legal in the simulator's phase. A fair target is half of
today's time on the same hardware; it has to be measured.

Frame rate experiment, same clip, CPU: stage 1 on every fourth frame (`--skip-frames 3`, 15
per second) took 241 s instead of 445 s (on 4 threads instead of 3), and the walk produced
the same recording with the same HUD agreements. On every sixth frame
(`--skip-frames 5`, 189 s) stage 1 missed a Discard click in the first round and the walk
lost the game: the prototype fills in missed forced clicks and missed sales, not missed
plays or discards. So half the frames is free on this clip, and going further needs plays
and discards inferred from the hand, which the simulator makes possible but which is not
built.

## 4. What was built

All under `cv/` in the worktree `C:\Users\Admin\Code\antezero-wt-cv`, branch `cv`.

| file | what |
|---|---|
| `cv/antezero_bridge/telemetry.py` | a telemetry log's start state and actions in the simulator; the field-by-field comparison |
| `cv/antezero_bridge/resync.py` | loads a logged state into the simulator after a miss |
| `cv/antezero_bridge/replay.py` | plays a log, scores every entry, names the Steamodded cause of a miss |
| `cv/antezero_bridge/codes.py` | the numbers of the simulator's injection codes |
| `cv/antezero_bridge/bal.py` | action codes to `.bal` lines and back; replay check |
| `cv/antezero_bridge/reconcile.py` | legal actions of a button, the ones that fit a screen, repair of a read |
| `cv/antezero_bridge/measure.py` | the table and repair numbers above, from `.bal` recordings |
| `cv/antezero_bridge/video.py` | the prototype stage 2: stage 1's clicks to a `.bal` file |
| `cv/testing/test_antezero_parity.py` | the parity test of section 2 |
| `cv/testing/test_antezero_bridge.py` | `.bal` round trip, repair, and the clip's recording against the video's deck view |
| `cv/testing/antezero_fixtures/wInI1W_EaEw_0-600.bal` | the 87 recorded actions of the clip |
| `cv/notes/antezero_phase1.md` | this report |

On disk but ignored: the clip `cv/assets/local/wInI1W_EaEw_0-600.mp4` (233 MB, the only
download) and `cv/output/` (stage 1 actions, their trajectory, the `.bal` files).

Environment: `C:\Users\Admin\Code\antezero-cv\.venv` (Python 3.12, CPU torch 2.15,
`antezero_sim` built from `C:\Users\Admin\Code\antezero\sim\py` with its build output kept
in the clone). From `cv/`, with `CUDA_VISIBLE_DEVICES=-1`:

```
PY=/c/Users/Admin/Code/antezero-cv/.venv/Scripts/python.exe
$PY -m unittest testing.test_antezero_parity testing.test_antezero_bridge
$PY testing/test_engine_parity.py --all          # their engine's own test
$PY -m antezero_bridge.measure <folder of .bal files> --games 8 --error 0.05
$PY main.py --input-video assets/local/wInI1W_EaEw_0-600.mp4 --full-video \
    --skip-post-processing --no-use-hwaccel --max-workers 3 \
    --actions-out output/vod_actions/clip.json --deck magic --stake gold --seed LNDW2KK8
PYTHONPATH=. $PY antezero_bridge/video.py assets/local/wInI1W_EaEw_0-600.mp4 \
    output/vod_actions/clip.json --seed LNDW2KK8 --deck b_magic --stake 8 --rules steamodded \
    --out output/clip.bal
```

## 5. Open: GPU, and the owner's decisions

Decisions:

1. **Old Steamodded builds in the simulator.** All three differences of the log (Marble
   Joker, Certificate, the picks of Sigil, Grim, Familiar and Incantation) are an old
   build's: the Steamodded installed here plays these cards as vanilla does (the two jokers
   checked in the real game, the spectral cards read in its code), and upstream's current
   code agrees for the two jokers. A video recorded on such a build stops where one of these
   cards acts. Marble Joker's changed on 2026-05-20; when the other two changed, and which
   build Doc ran on each video's date, I do not know. Modelling them would need a second set of rules for
   old builds, chosen per video; whether that is worth it depends on how many videos stop
   there (point 4).
2. **Unseeded videos**: keep their stage 2 for them, or leave them out.
3. **License**: upstream has none; nothing here can be published as is.

To measure, on the GPU or over a night of CPU:

4. Full videos: how many runs the walk follows to the end, and where the others stop. One
   10-minute clip is not a rate.
5. Stage 1 on every fourth frame on full videos, and with the decode on NVDEC.
6. Batched detectors on the GPU for the new stage 2 (expected: seconds per video).

Not verified, or not built:

- Joker and hand drags. The log replay turns them into `move_joker` / `move_hand`; the video
  prototype does not read them. Joker order changes scores, so the HUD check will catch it,
  but the order has to be read from the joker bar to repair it.
- Game speed. Doc's mods allow speeds above 4x and skipped animations; the simulator times
  events at 60 frames per second. No event-order race showed in the clip. The log's first
  divergence looked like one and turned out to be the setup quirk described above.
- The slot read for Sell buttons was off by one in 3 of 4 sales in the clip (repaired by the
  check each time); the geometry needs calibrating.
- Boss blinds with face-down cards, endless mode, runs that are lost: not exercised.
- The 5 to 6 minute figure is derived from their notes and defaults, not measured.
