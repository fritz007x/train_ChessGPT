# Baseline strength of the current engine

Measured with `match.py` on 2026-08-15, against `out-chess-16layer/ckpt.pt`
(`iter_num=600000`, `best_val_loss=0.211`). Every match is 20 book openings played twice
with the colours reversed, on CPU, at `move_temperature=0`.

Reproduce any row with:

```
python match.py --white policy --black sf:elo=1320 --openings 20
```

**One caveat on exact reproduction:** the four Stockfish rows were measured before the
adjudication threshold was flattened to a constant (ply 20; it previously scaled with the
opening length, starting at ply 14 for 2-ply book lines and 18 for 4-ply ones). Re-running
today can therefore shift a game or two. The mild-degradation match further down ran on the
current code.

## Where the engine sits

| Opponent | Blunders on purpose? | W-D-L | Score | Elo diff | 95% CI (paired) |
|---|---|---|---|---|---|
| `sf:elo=1320` (UCI_LimitStrength) | yes | 32-2-6 | 0.825 | +269 | [+161, +458] |
| `sf:skill=0` | yes | 24-8-8 | 0.700 | +147 | [+38, +295] |
| `sf:nodes=1` | no | 7-17-16 | 0.388 | **-80** | [-183, +11] |
| `sf:nodes=100` | no | 3-19-18 | 0.312 | **-137** | [-227, -62] |

**Use the two node rows as the baseline.** They are deterministic and Stockfish is playing
its best move at that budget, so the result means what it looks like. `nodes=1` is close to
the engine's own level (-80, interval touching zero), which makes it the natural yardstick.

**The absolute Elo number is soft, and the top two rows are why.** `UCI_LimitStrength` and
`Skill Level` both weaken Stockfish by deliberately choosing worse moves - and a policy
engine feeds on that, so the score against them inflates. The two blunder-injecting rows
also disagree in shape (2 draws vs 8) despite measuring the same engine. Taking the
`elo=1320` row at face value would put the engine near 1590; the honest statement is that
it is **somewhere in the 1400-1600 region, on Stockfish's own self-assessment of an
opponent that is throwing games away on purpose**. Don't quote a point estimate.

Pairing the colour-reversed games (the statistically correct treatment, since both games of
a pair share an opening and both engines) turned out to change the intervals by only a few
Elo in either direction - against Stockfish the two halves of a pair diverge too much to be
correlated. `match.py` reports both; the paired figure is above, computed with a t critical
value rather than 1.96 since it rests on only 20 data points.

Speed: **~2.2 s/move** on CPU with 4 threads and four matches running at once; ~1.0 s/move
with a match to itself. A 40-game match takes 45-70 minutes.

## Findings

**Repetition is a resource for this engine, not a leak.** 16 of 40 games vs `nodes=1` and
18 of 40 vs `nodes=100` ended in threefold repetition, which looks like the classic
deterministic-engine failure of shuffling away a won game. It is the opposite. Evaluating
every repetition-drawn final position with Stockfish at depth 16:

| Match | Model winning (>+150cp) | Roughly equal | Model losing (<-150cp) |
|---|---|---|---|
| `sf:nodes=1` | **0** | 2 | 14 |
| `sf:nodes=100` | **0** | 4 | 14 |
| `sf:skill=0` | **0** | 3 | 3 |

Not one repetition draw came from a winning position. The engine repeats when it is *lost*,
and those draws are salvaged half-points. Any future "avoid repetition" heuristic would
convert saved draws into losses and make the engine measurably weaker - so don't add one
without measuring it first.

**Stockfish node limits are stronger than they sound.** `nodes=1` is roughly the engine's
equal. There is no node setting weak enough to bracket the engine from below, which is why
`UCI_Elo` (floor 1320) and `Skill Level` exist as player specs.

## Harness verification

The numbers above are only worth as much as the instrument, so it was calibrated first:

| Check | Result |
|---|---|
| Reproducibility - same match run twice | **PASS** - game-for-game and PGN byte-identical |
| Sensitivity - vs a deliberately crippled variant (`second`, plays its 2nd-choice move) | **PASS** - 20-0, LOS 100% |
| Null test - engine against itself | **PASS** - exactly 0.500, colour-reversed pairs identical |
| Monotonicity - stronger opponent must score better | **PASS** - `nodes=1` -80 → `nodes=100` -137 |
| Engine regressions - `python test_engine.py` | **PASS** - all packed-vs-naive, permutation and mate-spelling checks |

The null test earned its place: the first run scored 0.500 but colour-reversed pairs ended
on *different* plies (79/77, 100/98) despite being the same deterministic game. Cause:
python-chess only emits `ucinewgame` when the `game` argument changes, so Stockfish carried
its transposition table across the entire match - inflating a node-limited opponent and
making results depend on the order games were played in. Fixed by handing every game a
fresh token (`match.py`, `play_game`); pairs now end on identical plies.

## How big a change can this actually detect?

The 20-0 crippled-variant test proves the harness detects a catastrophe. It says nothing
about the 30-80 Elo changes Phases 3-5 are likely to produce, so that was measured too:
`policy` vs `second:gap=1.0`, a variant that gives up the best move only when the top two
candidates are within 1.0 nats.

```
policy vs second:gap=1.0    23W 7D 10L in 40 games
                            Elo +117  [+17, +242]   LOS 99.3%
```

A ~117 Elo difference is resolved at 40 games, but only just - the interval nearly touches
zero. The per-game standard deviation in that match is **0.430**, which sizes everything
else (figures below are for variant-vs-variant A/B, the Phase 3-5 gate; the Stockfish rows
above have a different variance structure and should not be sized from this table):

| True effect | Games to exclude zero | Games for 80% power |
|---|---|---|
| 100 Elo | 36 | 74 |
| 80 Elo | 55 | 113 |
| 50 Elo | 139 | 283 |
| 30 Elo | 382 | 780 |

**So 40 games is a ~100 Elo instrument.** The book has 50 openings (100 games) available,
which covers an 80 Elo change properly. A 50 Elo change needs 130+ games to clear zero and
~270 to detect reliably - at ~2 min/game for model-vs-model that is 4-9 hours of CPU. Decide
the expected effect size *before* running a phase gate, and don't read a 40-game null result
as "the change didn't help".

Pairing the colour-reversed games does not rescue this: it moved the interval from
[+20, +236] to [+23, +232], i.e. not at all. The variance is between openings, not between
the two colours of one opening.

## Notes for re-running

- Matches are adjudicated once Stockfish sees ±900cp for 4 consecutive plies, which roughly
  halves wall-clock time. Pass `--adjudicate=off` to play everything out.
- Launching several Stockfish processes at the same instant intermittently fails on Windows
  with exit code `0xC0000142`; `open_engine()` retries with backoff, and parallel matches
  should still be staggered by a few seconds.
