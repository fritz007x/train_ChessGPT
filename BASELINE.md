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

## Phase 3 pre-flight: is the hanging-piece filter worth building?

Measured 2026-08-21, before writing any of it. The planned Phase 3 rests on a claim nobody had
checked - that the dominant failure of 1-ply policy play is moving a piece somewhere it is
simply lost - and Phase 1 had already inverted one such intuition, so this one was measured
first.

A fresh 40-game corpus vs `sf:nodes=1` (7W-16D-17L, score 0.375, Elo **-89** [-194, +3])
reproduces the baseline row above. Each of the engine's **1765 moves** was then re-scored under
the model and evaluated by Stockfish at depth 12 before and after. Two instrument checks passed
first: the rebuilt transcript prefix matches `test_engine.transcript`, and the top-scoring move
equals the move actually played in **1765 of 1765** positions, which is what determinism should
look like.

> Book plies are not engine moves - `match.py` plays the opening itself and never consults
> either player (`match.py:292-295`). Counting them costs nothing in the match but silently adds
> positions the model never chose; excluding them is what makes 1765 agree exactly with the move
> count `match.py` reports.

### Spelling audit (Phase 2): clean, no change to `_spellings`

`engine._spellings` (`engine.py:29`) knows only `#`->`+`. Every other spelling the model might
plausibly prefer was scored against python-chess's own, in the same packed forward pass, over
~440 sampled positions. Delta is `logprob(alternate) - logprob(canonical)` in nats, so positive
would mean a leak:

| bucket | example | n | mean delta | alternate wins |
|---|---|---|---|---|
| `mate_as_plus` (**control**) | `Qf7#`->`Qf7+` | 3 | **+9.30** | 100% |
| under-disambiguated | `Ree7`->`Re7` | 464 | -9.29 | 1.3% |
| over-disambiguated (file) | `Nb4`->`Ndb4` | 1939 | -17.70 | 0.1% |
| over-disambiguated (rank/full) | `Ba2`->`B6a2` | 3870 | -18.7 / -34.0 | 0.0% |
| check suffix dropped | `Rb7+`->`Rb7` | 258 | -16.68 | 0.0% |
| capture `x` dropped | `gxh6`->`gh6` | 72 | -23.77 | 0.0% |
| promotion `=` dropped | `d8=Q`->`d8Q` | 12 | -32.66 | 0.0% |
| castling as digits | `O-O`->`0-0` | 19 | -53.64 | 0.0% |

The control reproduces the already-known mate effect at +9.30 nats, which is what says the audit
is measuring anything at all. **Nothing else leaks.** The model writes SAN exactly as
python-chess does, including disambiguation - the bucket that was never checked and that a
tactical filter would lean on hardest. Phase 2 closes with no code change.

### Blunder census: the premise does not hold

Of 1765 engine moves, 6.4% lose at least 200cp at depth 12 (5.9% restricted to live positions,
`|eval| < 800`; half the corpus is adjudicated and pinned at the clamp, where a drop is
mechanically near zero). Breaking down those 113 blunders:

| what the blunder is | moves | share |
|---|---|---|
| the moved piece lands where a swap-off wins it | 16 | **14.2%** |
| the move leaves something else newly takeable | 6 | 5.3% |
| something was already takeable and the move ignored it | 10 | 8.8% |
| no material hanging at all - positional, or tactics deeper than one exchange | 81 | **71.7%** |

**Only 14% of the engine's blunders are the kind a destination-square SEE filter can see**, and
they account for 10% of the centipawns it throws away. The dominant failure mode is not hanging
pieces; it is play that is simply worse than it looks one exchange deep. The Phase 3 premise is
wrong - with the caveat that this is one opponent's worth of evidence, `sf:nodes=1`, and whether
the mix shifts against stronger opposition is untested. The 86%-out-of-reach margin is wide
enough that it would have to shift a long way to change the verdict.

### What the filter would actually do

Simulated over the same 1765 positions (reject a candidate with SEE <= -100, take the best
surviving move from the top 5):

- it fires on **54 moves (3.1%)**, taking the second-choice move in 47 of those
- when it fires it gains **+30cp on average, 95% CI [-53, +113]** - the interval contains zero
- the replacement is better 27 times, worse 24, identical 3 (sign test p = 0.39)

It does work on the cases it was designed for: on the 16 firings that were real blunders it
recovers +5575cp of the 6217 lost there. But the other 38 firings give almost all of it back -
positions where the model was playing a sound sacrifice, or where a swap-off eval misjudges the
position. **False positives cost about what the true positives gain.**

Making the trigger stricter does not rescue it. Raising the SEE bar, requiring the policy to be
nearly indifferent, or both, all raise the gain *per firing* while cutting the firing rate by
about as much:

| trigger | firings | mean gain | 95% CI | per engine move |
|---|---|---|---|---|
| SEE <= -100 (as specified) | 54 | +30cp | [-53, +113] | +0.91 cp |
| SEE <= -300 | 16 | +115cp | [-125, +354] | +1.04 cp |
| policy cost <= 1.0 nats | 34 | +62cp | [-44, +167] | +1.19 cp |
| SEE <= -300 and cost <= 1.0 nats | 11 | +161cp | [-102, +424] | +1.00 cp |

Every row sits between +0.3 and +1.2 cp per engine move and every interval spans zero. (One
narrower slice, SEE <= -500 with cost <= 1.0 nats, does exclude zero at n=7 - across fourteen
thresholds tried, that is what chance produces, and it is not a result.)

### Converting that to Elo, and why the answer is a range

`second:gap=1.0` gives a fixed point: BASELINE measured it at -117 Elo. It fires on 52.1% of
moves and gives up a mean of only 3cp per override, i.e. -1.4 cp/move. Scaling the filter's
+0.9 cp/move by that exchange rate puts it at **+74 Elo**.

Do not read that as the expected gain. It is the optimistic tail of a mean whose own interval is
[-53, +113] cp, and the filter's *median* firing gains +2cp - run the same conversion off the
medians and the answer is +3 Elo. The point estimate is not distinguishable from zero; +74 is
what the top of the interval would be worth if it were real. The conversion factor is doing
enormous work either way: a one-ply depth-12 eval barely separates the policy's top two moves
(3cp) even though consistently taking the second one costs over 100 Elo, so none of this is
trustworthy to better than an order of magnitude. Sized against the power table above and
inflated for the 11 of 40 games in which the filter never fires at all - those play out
byte-identically and score exactly 0.500, carrying no information - even the optimistic +74
would need **~90 games** to clear zero, and anything near the point estimate is unmeasurable at
any practical number of games.

### Verdict: Phase 3 does not ship; go to Phase 4

Three independent readings agree, and none of them depends on the shaky Elo conversion:

1. **86% of the engine's blunders are out of the filter's reach** by construction.
2. **Its measured benefit contains zero** at every trigger threshold tried.
3. **No threshold improves cp/move** - selectivity trades firing rate against gain at par.

The 72% "positional or deeper than one exchange" bucket is the actual target, and it is exactly
what a value head (Phase 4) and policy-pruned search (Phase 5) address. Phase 4 becomes the next
step; the SEE filter is not worth the several CPU-hours its gate match would cost, and the
`engine.py` complexity it would add is complexity spent on a seventh of the problem.

Two things from this work are worth keeping if Phase 5 ever needs them: the SEE routine itself
(python-chess ships none; `attackers_mask(color, square, occupied)` takes a custom occupancy, so
x-rays behind a departing attacker come out right), and the finding that the model's SAN
spelling needs no special-casing beyond the mate suffix.

## Notes for re-running

- Matches are adjudicated once Stockfish sees ±900cp for 4 consecutive plies, which roughly
  halves wall-clock time. Pass `--adjudicate=off` to play everything out.
- Launching several Stockfish processes at the same instant intermittently fails on Windows
  with exit code `0xC0000142`; `open_engine()` retries with backoff, and parallel matches
  should still be staggered by a few seconds.
