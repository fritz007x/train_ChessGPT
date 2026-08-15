"""Move selection for ChessGPT.

Instead of sampling characters one at a time and hoping the result parses as a legal
move, we enumerate every legal move with python-chess, ask the model how likely each
complete move is, and play the best one. This makes illegal output impossible, removes
sampling noise, and avoids committing to a bad prefix (greedy character decoding can
pick "R" and then discover every rook move is bad).

The model was trained on transcripts like ";1.e4 e5 2.Nf3 Nc6 ", so a candidate move is
scored as the continuation `san + " "` after the running game string. The trailing space
matters: it forces the model to commit to ending the move there rather than leaking
probability onto longer moves ("Nf3" vs "Nf3+").

Two scorers are provided and they must always agree:

  "naive"  - one padded row per candidate, each a full copy of the prefix. Simple enough
             to be obviously correct, and used as the test oracle. Cost grows with the
             number of legal moves, which on CPU is slow (~4s/move in the middlegame).
  "packed" - the default. The prefix is laid down once and every candidate suffix is
             appended after it in a single sequence, with an attention mask that lets each
             suffix see the prefix and itself but not the other candidates, and position
             ids that restart each suffix right after the prefix. One forward pass over
             roughly (prefix + 5 chars per candidate) tokens instead of one per candidate.
"""
import torch
import torch.nn.functional as F


def _spellings(san):
    """The SAN spellings this model might use for a move.

    python-chess writes checkmate as "Qxf7#", but the training transcripts spell it
    "Qxf7+": measured after a mating stem, the model puts p=0.999 on "+" and p=0.0005 on
    "#". Scoring only the "#" form makes every mate look ~10 nats unlikely, and the engine
    then refuses to deliver checkmate - so both spellings are scored and their probability
    summed. Castling ("O-O") and promotion ("e8=Q") were checked and match python-chess.
    """
    if san.endswith("#"):
        return [san, san[:-1] + "+"]
    return [san]


def _prepare(board, game_str, encode, block_size):
    """Legal moves plus token ids for the shared prefix and every candidate spelling.

    Returns flat `suffix_ids` with `owners[k]` giving the index of the move that spelling
    k belongs to, so a move with several spellings is scored once per spelling.
    """
    moves = list(board.legal_moves)
    sans = [board.san(m) for m in moves]
    prefix_ids = encode(game_str)
    suffix_ids, owners = [], []
    for i, san in enumerate(sans):
        for variant in _spellings(san):
            suffix_ids.append(encode(variant + " "))
            owners.append(i)

    # Keep prefix + longest suffix inside the model's context. Cropping drops the leading
    # ";" and the opening, which is off-distribution, but only bites past ~90 turns.
    longest = max((len(s) for s in suffix_ids), default=1)
    limit = block_size - longest
    if len(prefix_ids) > limit:
        prefix_ids = prefix_ids[len(prefix_ids) - limit:]
    return moves, sans, prefix_ids, suffix_ids, owners


def _token_logprobs(logits, x):
    """lp[b, j] = log P(x[b, j+1] | x[b, :j+1]) for a batch of token id rows."""
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp[:, :-1, :].gather(2, x[:, 1:].unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def _score_naive(model, prefix_ids, suffix_ids, device, chunk):
    L = len(prefix_ids)
    seqs = [prefix_ids + s for s in suffix_ids]
    width = max(len(s) for s in seqs)
    scores = []
    for start in range(0, len(seqs), chunk):
        part = seqs[start:start + chunk]
        # right-padding is safe: causal attention means pad tokens only influence positions
        # after them, and those positions are never scored.
        x = torch.zeros(len(part), width, dtype=torch.long, device=device)
        for row, s in enumerate(part):
            x[row, :len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        logits, _ = model(x, targets=x)  # targets given only to get logits at every position
        lp = _token_logprobs(logits, x)
        for row, s in enumerate(part):
            # score target tokens L .. len(s)-1, which live in columns L-1 .. len(s)-2
            scores.append(lp[row, L - 1:len(s) - 1].sum().item())
    return scores


@torch.no_grad()
def _score_packed(model, prefix_ids, suffix_ids, device):
    L = len(prefix_ids)
    lengths = [len(s) for s in suffix_ids]
    total = L + sum(lengths)

    ids = list(prefix_ids)
    positions = list(range(L))
    starts = []
    for suf in suffix_ids:
        starts.append(len(ids))
        ids.extend(suf)
        positions.extend(range(L, L + len(suf)))  # every suffix restarts right after the prefix

    x = torch.tensor([ids], dtype=torch.long, device=device)
    pos = torch.tensor(positions, dtype=torch.long, device=device)

    # A candidate must see the prefix and its own characters, never another candidate:
    # plain causality would let a later candidate read the earlier ones.
    ar = torch.arange(total, device=device)
    mask = torch.zeros(total, total, dtype=torch.bool, device=device)
    mask[:L, :L] = ar[:L].unsqueeze(1) >= ar[:L].unsqueeze(0)
    for st, ln in zip(starts, lengths):
        mask[st:st + ln, :L] = True
        own = ar[st:st + ln]
        mask[st:st + ln, st:st + ln] = own.unsqueeze(1) >= own.unsqueeze(0)

    logits, _ = model(x, targets=x, attn_mask=mask.view(1, 1, total, total), pos=pos)
    logp = F.log_softmax(logits[0].float(), dim=-1)

    scores = []
    for st, suf in zip(starts, suffix_ids):
        # the first character of every candidate is predicted by the last prefix position,
        # so all candidates share that one distribution; later characters are predicted by
        # the packed position of the character before them.
        total_lp = logp[L - 1, suf[0]]
        for j in range(1, len(suf)):
            total_lp = total_lp + logp[st + j - 1, suf[j]]
        scores.append(total_lp.item())
    return scores


@torch.no_grad()
def score_legal_moves(model, encode, device, board, game_str, method="packed", chunk=32):
    """Score every legal move, best first.

    Returns a list of dicts with "move" (chess.Move), "san", "logprob" (total log
    probability of the move's characters) and "prob" (that renormalised over the legal
    moves, i.e. the model's implied distribution given it must play something legal).
    """
    if not board.legal_moves:
        return []
    block_size = model.config.block_size
    moves, sans, prefix_ids, suffix_ids, owners = _prepare(board, game_str, encode, block_size)
    assert len(prefix_ids) >= 1, "game string must be non-empty (it always starts with ';')"

    if method == "packed":
        scores = _score_packed(model, prefix_ids, suffix_ids, device)
    elif method == "naive":
        scores = _score_naive(model, prefix_ids, suffix_ids, device, chunk)
    else:
        raise ValueError(f"unknown scoring method: {method}")

    # a move's score is the total probability of it being written any of its spellings
    per_move = [[] for _ in moves]
    for score, owner in zip(scores, owners):
        per_move[owner].append(score)
    totals = [torch.logsumexp(torch.tensor(v), dim=0).item() for v in per_move]

    probs = F.softmax(torch.tensor(totals), dim=0).tolist()
    scored = [{"move": m, "san": s, "logprob": lp, "prob": p}
              for m, s, lp, p in zip(moves, sans, totals, probs)]
    scored.sort(key=lambda d: d["logprob"], reverse=True)
    return scored


def pick_move(model, encode, device, board, game_str, move_temperature=0.0, method="packed", chunk=32):
    """Choose a move. Returns (chess.Move, san, scored_candidates).

    move_temperature == 0 plays the best move, which is the strongest setting. Higher
    values sample over whole moves - a principled variety knob, unlike character-level
    temperature, since every option is still a legal move.
    """
    scored = score_legal_moves(model, encode, device, board, game_str, method=method, chunk=chunk)
    if not scored:
        return None, None, []
    if move_temperature <= 0:
        choice = scored[0]
    else:
        weights = torch.tensor([d["logprob"] for d in scored]) / move_temperature
        choice = scored[torch.multinomial(F.softmax(weights, dim=0), 1).item()]
    return choice["move"], choice["san"], scored
