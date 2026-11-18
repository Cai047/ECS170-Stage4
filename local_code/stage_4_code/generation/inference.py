"""CPU-only decoding from a Stage 4 checkpoint and a new three-word opening."""
from pathlib import Path
import math
import re
import unicodedata

import torch

from .data import BOS, EOS, PAD, UNK, tokenize
from .model import GenerationModel, render_text


def load_generator(path, cell):
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    config, vocab = saved["config"], saved["vocab"]
    if config["cell"] != cell:
        raise ValueError(f"Expected a {cell.upper()} checkpoint, found {config['cell'].upper()}")
    model = GenerationModel(cell, len(vocab), config["embedding"], config["hidden"])
    model.load_state_dict(saved["state_dict"])
    return model.eval(), vocab


def normalized_score(score, length):
    """A mild length penalty avoids always favoring the shortest completion."""
    return score / (((5.0 + length) / 6.0) ** 0.7)


def allowed_scores(log_probs, history, blocked, eos, word_ids, last_position):
    scores = log_probs.clone()
    scores[blocked] = -torch.inf
    has_word = any(word_ids[idx] for idx in history)
    if len(history) < 3 or not has_word:
        scores[eos] = -torch.inf
    if last_position and not has_word:
        scores[~word_ids] = -torch.inf
    # Prevent a generated three-token phrase from repeating verbatim in the continuation.
    if len(history) >= 3:
        final_pair = history[-2:]
        banned = [c for a, b, c in zip(history, history[1:], history[2:])
                  if (a, b) == final_pair]
        scores[banned] = -torch.inf
    return scores


def display_continuation(prompt, generated):
    # Keep the original opening. Apply ordinary English casing to generated words.
    words = []
    sentence_start = prompt.endswith((".", "!", "?"))
    for token in generated:
        if token == "i" or token.startswith("i'") or (sentence_start and token[0].isalpha()):
            token = token[:1].upper() + token[1:]
        words.append(token)
        if any(c.isalnum() for c in token):
            sentence_start = False
        if token in (".", "!", "?"):
            sentence_start = True
    return prompt + render_text(["", *words])


@torch.inference_mode()
def generate_joke(model, vocab, prompt, strategy="beam", beam_width=5, max_tokens=32,
                  seed=170, top_k=15, temperature=0.85):
    """Decode from the recurrent model; no dataset or reference text is consulted."""
    if strategy not in {"beam", "greedy", "sample"}:
        raise ValueError("strategy must be beam, greedy or sample")
    if beam_width < 1 or max_tokens < 3 or top_k < 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Use beam_width/top_k >= 1, max_tokens >= 3 and finite positive temperature")
    prompt = prompt.strip()
    # Hyphenated words and contractions count as one input word. Model preprocessing
    # still uses its own training tokenizer, which may split that word into subparts.
    words = re.findall(r"[^\W_]+(?:['’‐‑-][^\W_]+)*", unicodedata.normalize("NFKC", prompt))
    if len(words) != 3:
        raise ValueError(f"Expected exactly three starting words; received {len(words)}")
    stoi = {token: i for i, token in enumerate(vocab)}
    tokens = tokenize(prompt)
    unknown = list(dict.fromkeys(token for token in tokens if token not in stoi))
    ids = [stoi[BOS]] + [stoi.get(token, stoi[UNK]) for token in tokens]
    blocked = [stoi[token] for token in (PAD, UNK, BOS)]
    eos = stoi[EOS]
    word_ids = torch.tensor([idx not in blocked and idx != eos and any(c.isalnum() for c in token)
                             for idx, token in enumerate(vocab)], dtype=torch.bool)
    model.eval()
    logits, initial_state = model(torch.tensor([ids]))
    # Each live hypothesis holds the state AFTER consuming its own prefix and the
    # logits predicting its next token. This also handles the LSTM's (h,c) state.
    live = [((), 0.0, initial_state, logits[0, -1])]
    finished = []
    rng = torch.Generator(device="cpu").manual_seed(seed)
    width = beam_width if strategy == "beam" else 1
    for step in range(max_tokens):
        next_live = []
        for history, score, state, logits in live:
            scale = temperature if strategy == "sample" else 1.0
            probs = (logits.float() / scale).log_softmax(-1)
            probs = allowed_scores(probs, history, blocked, eos, word_ids, step == max_tokens - 1)
            if strategy == "beam" and torch.isfinite(probs[eos]):
                end_score = normalized_score(score + probs[eos].item(), len(history) + 1)
                finished.append((end_score, history))
                probs[eos] = -torch.inf
            k = top_k if strategy == "sample" else width
            values, indices = probs.topk(min(k, int(torch.isfinite(probs).sum())))
            if not indices.numel():
                continue
            if strategy == "sample":
                pick = torch.multinomial(values.softmax(0), 1, generator=rng)
                values, indices = values[pick], indices[pick]
            for v, idx in zip(values.tolist(), indices.tolist()):
                if idx == eos:
                    finished.append((normalized_score(score + v, len(history) + 1), history))
                else:
                    next_live.append((history + (idx,), score + v, state))
        best = sorted(next_live, key=lambda item: item[1], reverse=True)[:width]
        if not best:
            break
        if step == max_tokens - 1:
            break
        live = []
        for history, score, state in best:
            logits, updated = model(torch.tensor([[history[-1]]]), state)
            live.append((history, score, updated, logits[0, -1]))
    if finished:
        _, selected = max(finished, key=lambda item: item[0])
        stop = "eos"
    elif best:
        selected = best[0][0]
        stop = "length"
    else:
        raise ValueError("No continuation could be decoded from this checkpoint")
    generated = [vocab[i] for i in selected]
    return {"prompt": prompt, "prompt_tokens": tokens, "unknown_tokens": unknown,
            "strategy": strategy, "generated_tokens": generated, "stop": stop,
            "text": display_continuation(prompt, generated)}
