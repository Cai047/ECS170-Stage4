"""Matched recurrent language models; only the recurrent cell varies."""
import re

import torch
from torch import nn

from local_code.base_class.method import method
from .data import BOS, EOS, PAD, UNK, tokenize


class GenerationModel(nn.Module, method):
    def __init__(self, cell, vocab_size, embedding, hidden):
        nn.Module.__init__(self)
        method.__init__(self, cell.upper(), "Word-level recurrent language model")
        self.embedding = nn.Embedding(vocab_size, embedding, padding_idx=0)
        options = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        self.recurrent = options[cell](embedding, hidden, batch_first=True)
        self.decoder = nn.Linear(hidden, vocab_size)

    def forward(self, ids, state=None):
        h, state = self.recurrent(self.embedding(ids), state)
        return self.decoder(h), state

    def run(self, ids, state=None):
        return self(ids, state)


def render_text(tokens):
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%)\]])", r"\1", text)
    return re.sub(r"([(\[])\s+", r"\1", text)


@torch.inference_mode()
def sample_story(model, vocab, prompt, seed, max_tokens, top_k, temperature):
    """Generate a continuation from exactly three in-vocabulary words."""
    stoi = {t: i for i, t in enumerate(vocab)}
    tokens = tokenize(prompt)
    if len(tokens) != 3 or any(t not in stoi or not any(c.isalnum() for c in t) for t in tokens):
        raise ValueError(f"Prompt must contain exactly three words in training vocabulary: {prompt!r}")
    model.eval()
    device = next(model.parameters()).device
    rng = torch.Generator(device="cpu").manual_seed(seed)
    # BOS marks the beginning of a new prompt, just as it does for a training fragment.
    ids = torch.tensor([[stoi[BOS]] + [stoi[t] for t in tokens]], device=device)
    state, continuation, stop = None, [], "length"
    for _ in range(max_tokens):
        logits, state = model(ids, state)
        scores = logits[0, -1].float().cpu() / temperature
        scores[[stoi[PAD], stoi[UNK], stoi[BOS]]] = -torch.inf
        candidates, indices = scores.topk(min(top_k, len(vocab)))
        idx = indices[torch.multinomial(candidates.softmax(0), 1, generator=rng)].item()
        if idx == stoi[EOS]:
            stop = "eos"
            break
        continuation.append(vocab[idx])
        ids = torch.tensor([[idx]], device=device)
    return {"prompt": prompt, "prompt_tokens": tokens, "generated_tokens": continuation,
            "text": render_text(tokens + continuation), "seed": seed, "stop": stop}
