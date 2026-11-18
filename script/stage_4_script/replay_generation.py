"""Replay the saved continuations and test metrics from the submitted CPU checkpoints."""
import argparse
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from local_code.stage_4_code.generation.data import GenerationDataset, language_windows
from local_code.stage_4_code.generation.model import GenerationModel, sample_story
from local_code.stage_4_code.generation.training import pass_over


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Original course ID,Joke data file")
    parser.add_argument("--results-dir", type=Path, default=Path("results/generation"))
    args = parser.parse_args()
    recorded = read_json(args.results_dir / "dataset.json")
    corpus = GenerationDataset(args.data, max_vocab=recorded["max_vocab"],
                               min_freq=recorded["min_freq"]).load()
    if corpus.metadata != recorded:
        raise ValueError("Corpus, vocabulary settings or data split do not match this experiment")

    for name in ("rnn", "lstm", "gru"):
        folder = args.results_dir / name
        saved = torch.load(folder / "checkpoint.pt", map_location="cpu", weights_only=True)
        scores = read_json(folder / "metrics.json")
        cfg = saved["config"]
        if saved["vocab"] != corpus.vocab or cfg != scores["config"] or cfg["cell"] != name:
            raise ValueError(f"{name}: inconsistent checkpoint vocabulary or configuration")
        torch.set_num_threads(cfg["threads"])
        model = GenerationModel(name, len(corpus.vocab), cfg["embedding"], cfg["hidden"])
        model.load_state_dict(saved["state_dict"])
        test = DataLoader(language_windows(corpus.test_tokens, corpus.vocab, cfg["context"]),
                          batch_size=cfg["batch_size"])
        actual = pass_over(model, test, torch.device("cpu"))
        for metric, value in actual.items():
            if not math.isclose(value, scores["test"][metric], rel_tol=1e-5, abs_tol=1e-6):
                raise ValueError(f"{name}: mismatch for test {metric}")
        samples = [sample_story(model, corpus.vocab, prompt, seed, cfg["sample_tokens"],
                                cfg["top_k"], cfg["temperature"])
                   for prompt in cfg["prompts"] for seed in cfg["sample_seeds"]]
        if samples != read_json(folder / "samples.json"):
            raise ValueError(f"{name}: sampled continuations differ from the recorded run")
        print(f"{name.upper()}: verified {actual['targets']} test targets and {len(samples)} samples")


if __name__ == "__main__":
    main()
