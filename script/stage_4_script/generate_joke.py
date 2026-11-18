"""Generate a joke from any three-word opening with a packaged CPU checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import torch

from local_code.stage_4_code.generation.inference import generate_joke, load_generator


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", choices=["rnn", "lstm", "gru"], default="lstm")
    p.add_argument("--prompt", required=True, help="Three starting words, in quotes")
    p.add_argument("--results-dir", type=Path, default=Path("results/generation_wide"))
    p.add_argument("--strategy", choices=["beam", "greedy", "sample"], default="beam")
    p.add_argument("--beam-width", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=32, help="Maximum continuation length in model tokens")
    p.add_argument("--seed", type=int, default=170, help="Sampling seed")
    p.add_argument("--top-k", type=int, default=15, help="Candidate count for sampling")
    p.add_argument("--temperature", type=float, default=0.85, help="Temperature for sampling")
    p.add_argument("--json", action="store_true", help="Print a machine-readable record")
    args = p.parse_args()
    path = args.results_dir / args.model / "checkpoint.pt"
    if not path.is_file():
        p.error(f"Missing checkpoint: {path}")
    torch.set_num_threads(1)
    model, vocab = load_generator(path, args.model)
    try:
        record = generate_joke(model, vocab, args.prompt, args.strategy, args.beam_width,
                               args.max_tokens, args.seed, args.top_k, args.temperature)
    except ValueError as e:
        p.error(str(e))
    if record["unknown_tokens"]:
        unknown = ", ".join(record["unknown_tokens"])
        print(f"Out-of-vocabulary prompt tokens encoded as <unk>: {unknown}", file=sys.stderr)
    print(json.dumps(record, ensure_ascii=False) if args.json else record["text"])


if __name__ == "__main__":
    main()
