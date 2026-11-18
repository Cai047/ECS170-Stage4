"""Train the three Stage 4 language models and generate matched joke samples."""
import argparse
from pathlib import Path

from local_code.stage_4_code.generation.training import GenerationSetting


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", required=True, type=Path, help="Path to the official extensionless ID,Joke CSV")
    p.add_argument("--test-data", type=Path, help="Separate ID,Joke test CSV, only when an external split is supplied")
    p.add_argument("--output", default=Path("results/generation"), type=Path)
    p.add_argument("--models", nargs="+", choices=["rnn", "lstm", "gru"], default=["rnn", "lstm", "gru"])
    prompts = p.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", help="Three words that start a training joke")
    prompts.add_argument("--prompts", nargs="+", help="Quoted three-word training joke openings")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--context", type=int, default=32, help="Maximum tokens per recurrent training window")
    p.add_argument("--embedding", type=int, default=48)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--max-vocab", type=int, default=3000, help="Maximum total entries including reserved IDs")
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=0.003)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=170)
    p.add_argument("--sample-seeds", nargs="+", type=int, default=[170, 171, 172])
    p.add_argument("--sample-tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.85)
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    args.prompts = args.prompts or [args.prompt]
    del args.prompt
    for name in ("epochs", "batch_size", "context", "embedding", "hidden", "max_vocab", "min_freq",
                 "learning_rate", "clip", "sample_tokens", "temperature", "top_k", "threads"):
        if getattr(args, name) <= 0:
            p.error(f"{name} must be positive")
    if len(set(args.models)) != len(args.models):
        p.error("Specify each model only once")
    GenerationSetting(args).load_run_save_evaluate()


if __name__ == "__main__":
    main()
