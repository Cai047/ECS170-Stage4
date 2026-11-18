"""Evaluate stored Stage 4 generation experiments against the original training text."""
import argparse
import json
from pathlib import Path

from local_code.stage_4_code.generation.data import GenerationDataset
from local_code.stage_4_code.generation.evaluation import evaluate_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='Course generation text file or directory')
    parser.add_argument('--test-data', type=Path, help='Supply only when the original experiment used a separate test file')
    parser.add_argument('--results-dir', type=Path, default=Path('results/generation'))
    args = parser.parse_args()
    recorded = json.loads((args.results_dir / 'dataset.json').read_text())
    corpus = GenerationDataset(args.data, args.test_data, max_vocab=recorded['max_vocab'], min_freq=recorded['min_freq']).load()
    if corpus.metadata != recorded:
        raise ValueError('Generation corpus or split differs from the original experiment')
    rows = evaluate_runs(corpus, args.results_dir)
    for row in rows:
        print(f"{row['model']:4s}  test PPL {row['test_ppl']:.2f}  longest training match {row['max_training_match_tokens']} tokens")


if __name__ == '__main__':
    main()
