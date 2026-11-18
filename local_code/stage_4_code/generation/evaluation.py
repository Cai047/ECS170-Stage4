"""Compare sampled continuations to the training text, excluding their prompts."""
import csv
from difflib import SequenceMatcher
import json
import math
from pathlib import Path

from local_code.base_class.evaluate import evaluate
from .model import render_text


CELLS = ('rnn', 'lstm', 'gru')
COLORS = ('#19547B', '#B45521', '#29794B')


def ngrams(tokens, length):
    return [tuple(tokens[i:i + length]) for i in range(len(tokens) - length + 1)]


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


class GenerationEvaluator(evaluate):
    """Exact lexical overlap is a comparison with the corpus, not a fluency score."""
    def __init__(self, training_documents):
        super().__init__('Generation evaluation', 'Held-out learning curves and training-text comparison')
        self.training_documents = training_documents

    def evaluate(self, sample):
        continuation = sample['generated_tokens']
        prompt = tuple(sample['prompt_tokens'])
        if len(prompt) != 3:
            raise ValueError('Each sample must start with three tokens')
        fours = ngrams(continuation, 4)
        wanted = set(fours)
        seen = set()
        reference = None
        exact_training_joke = False
        longest = {'length': 0, 'tokens': []}
        matcher = SequenceMatcher(b=continuation, autojunk=False)
        for doc_index, doc in enumerate(self.training_documents):
            # All searches stop at the training document or split boundary.
            if tuple(doc[:3]) == prompt:
                if doc[3:] == continuation:
                    exact_training_joke = True
                if reference is None:
                    reference = {'training_document_index': doc_index, 'text': render_text(doc)}
            if len(seen) < len(wanted):
                seen.update(wanted.intersection(ngrams(doc, 4)))
            matcher.set_seq1(doc)
            match = matcher.find_longest_match()
            if match.size > longest['length']:
                longest = {'length': match.size, 'training_document_index': doc_index,
                           'training_token_offset': match.a, 'generated_token_offset': match.b,
                           'tokens': continuation[match.b:match.b + match.size]}
        trigrams = ngrams(continuation, 3)
        repeats = len(trigrams) - len(set(trigrams))
        hits = sum(g in seen for g in fours)
        return {'prompt': sample['prompt'], 'seed': sample['seed'], 'text': sample['text'],
                'stop': sample['stop'], 'continuation_tokens': len(continuation),
                'training_prompt_example': reference, 'exact_training_joke': exact_training_joke,
                'longest_training_match': longest, 'fourgrams': len(fours), 'training_fourgram_hits': hits,
                'training_fourgram_fraction': fraction(hits, len(fours)), 'trigrams': len(trigrams),
                'repeated_trigrams': repeats, 'repeated_trigram_fraction': fraction(repeats, len(trigrams))}


def summarize_continuations(rows):
    def total(key):
        return sum(row[key] for row in rows)

    return {'num_samples': len(rows), 'eos_samples': sum(row['stop'] == 'eos' for row in rows),
            'exact_training_jokes': total('exact_training_joke'),
            'continuation_tokens': total('continuation_tokens'),
            'training_fourgram_fraction': fraction(total('training_fourgram_hits'), total('fourgrams')),
            'repeated_trigram_fraction': fraction(total('repeated_trigrams'), total('trigrams')),
            'max_training_match_tokens': max(row['longest_training_match']['length'] for row in rows)}


def plot_curves(histories, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with plt.rc_context({'font.size': 9, 'axes.spines.right': False, 'axes.spines.top': False}):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), sharex=True, sharey=True, layout='constrained')
        for ax, field, title in zip(axes, ('train_nll', 'val_nll'), ('Training', 'Validation')):
            for cell, color in zip(CELLS, COLORS):
                history = histories[cell]
                ax.plot([h['epoch'] for h in history], [h[field] for h in history],
                        color=color, linewidth=1.8, label=cell.upper())
            ax.set(title=title, xlabel='Epoch')
            ax.grid(axis='y', alpha=0.22)
        axes[0].set_ylabel('Mean next-token NLL (nats)')
        axes[1].legend(frameon=False)
        fig.savefig(path, dpi=240)
        plt.close(fig)


def evaluate_runs(corpus, results_dir):
    results = Path(results_dir)
    evaluator = GenerationEvaluator(corpus.train_tokens)
    comparisons, summaries, prompt_summaries, histories = {}, [], [], {}
    shared_conditions = None
    for cell in CELLS:
        run = results / cell
        samples = json.loads((run / 'samples.json').read_text())
        if not samples:
            raise ValueError(f'No generated samples for {cell}')
        conditions = [(tuple(s['prompt_tokens']), s['seed']) for s in samples]
        if len(set(conditions)) != len(conditions):
            raise ValueError(f'Duplicate prompt and sampling seed in {cell}')
        if shared_conditions is not None and conditions != shared_conditions:
            raise ValueError('Use the same prompts and sampling seeds for every model')
        shared_conditions = conditions
        with (run / 'history.csv').open() as stream:
            history = [{'epoch': int(r['epoch']), 'train_nll': float(r['train_nll']),
                        'val_nll': float(r['val_nll'])} for r in csv.DictReader(stream)]
        if [h['epoch'] for h in history] != list(range(1, len(history) + 1)) or not history:
            raise ValueError(f'Missing epoch record in {cell}')
        if not all(math.isfinite(h[s]) for h in history for s in ('train_nll', 'val_nll')):
            raise ValueError(f'Nonfinite NLL in {cell}')
        histories[cell] = history
        metrics = json.loads((run / 'metrics.json').read_text())
        details = [evaluator.evaluate(sample) for sample in samples]
        comparisons[cell] = details
        summaries.append({'model': cell.upper(), 'test_nll': metrics['test']['nll'],
                          'test_ppl': metrics['test']['ppl'], 'test_accuracy': metrics['test']['accuracy'],
                          **summarize_continuations(details)})
        for prompt in dict.fromkeys(s['prompt'] for s in details):
            rows = [s for s in details if s['prompt'] == prompt]
            prompt_summaries.append({'model': cell.upper(), 'prompt': prompt, **summarize_continuations(rows)})
    plot_curves(histories, results / 'learning_curves.png')
    for filename, rows in [('summary.csv', summaries), ('summary_by_prompt.csv', prompt_summaries)]:
        with (results / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
    definitions = {'training_fourgram_fraction': 'Fraction of generated four-token positions also seen contiguously in at least one training document.',
                   'repeated_trigram_fraction': 'Fraction of generated three-token positions duplicating an earlier trigram in the same continuation.',
                   'max_training_match_tokens': 'Longest exact contiguous overlap with one training document, measured over all continuations.',
                   'tokenization': 'Same lowercase tokens as the corpus loader, including punctuation. Fourgrams, trigrams, and longest match exclude the three prompt words. Document indices and offsets are zero-based.',
                   'exact_training_jokes': 'Count of complete normalized token sequences (including prompt) identical to a training joke.',
                   'eos_samples': 'Count of samples that emitted end-of-sequence before the token limit.',
                   'aggregation': 'Rates pool position counts across samples; undefined rates are null. training_prompt_example is the first training joke with exactly the same three opening tokens, when one exists.'}
    (results / 'evaluation.json').write_text(json.dumps({'definitions': definitions, 'samples': comparisons}, indent=2, ensure_ascii=False) + '\n')
    return summaries
