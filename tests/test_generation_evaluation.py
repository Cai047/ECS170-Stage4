"""Small hand-calculated examples of the comparisons used in the report."""
import unittest

from local_code.stage_4_code.generation.evaluation import GenerationEvaluator, summarize_continuations


def sample(continuation, seed=1):
    tokens = continuation.split()
    return {'prompt': 'what did the', 'prompt_tokens': ['what', 'did', 'the'],
            'generated_tokens': tokens, 'text': 'what did the ' + continuation,
            'stop': 'eos', 'seed': seed}


class GenerationEvaluationTests(unittest.TestCase):
    def test_comparison_excludes_prompt_and_stops_at_record_boundaries(self):
        docs = ['read what did the answer say'.split(), 'what did the a b c a b'.split(), ['c', 'd']]
        row = GenerationEvaluator(docs).evaluate(sample('a b c a b c d'))
        self.assertEqual(row['training_prompt_example']['training_document_index'], 1)
        self.assertEqual(row['longest_training_match']['length'], 5)
        self.assertEqual(row['training_fourgram_hits'], 2)
        self.assertEqual(row['fourgrams'], 4)
        self.assertEqual(row['repeated_trigrams'], 1)
        self.assertEqual(row['trigrams'], 5)
        self.assertFalse(row['exact_training_joke'])

    def test_pooled_rates_and_short_continuations(self):
        evaluator = GenerationEvaluator(['what did the four five six seven'.split()])
        rows = [evaluator.evaluate(sample('four five six seven')), evaluator.evaluate(sample('', 2))]
        self.assertIsNone(rows[1]['training_fourgram_fraction'])
        self.assertEqual(rows[1]['longest_training_match']['length'], 0)
        summary = summarize_continuations(rows)
        self.assertEqual(summary['training_fourgram_fraction'], 1.0)
        self.assertEqual(summary['exact_training_jokes'], 1)
        self.assertEqual(summary['eos_samples'], 2)


if __name__ == '__main__':
    unittest.main()
