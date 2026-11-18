"""Regression tests for CSV record boundaries, grouped splits and next-token targets."""
import csv
from pathlib import Path
import tempfile
import unittest

from local_code.stage_4_code.generation.data import (
    BOS, EOS, PAD, UNK, GenerationDataset, group_key, language_windows, read_documents, tokenize,
)


class CourseJokeDataTests(unittest.TestCase):
    def test_group_split_and_train_only_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data"
            # Includes a quoted comma, newline and apostrophe as a single CSV field.
            rows = [(i + 1, f'What did the "captain,\nfriend" call marker{i}?') for i in range(30)]
            rows.extend([(31, "Do you know my cat?"), (32, "DO YOU KNOW MY CAT!!!")])
            with path.open("w", newline="", encoding="utf-8") as stream:
                w = csv.writer(stream)
                w.writerow(["ID", "Joke"])
                w.writerows(rows)
            docs, _ = read_documents(path)
            self.assertEqual(len(docs), 32)
            self.assertEqual(docs[0], tokenize(rows[0][1]))
            corpus = GenerationDataset(path, min_freq=1).load()
            splits = corpus.train_tokens, corpus.val_tokens, corpus.test_tokens
            keys = [set(map(group_key, split)) for split in splits]
            self.assertTrue(keys[0].isdisjoint(keys[1]) and keys[1].isdisjoint(keys[2])
                            and keys[0].isdisjoint(keys[2]))
            self.assertEqual(sum(map(len, splits)), len(rows))
            self.assertEqual(len(set.union(*keys)), 31)
            self.assertEqual(sum(sum(group_key(doc) == group_key(docs[-1]) for doc in split) == 2
                                 for split in splits), 1)
            train_tokens = set(t for doc in corpus.train_tokens for t in doc)
            held_out = set(t for doc in corpus.val_tokens + corpus.test_tokens for t in doc) - train_tokens
            self.assertTrue(held_out)
            self.assertTrue(held_out.isdisjoint(corpus.vocab))

    def test_windows_never_place_the_next_joke_in_a_target(self):
        voc = [PAD, UNK, BOS, EOS, "a", "b", "c", "d"]
        # A six-token joke tests continuation windows and the rightmost target;
        # a one-token joke tests BOS/EOS, padding and no cross-joke transitions.
        windows = language_windows([["a", "b", "c", "d", "a", "b"], ["x"]], voc, 3)
        pairs = [(i, j) for x, y in windows for i, j in zip(x.tolist(), y.tolist()) if j != 0]
        self.assertEqual(pairs, [(2, 4), (4, 5), (5, 6), (6, 7), (7, 4), (4, 5), (5, 3),
                                 (2, 1), (1, 3)])


if __name__ == "__main__":
    unittest.main()
