"""Word-level corpus preparation for the Stage 4 generation experiments."""
from collections import Counter
from dataclasses import dataclass
import csv
import hashlib
import io
import random
from pathlib import Path
import re
import unicodedata

import torch
from torch.utils.data import TensorDataset

from local_code.base_class.dataset import dataset


PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIAL = (PAD, UNK, BOS, EOS)
TOKEN = re.compile(r"\w+(?:'\w+)*|[^\w\s]", re.UNICODE)


def tokenize(text):
    normalized = unicodedata.normalize("NFKC", text).replace("’", "'").lower()
    return TOKEN.findall(normalized)


def read_documents(source):
    """Read the course ID,Joke CSV without joining jokes or treating IDs as text."""
    path = Path(source)
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline=""))
    if reader.fieldnames != ["ID", "Joke"]:
        raise ValueError("Expected the course generation CSV with header ID,Joke")
    docs, ids = [], set()
    for row in reader:
        if not row["ID"] or row["ID"] in ids or None in row or not row["Joke"]:
            raise ValueError(f"Invalid or duplicate record ID near CSV line {reader.line_num}")
        ids.add(row["ID"])
        tokens = tokenize(row["Joke"])
        if not tokens:
            raise ValueError(f"Empty joke at ID {row['ID']}")
        docs.append(tokens)
    source = {"file": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
              "tokens": sum(map(len, docs)), "records": len(docs)}
    return docs, [source]


def group_key(doc):
    # Ignore case and punctuation differences when assigning jokes to splits.
    return tuple(t for t in doc if any(c.isalnum() for c in t))


def split_jokes(documents, with_test=True):
    groups = sorted({group_key(doc) for doc in documents})
    if len(groups) < 10:
        raise ValueError("Expected at least 10 unique jokes for group splitting")
    random.Random(170).shuffle(groups)
    first = int((0.8 if with_test else 0.9) * len(groups))
    second = int(0.9 * len(groups)) if with_test else len(groups)
    assignment = {group: i for i, block in enumerate([groups[:first], groups[first:second],
                                                       groups[second:]]) for group in block}
    parts = [[], [], []]
    for doc in documents:
        parts[assignment[group_key(doc)]].append(doc)
    return tuple(parts)


@dataclass
class Corpus:
    train_tokens: list
    val_tokens: list
    test_tokens: list
    vocab: list
    metadata: dict


class GenerationDataset(dataset):
    """Fit the vocabulary on training tokens only, then encode all splits using it."""
    def __init__(self, data_path, test_path=None, max_vocab=3000, min_freq=2):
        super().__init__("Stage 4 generation", "Next-token prediction on the course joke corpus")
        self.path = Path(data_path)
        self.test_path = Path(test_path) if test_path is not None else None
        self.max_vocab = max_vocab
        self.min_freq = min_freq

    def load(self):
        docs, sources = read_documents(self.path)
        size = sum(map(len, docs))
        if size < 30 or self.max_vocab < 8 or self.min_freq < 1:
            raise ValueError("Need at least 30 tokens, max_vocab >= 8 and min_freq >= 1")
        train, val, test = split_jokes(docs, with_test=self.test_path is None)
        test_sources = []
        split = "80/10/10 by unique normalized joke groups, fixed shuffle seed 170"
        if self.test_path is not None:
            test, test_sources = read_documents(self.test_path)
            if {group_key(x) for x in test}.intersection(group_key(x) for x in train + val):
                raise ValueError("Supplied test file contains training/validation jokes")
            split = "90/10 group shuffle of supplied training jokes; separate supplied test jokes"
        counts = Counter(t for doc in train for t in doc)
        ordered = sorted((t for t, n in counts.items() if n >= self.min_freq),
                         key=lambda t: (-counts[t], t))
        vocab = list(SPECIAL) + ordered[:self.max_vocab - len(SPECIAL)]
        known = set(vocab)
        meta = {"sources": sources, "test_sources": test_sources, "split": split,
                "split_seed": 170, "unique_groups": len({group_key(doc) for doc in train + val + test}),
                "vocab_size": len(vocab), "max_vocab": self.max_vocab, "min_freq": self.min_freq,
                "special_tokens": list(SPECIAL), "splits": {}}
        for name, part in (("train", train), ("val", val), ("test", test)):
            n = sum(map(len, part))
            unknown = sum(t not in known for doc in part for t in doc)
            meta["splits"][name] = {"tokens": n, "documents_or_fragments": len(part),
                                    "unique_groups": len({group_key(doc) for doc in part}),
                                    "oov_tokens": unknown, "oov_fraction": unknown / n}
        self.data = Corpus(train, val, test, vocab, meta)
        return self.data


def language_windows(documents, vocab, length):
    """Each within-joke transition is scored once; padding labels are ignored."""
    stoi = {t: i for i, t in enumerate(vocab)}
    batches_x, batches_y = [], []
    for doc in documents:
        ids = [stoi[BOS], *(stoi.get(t, stoi[UNK]) for t in doc), stoi[EOS]]
        for start in range(0, len(ids) - 1, length):
            x = ids[start:start + length]
            y = ids[start + 1:start + 1 + length]
            # At a joke boundary, x can include the unpredicted final EOS.
            x = x[:len(y)]
            missing = length - len(y)
            batches_x.append(x + [stoi[PAD]] * missing)
            batches_y.append(y + [stoi[PAD]] * missing)
    return TensorDataset(torch.tensor(batches_x), torch.tensor(batches_y))
