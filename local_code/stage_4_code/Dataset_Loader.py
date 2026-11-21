"""Load IMDb reviews and apply training-only vocabulary preprocessing."""
import html
import re
import zipfile
from pathlib import Path
import numpy as np
from local_code.base_class.dataset import dataset

def tokenize(text):
    text = re.sub(r'<[^>]+>', ' ', html.unescape(text)).lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", text)

def encode(tokens, vocab, max_length):
    out = np.zeros((len(tokens), max_length), dtype=np.int64)
    for i, row in enumerate(tokens):
        ids = [vocab.get(w, 1) for w in row[:max_length]] or [1]
        out[i, :len(ids)] = ids
    return out

def read_data(path):
    rows = {'train': [], 'test': []}
    if path.is_file():
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                match = re.fullmatch(r'(?:.*/)?text_classification/(train|test)/(neg|pos)/([^/]+\.txt)', name)
                if match and not name.startswith('__MACOSX/'):
                    split, label, filename = match.groups()
                    text = z.read(name).decode('utf-8')
                    rows[split].append((f'{split}/{label}/{filename}', int(label == 'pos'), text))
    else:
        for split in rows:
            for label in ['neg', 'pos']:
                for p in sorted((path / split / label).glob('*.txt')):
                    rows[split].append((f'{split}/{label}/{p.name}', int(label == 'pos'), p.read_text(encoding='utf-8')))
    if not all(rows.values()):
        raise ValueError('Expected stage_4_data.zip or text_classification directory with train/test neg/pos files.')
    return rows

class Dataset_Loader(dataset):
    def load(self):
        path = Path(self.dataset_source_folder_path)
        if self.dataset_source_file_name:
            path = path / self.dataset_source_file_name
        self.data = read_data(path)
        return self.data
