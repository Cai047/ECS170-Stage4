"""Train and evaluate each recurrent cell under the same next-token objective."""
import csv
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from local_code.base_class.result import result
from local_code.base_class.setting import setting
from .data import GenerationDataset, language_windows, tokenize
from .model import GenerationModel, sample_story


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def pass_over(model, batches, device, optimizer=None, clip=1.0):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss_sum = F.cross_entropy(logits.flatten(0, 1), y.flatten(),
                                       ignore_index=0, reduction="sum")
            valid = y.ne(0)
            n = int(valid.sum().item())
            if training:
                (loss_sum / n).backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            total_loss += loss_sum.detach().item()
            count += n
            correct += ((logits.detach().argmax(-1) == y) & valid).sum().item()
    nll = total_loss / count
    return {"nll": nll, "ppl": math.exp(nll), "accuracy": correct / count, "targets": count}


class GenerationResult(result):
    def __init__(self, destination):
        super().__init__("Stage 4 generation results", "metrics, samples and model weights")
        self.destination = Path(destination)

    def save(self):
        scores, history, samples, checkpoint = self.data
        torch.save(checkpoint, self.destination / "checkpoint.pt")
        with (self.destination / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        write_json(self.destination / "samples.json", samples)
        write_json(self.destination / "metrics.json", scores)

    def load(self):
        return json.loads((self.destination / "metrics.json").read_text(encoding="utf-8"))


class GenerationSetting(setting):
    def __init__(self, args):
        super().__init__("Stage 4 generation", "Fixed-epoch RNN/LSTM/GRU comparison")
        self.args = args

    def load_run_save_evaluate(self):
        args = self.args
        folder = Path(args.output)
        folder.mkdir(parents=True, exist_ok=True)
        if any((folder / cell).exists() for cell in args.models):
            raise FileExistsError("A requested model output already exists; use a fresh output directory")
        corpus = GenerationDataset(args.data, args.test_data, args.max_vocab, args.min_freq).load()
        starts = {tuple(doc[:3]) for doc in corpus.train_tokens}
        for prompt in args.prompts:
            words = tokenize(prompt)
            if (len(words) != 3 or tuple(words) not in starts or
                    any(t not in corpus.vocab or not any(c.isalnum() for c in t) for t in words)):
                raise ValueError(f"Use three in-vocabulary words that start a training joke: {prompt!r}")
        windows = {split: language_windows(getattr(corpus, split + "_tokens"), corpus.vocab, args.context)
                   for split in ("train", "val", "test")}
        manifest = folder / "dataset.json"
        if manifest.exists() and json.loads(manifest.read_text()) != corpus.metadata:
            raise ValueError("The output directory belongs to a different dataset setup")
        write_json(manifest, corpus.metadata)
        print(json.dumps(corpus.metadata["splits"]), flush=True)
        torch.set_num_threads(args.threads)
        device = torch.device(args.device)
        for cell in args.models:
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            model = GenerationModel(cell, len(corpus.vocab), args.embedding, args.hidden).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            destination = folder / cell
            destination.mkdir()
            val_loader = DataLoader(windows["val"], batch_size=args.batch_size)
            history = []
            start = time.perf_counter()
            for epoch in range(1, args.epochs + 1):
                generator = torch.Generator().manual_seed(args.seed + epoch)
                train_loader = DataLoader(windows["train"], batch_size=args.batch_size,
                                          shuffle=True, generator=generator)
                train = pass_over(model, train_loader, device, optimizer, args.clip)
                val = pass_over(model, val_loader, device)
                row = {"epoch": epoch, "train_nll": train["nll"], "val_nll": val["nll"],
                       "train_ppl": train["ppl"], "val_ppl": val["ppl"],
                       "train_accuracy": train["accuracy"], "val_accuracy": val["accuracy"]}
                history.append(row)
                print(f"{cell:4} epoch {epoch:2}   train NLL {train['nll']:.4f}   "
                      f"val NLL {val['nll']:.4f}", flush=True)
            elapsed = time.perf_counter() - start
            test_loader = DataLoader(windows["test"], batch_size=args.batch_size)
            test = pass_over(model, test_loader, device)
            samples = [sample_story(model, corpus.vocab, prompt, s, args.sample_tokens,
                                    args.top_k, args.temperature) for prompt in args.prompts for s in args.sample_seeds]
            config = {name: value for name, value in vars(args).items()
                      if name not in {"data", "test_data", "output", "models"}}
            config["cell"] = cell
            scores = {"config": config, "vocab_size": len(corpus.vocab),
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "training_seconds": elapsed, "train": train, "val": val, "test": test,
                      "torch_version": str(torch.__version__), "python": sys.version.split()[0]}
            checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "config": config, "vocab": corpus.vocab}
            writer = GenerationResult(destination)
            writer.data = scores, history, samples, checkpoint
            writer.save()
            print(f"{cell}: test NLL {test['nll']:.4f}  perplexity {test['ppl']:.2f}", flush=True)
