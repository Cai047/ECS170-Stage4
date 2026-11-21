"""Training loop, reproducible split, checkpoint selection, and inference."""
import collections
import csv
import hashlib
import json
import os
import random
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from local_code.stage_4_code.Dataset_Loader import read_data, tokenize, encode
from local_code.stage_4_code.Method_RNN import SentimentRNN
from local_code.stage_4_code.Evaluate_Metrics import metrics

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate(model, x, y, batch_size):
    model.eval()
    probabilities, loss_sum = [], 0.0
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            logits = model(x[start:start+batch_size])
            loss_sum += nn.functional.cross_entropy(logits, y[start:start+batch_size], reduction='sum').item()
            probabilities.append(logits.softmax(-1)[:, 1].cpu().numpy())
    prob = np.concatenate(probabilities)
    return loss_sum / len(y), prob

def train(args, rows=None):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')
    if rows is None:
        rows = read_data(args.data)
    y_all = np.array([r[1] for r in rows['train']])
    train_idx, val_idx = train_test_split(np.arange(len(y_all)), test_size=0.2, random_state=args.seed, stratify=y_all)
    tokenized = {k: [tokenize(r[2]) for r in v] for k, v in rows.items()}
    counts = collections.Counter(w for i in train_idx for w in tokenized['train'][i])
    vocab = {'<PAD>': 0, '<UNK>': 1}
    vocab.update({w: i+2 for i, (w, _) in enumerate(counts.most_common(args.vocab_size-2))})
    (out / 'vocab.json').write_text(json.dumps(vocab), encoding='utf-8')
    train_array = encode(tokenized['train'], vocab, args.max_length)
    test_array = encode(tokenized['test'], vocab, args.max_length)
    x_train = torch.as_tensor(train_array[train_idx], device=device)
    y_train = torch.as_tensor(y_all[train_idx], device=device)
    x_val = torch.as_tensor(train_array[val_idx], device=device)
    y_val = torch.as_tensor(y_all[val_idx], device=device)
    x_test = torch.as_tensor(test_array, device=device)
    y_test_np = np.array([r[1] for r in rows['test']])
    y_test = torch.as_tensor(y_test_np, device=device)
    with (out / 'split_manifest.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f); writer.writerow(['file', 'label', 'partition'])
        for partition, indices in [('train', train_idx), ('validation', val_idx)]:
            writer.writerows((rows['train'][i][0], int(y_all[i]), partition) for i in indices)
        writer.writerows((r[0], r[1], 'test') for r in rows['test'])
    stats = {}
    for split, tokens in [('train', [tokenized['train'][i] for i in train_idx]),
                          ('validation', [tokenized['train'][i] for i in val_idx]), ('test', tokenized['test'])]:
        lengths = np.array([len(t) for t in tokens])
        total = sum(len(t) for t in tokens)
        stats[split] = dict(size=len(tokens), mean_length=float(lengths.mean()), median_length=float(np.median(lengths)),
                            truncated_fraction=float((lengths > args.max_length).mean()),
                            oov_fraction=sum(w not in vocab for t in tokens for w in t)/total)
    meta = dict(seed=args.seed, max_length=args.max_length, vocab_size=len(vocab), embedding_dim=128,
                batch_size=args.batch_size, learning_rate=0.001, dropout=0.3, max_epochs=args.epochs, patience=3,
                gradient_clip=1.0, device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                torch_version=torch.__version__, numpy_version=np.__version__, stats=stats,
                dataset_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest() if args.data.is_file() else None)
    (out / 'experiment.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta), flush=True)
    results = []
    configurations = [('RNN', 128), ('LSTM', 128), ('GRU', 128), ('GRU', 64)]
    # Architecture comparison and ablation are fixed before any test evaluation.
    for cell, hidden in configurations:
        name = f'{cell}_h{hidden}'
        seed_all(args.seed)
        config = dict(vocab_size=len(vocab), cell=cell, embedding_dim=128, hidden_dim=hidden, dropout=0.3)
        model = SentimentRNN(**config).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        best_loss, bad_epochs, history = float('inf'), 0, []
        start_time = time.perf_counter()
        for epoch in range(1, args.epochs+1):
            model.train()
            order = torch.randperm(len(y_train), device=device)
            total_loss, correct = 0., 0
            for start in range(0, len(order), args.batch_size):
                idx = order[start:start+args.batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_train[idx])
                loss = nn.functional.cross_entropy(logits, y_train[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(idx)
                correct += (logits.argmax(-1) == y_train[idx]).sum().item()
            val_loss, val_prob = evaluate(model, x_val, y_val, args.batch_size)
            row = dict(epoch=epoch, train_loss=total_loss/len(y_train), train_accuracy=correct/len(y_train),
                       val_loss=val_loss, val_accuracy=float(((val_prob >= .5) == y_all[val_idx]).mean()))
            history.append(row)
            print(name, json.dumps(row), flush=True)
            if val_loss < best_loss:
                best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
                torch.save(dict(state_dict=model.state_dict(), config=config, max_length=args.max_length,
                                seed=args.seed, best_epoch=epoch), out / f'{name}.pt')
            else:
                bad_epochs += 1
            (out / f'{name}_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            if bad_epochs >= 3:
                break
        duration = time.perf_counter()-start_time
        results.append(dict(model=name, cell=cell, hidden_dim=hidden, parameters=sum(p.numel() for p in model.parameters()),
                            best_epoch=best_epoch, epochs_run=len(history), best_val_loss=best_loss,
                            best_val_accuracy=history[best_epoch-1]['val_accuracy'], training_seconds=duration))
    selected = min(results, key=lambda r: r['best_val_loss'])['model']
    # Evaluate all pre-specified checkpoints on the official test set once, after training.
    for result in results:
        name = result['model']
        ckpt = torch.load(out / f'{name}.pt', map_location=device, weights_only=True)
        model = SentimentRNN(**ckpt['config']).to(device)
        model.load_state_dict(ckpt['state_dict'])
        test_loss, prob = evaluate(model, x_test, y_test, args.batch_size)
        result.update(metrics(y_test_np, prob)); result['test_loss'] = test_loss
        result['selected_by_validation'] = name == selected
        with (out / f'{name}_predictions.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow(['file', 'true_label', 'predicted_label', 'positive_probability'])
            writer.writerows((r[0], r[1], int(p >= .5), float(p)) for r,p in zip(rows['test'], prob))
        print('TEST', json.dumps(result), flush=True)
    print('COMPLETE', selected, flush=True)
    return results

def predict(args):
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    vocab = json.loads((args.checkpoint.parent / 'vocab.json').read_text(encoding='utf-8'))
    model = SentimentRNN(**ckpt['config'])
    model.load_state_dict(ckpt['state_dict']); model.eval()
    x = torch.from_numpy(encode([tokenize(args.text)], vocab, ckpt['max_length']))
    with torch.no_grad():
        prob = model(x).softmax(-1)[0, 1].item()
    print(json.dumps(dict(label='positive' if prob >= .5 else 'negative', positive_probability=prob)))
