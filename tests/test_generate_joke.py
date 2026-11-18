"""Exercise the checkpoint-only manual testing entry point and recurrent decoding."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from torch import nn

from local_code.stage_4_code.generation.data import BOS, EOS, PAD, UNK
from local_code.stage_4_code.generation.inference import generate_joke
from local_code.stage_4_code.generation.model import GenerationModel


VOCAB = [PAD, UNK, BOS, EOS, "red", "fox", "jumps", "."]


class CountingModel(nn.Module):
    """Predict a different token at each recurrent step, with a preferred UNK."""
    def forward(self, ids, state=None):
        position = (0 if state is None else state) + ids.shape[1]
        next_id = {4: 4, 5: 5, 6: 6}.get(position, 3)
        logits = torch.full((1, ids.shape[1], len(VOCAB)), -40.0)
        logits[:, -1, 1] = 50
        logits[:, -1, next_id] = 30
        return logits, position


class ManualGenerationTests(unittest.TestCase):
    def test_state_unknown_prompt_and_special_tokens_across_strategies(self):
        for strategy in ("beam", "greedy", "sample"):
            with self.subTest(strategy=strategy):
                sample = generate_joke(CountingModel(), VOCAB, "Xylophone fox jumps",
                                       strategy=strategy, top_k=1)
                self.assertEqual(sample["unknown_tokens"], ["xylophone"])
                self.assertEqual(sample["generated_tokens"], ["red", "fox", "jumps"])
                self.assertEqual(sample["text"], "Xylophone fox jumps red fox jumps")
                self.assertEqual(sample["stop"], "eos")

    def test_input_word_count(self):
        for prompt in ("only two", "one two three four", "   "):
            with self.subTest(prompt=prompt), self.assertRaisesRegex(ValueError, "three starting words"):
                generate_joke(CountingModel(), VOCAB, prompt)

    def test_cli_only_needs_checkpoint(self):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "results" / "gru"
            folder.mkdir(parents=True)
            model = GenerationModel("gru", len(VOCAB), 8, 8)
            config = {"cell": "gru", "embedding": 8, "hidden": 8}
            torch.save({"config": config, "vocab": VOCAB, "state_dict": model.state_dict()},
                       folder / "checkpoint.pt")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(project) + os.pathsep + env.get("PYTHONPATH", "")
            run = subprocess.run([sys.executable, "-m", "script.stage_4_script.generate_joke",
                                  "--prompt", "Why red fox", "--model", "gru", "--results-dir",
                                  "results", "--max-tokens", "8", "--json"], cwd=tmp, env=env,
                                 text=True, capture_output=True, check=True)
            self.assertIn("encoded as <unk>: why", run.stderr)
            result = json.loads(run.stdout)
            self.assertTrue(result["text"].startswith("Why red fox"))
            self.assertLessEqual(len(result["generated_tokens"]), 8)


if __name__ == "__main__":
    unittest.main()
