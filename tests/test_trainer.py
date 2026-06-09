import unittest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from unittest.mock import MagicMock, patch
import tempfile
import os
import shutil

from training.trainer import MultiTaskTrainer

def make_dummy_loader(task: str, n=8):
    """สร้าง DataLoader จำลองสำหรับแต่ละ task"""
    B, T = 4, 16
    base = {
        "input_ids":      torch.randint(1, 100, (B, T)),
        "attention_mask": torch.ones(B, T, dtype=torch.long),
    }
    if task == "ner":
        ner_labels = torch.randint(-1, 7, (B, T))
        ner_labels[ner_labels == -1] = -100
        base["ner_labels"] = ner_labels
    elif task == "sentiment":
        base["sentiment_labels"] = torch.randint(0, 3, (B,))
    elif task == "qa":
        base["qa_start_labels"] = torch.randint(0, T, (B,))
        base["qa_end_labels"]   = torch.randint(0, T, (B,))
        base["context_start"]   = torch.full((B,), 4)

    # wrap เป็น list of dicts ให้ DataLoader ใช้ได้
    dataset = [
        {k: v[i] for k, v in base.items()}
        for i in range(B)
    ]
    return DataLoader(dataset, batch_size=2,
                      collate_fn=lambda x: {
                          k: torch.stack([d[k] for d in x])
                          for k in x[0]
                      })

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config = {
            "training": {
                "learning_rate": 1e-4, "weight_decay": 0.01,
                "warmup_steps": 2,     "max_steps": 10,
                "grad_accum_steps": 2, "max_grad_norm": 1.0,
                "mixed_precision": False,
                "task_weights": {"ner": 1.0, "sentiment": 0.8, "qa": 1.2},
            },
            "evaluation": {"eval_every": 999, "save_every": 999},
            "paths": {"output_dir": self.tmp_dir},
        }

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_train_epoch_runs(self):
        model = MagicMock()
        
        # Create a dummy parameter that requires grad
        dummy_param = torch.randn(10, requires_grad=True)
        model.named_parameters.return_value = [
            ("some_layer.weight", dummy_param)
        ]
        model.parameters.return_value = iter([dummy_param])

        # Link mock outputs to dummy_param so backward pass works
        # Use side_effect to return new tensors with new computation graphs on every call
        scale_fn = lambda: dummy_param.sum() * 0.0 + 1.0
        
        # mock encoder + heads
        B, T, D = 2, 16, 64
        model.encoder.side_effect = lambda *args, **kwargs: (torch.randn(B, T, D) * scale_fn(), None)
        model.ner_head.side_effect = lambda *args, **kwargs: torch.randn(B, T, 7) * scale_fn()
        model.sentiment_head.side_effect = lambda *args, **kwargs: torch.randn(B, 3) * scale_fn()
        model.qa_head.side_effect = lambda *args, **kwargs: (torch.randn(B, T) * scale_fn(), torch.randn(B, T) * scale_fn())

        tokenizer = MagicMock()

        trainer = MultiTaskTrainer(
            model, tokenizer, self.config,
            train_loaders={"ner": make_dummy_loader("ner"),
                           "sentiment": make_dummy_loader("sentiment")},
            val_loaders={},
        )
        losses = trainer.train_epoch()
        self.assertIn("total", losses)
        self.assertGreater(losses["total"], 0)
        print("test_train_epoch_runs OK")

    def test_checkpoint_roundtrip(self):
        model = MagicMock()
        dummy_param = torch.randn(10, requires_grad=True)
        model.named_parameters.return_value = [
            ("some_layer.weight", dummy_param)
        ]
        model.parameters.return_value = iter([dummy_param])
        model.state_dict.return_value = {"dummy_weight": torch.tensor([1.23])}
        model.load_state_dict = MagicMock()

        tokenizer = MagicMock()

        trainer = MultiTaskTrainer(
            model, tokenizer, self.config,
            train_loaders={"ner": make_dummy_loader("ner")},
            val_loaders={},
        )

        trainer.global_step = 42
        trainer.best_metric = 0.5

        ckpt_path = os.path.join(self.tmp_dir, "test_ckpt")
        trainer.save_checkpoint(ckpt_path)

        # Create new trainer instance and load checkpoint
        trainer2 = MultiTaskTrainer(
            model, tokenizer, self.config,
            train_loaders={"ner": make_dummy_loader("ner")},
            val_loaders={},
        )
        
        trainer2.load_checkpoint(ckpt_path)
        self.assertEqual(trainer2.global_step, 42)
        self.assertEqual(trainer2.best_metric, 0.5)
        model.load_state_dict.assert_called()
        print("test_checkpoint_roundtrip OK")

if __name__ == "__main__":
    unittest.main()
