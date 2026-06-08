import unittest
import torch
from training.losses import MultiTaskLoss

class TestMultiTaskLoss(unittest.TestCase):
    def setUp(self):
        self.B = 2
        self.T = 16
        self.task_weights = {"ner": 1.0, "sentiment": 0.8, "qa": 1.2}
        self.loss_fn = MultiTaskLoss(self.task_weights)

    def test_full_batch_loss(self):
        # inputs requiring gradients to test backprop
        preds = {
            "ner":       torch.randn(self.B, self.T, 7, requires_grad=True),
            "sentiment": torch.randn(self.B, 3, requires_grad=True),
            "qa_start":  torch.randn(self.B, self.T, requires_grad=True),
            "qa_end":    torch.randn(self.B, self.T, requires_grad=True),
        }

        # NER labels: -100 for padding positions
        ner_labels = torch.randint(0, 7, (self.B, self.T))
        ner_labels[:, 12:] = -100   # positions 12-15 are padding

        targets = {
            "ner_labels":       ner_labels,
            "sentiment_labels": torch.randint(0, 3, (self.B,)),
            "qa_start_labels":  torch.randint(0, self.T, (self.B,)),
            "qa_end_labels":    torch.randint(0, self.T, (self.B,)),
        }

        # Calculate loss
        losses = self.loss_fn(preds, targets)

        # Assertions
        self.assertIn("total", losses)
        self.assertIn("ner", losses)
        self.assertIn("sentiment", losses)
        self.assertIn("qa", losses)
        self.assertTrue(losses["total"] > 0)
        self.assertFalse(torch.isnan(losses["total"]))

        # Check gradient flow
        losses["total"].backward()
        for task, pred in preds.items():
            self.assertIsNotNone(pred.grad, f"Gradient should flow back to predictions of {task}")
            self.assertFalse(torch.isnan(pred.grad).any(), f"Gradients of {task} should not contain NaNs")

    def test_partial_batch_loss(self):
        # Test partial batch (NER only)
        preds_ner_only = {"ner": torch.randn(self.B, self.T, 7, requires_grad=True)}
        targets_ner_only = {"ner_labels": torch.randint(0, 7, (self.B, self.T))}
        
        losses_partial = self.loss_fn(preds_ner_only, targets_ner_only)
        
        self.assertIn("ner", losses_partial)
        self.assertIn("total", losses_partial)
        self.assertNotIn("sentiment", losses_partial)
        self.assertNotIn("qa", losses_partial)
        self.assertTrue(losses_partial["total"] > 0)
        self.assertFalse(torch.isnan(losses_partial["total"]))

        # Check backprop on partial batch
        losses_partial["total"].backward()
        self.assertIsNotNone(preds_ner_only["ner"].grad)

if __name__ == "__main__":
    unittest.main()
