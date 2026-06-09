import unittest
import torch
from unittest.mock import MagicMock
from data.collator import MultiTaskDataCollator

class TestCollator(unittest.TestCase):
    def test_collator(self):
        tok = MagicMock()
        tok.pad_id = 0
        collator = MultiTaskDataCollator(tok, max_length=512)

        # ── NER batch ──────────────────────────────────────────────
        ner_batch = [
            {"input_ids": torch.tensor([4, 10, 20, 5]),
             "attention_mask": torch.tensor([1, 1, 1, 1]),
             "labels": torch.tensor([-100, 1, -100, -100])},         # len=4
            {"input_ids": torch.tensor([4, 30, 5]),
             "attention_mask": torch.tensor([1, 1, 1]),
             "labels": torch.tensor([-100, 0, -100])},              # len=3
        ]
        out = collator(ner_batch)
        self.assertEqual(out["input_ids"].shape, (2, 4))   # pad ถึง longest=4
        self.assertEqual(out["ner_labels"].shape, (2, 4))
        self.assertEqual(out["ner_labels"][1, 3].item(), -100)  # padded position = -100 ไม่ใช่ 0
        self.assertEqual(out["attention_mask"][1, 3].item(), 0) # padded attention = 0
        print("NER collator OK")

        # ── Sentiment batch ─────────────────────────────────────────
        sent_batch = [
            {"input_ids": torch.tensor([4, 10, 20, 30, 5]),
             "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
             "labels": torch.tensor(2)},                          # pos
            {"input_ids": torch.tensor([4, 10, 5]),
             "attention_mask": torch.tensor([1, 1, 1]),
             "labels": torch.tensor(0)},                          # neg
        ]
        out = collator(sent_batch)
        self.assertEqual(out["input_ids"].shape, (2, 5))
        self.assertEqual(out["sentiment_labels"].shape, (2,))
        self.assertEqual(out["sentiment_labels"][0].item(), 2)
        self.assertEqual(out["input_ids"][1, 3].item(), 0)   # padded = pad_id
        print("Sentiment collator OK")

        # ── QA batch ────────────────────────────────────────────────
        qa_batch = [
            {"input_ids": torch.tensor([4, 10, 5, 20, 30, 5]),
             "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1]),
             "start_labels": torch.tensor(3),
             "end_labels":   torch.tensor(4),
             "context_start": torch.tensor(3)},
            {"input_ids": torch.tensor([4, 10, 20, 5, 30, 40, 50, 5]),
             "attention_mask": torch.tensor([1, 1, 1, 1, 1, 1, 1, 1]),
             "start_labels": torch.tensor(4),
             "end_labels":   torch.tensor(6),
             "context_start": torch.tensor(4)},
        ]
        out = collator(qa_batch)
        self.assertEqual(out["input_ids"].shape, (2, 8))   # pad ถึง longest=8
        self.assertEqual(out["qa_start_labels"].shape, (2,))
        self.assertEqual(out["qa_end_labels"].shape, (2,))
        self.assertEqual(out["context_start"].shape, (2,))
        print("QA collator OK")

        # ── Dynamic padding: batch สั้น ไม่ควร pad ถึง 512 ───────────
        short_batch = [
            {"input_ids": torch.tensor([4, 10, 5]),
             "attention_mask": torch.tensor([1, 1, 1]),
             "labels": torch.tensor(1)},
        ]
        out = collator(short_batch)
        self.assertEqual(out["input_ids"].shape[1], 3)   # ไม่ pad เกินความจำเป็น
        print("Dynamic padding OK")

if __name__ == "__main__":
    unittest.main()
