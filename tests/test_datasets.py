import unittest
import json
import tempfile
import os
import torch
from unittest.mock import MagicMock
from data.datasets import NERDataset, SentimentDataset, QADataset

class TestDatasets(unittest.TestCase):
    def test_datasets(self):
        # Mock tokenizer
        tok = MagicMock()
        tok.cls_id = 4
        tok.sep_id = 5
        tok.sp.encode = lambda t, out_type: [10, 20] if len(t) > 2 else [10]
        tok.sp.decode = lambda ids: (
            "" if not ids else
            "สมชาย" if ids == [20] else
            "สมชายเดินไปตลาด"
        )
        tok.batch_encode = lambda texts, **kw: {
            "input_ids":      torch.tensor([[4, 10, 20, 5]]),
            "attention_mask": torch.tensor([[1,  1,  1, 1]]),
        }
        tok.encode_qa = lambda question, context, max_length: {
            "input_ids":      [4, 10, 5, 20, 30, 5],
            "attention_mask": [1,  1, 1,  1,  1, 1],
            "context_start":  3,
        }

        # ── NER ──────────────────────────────────────────────────────────
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"tokens": ["สมชาย", "เดิน"], "ner_tags": ["B-PER", "O"]}) + "\n")
            f.write(json.dumps({"tokens": ["กรุงเทพ"], "ner_tags": ["B-LOC"]}) + "\n")
            ner_path = f.name

        try:
            ner_ds = NERDataset(ner_path, tok)
            self.assertEqual(len(ner_ds), 2)
            item = ner_ds[0]
            self.assertIn("input_ids", item)
            self.assertIn("labels", item)
            self.assertEqual(item["input_ids"][0].item(), 4)           # CLS
            self.assertEqual(item["labels"][0].item(), -100)        # CLS ไม่มี label
            self.assertEqual(item["labels"].shape, item["input_ids"].shape)
            print("NER OK")
        finally:
            if os.path.exists(ner_path):
                os.unlink(ner_path)

        # ── Sentiment ────────────────────────────────────────────────────
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".tsv", delete=False) as f:
            f.write("สินค้าดีมาก\tpos\n")
            f.write("แย่มาก\tneg\n")
            f.write("bad_line_no_tab\n")          # ต้องถูก skip
            sent_path = f.name

        try:
            sent_ds = SentimentDataset(sent_path, tok)
            self.assertEqual(len(sent_ds), 2)                  # bad_line ถูก skip
            item = sent_ds[0]
            self.assertEqual(item["labels"].item(), 2)        # pos = 2
            item = sent_ds[1]
            self.assertEqual(item["labels"].item(), 0)        # neg = 0
            print("Sentiment OK")
        finally:
            if os.path.exists(sent_path):
                os.unlink(sent_path)

        # ── QA ──────────────────────────────────────────────────────────
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as f:
            json.dump([{
                "question": "ใครเดิน",
                "context":  "สมชายเดินไปตลาด",
                "answers":  ["สมชาย"],
            }], f, ensure_ascii=False)
            qa_path = f.name

        try:
            qa_ds = QADataset(qa_path, tok)
            self.assertEqual(len(qa_ds), 1)
            item = qa_ds[0]
            self.assertIn("start_labels", item)
            self.assertIn("end_labels", item)
            self.assertIn("context_start", item)
            self.assertGreaterEqual(item["start_labels"].item(), item["context_start"].item())   # span ต้องอยู่ใน context
            print("QA OK")
        finally:
            if os.path.exists(qa_path):
                os.unlink(qa_path)

if __name__ == "__main__":
    unittest.main()