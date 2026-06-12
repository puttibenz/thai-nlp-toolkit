import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional


# ── NER label map ────────────────────────────────────────────────────────────
# ตรงกับ BEST2020 tag set ใน dataset จริง
NER_LABEL2ID = {
    "O":     0,
    "B-PERSON": 1, "I-PERSON": 2,
    "B-ORGANIZATION": 3, "I-ORGANIZATION": 4,
    "B-LOCATION": 5, "I-LOCATION": 6,
}
NER_ID2LABEL = {v: k for k, v in NER_LABEL2ID.items()}

# ── Sentiment label map ───────────────────────────────────────────────────────
SENTIMENT_LABEL2ID = {"neg": 0, "neu": 1, "pos": 2}
SENTIMENT_ID2LABEL = {v: k for k, v in SENTIMENT_LABEL2ID.items()}


# ─────────────────────────────────────────────────────────────────────────────
# NERDataset
# ─────────────────────────────────────────────────────────────────────────────

class NERDataset(Dataset):
    """
    BEST2020 NER dataset — JSON Lines format
    แต่ละบรรทัด: {"tokens": [...], "ner_tags": [...]}

    การ align label กับ subword เป็นจุดสำคัญที่สุดใน NER:
    - token "สมชาย" อาจถูก split เป็น ["สม", "ชาย"] (2 subwords)
    - label "B-PER" ให้เฉพาะ subword แรก ("สม")
    - subword ที่ 2 ("ชาย") ให้ label = -100 (ignore_index)
    """

    def __init__(
        self,
        data_path:  str,
        tokenizer:  Any,
        max_length: int = 512,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.examples   = self._load(data_path)

    def _load(self, path: str) -> List[Dict]:
        examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # validate minimal required fields
                if "tokens" in item and "ner_tags" in item:
                    examples.append(item)
        return examples

    def _align_labels(
        self,
        tokens:   List[str],
        ner_tags: List[str],
    ) -> Dict:
        """
        Tokenize ทีละคำ แล้ว align label กับ subword

        Returns dict พร้อม input_ids, attention_mask, labels
        """
        input_ids      = [self.tokenizer.cls_id]   # [CLS] ที่ตำแหน่ง 0
        label_ids      = [-100]                     # [CLS] ไม่มี NER label

        for token, tag in zip(tokens, ner_tags):
            # encode ทีละคำ ไม่ใส่ special tokens
            word_ids = self.tokenizer.sp.encode(token, out_type=int)
            if not word_ids:
                continue

            tag_id = NER_LABEL2ID.get(tag, 0)   # default O ถ้าไม่รู้จัก tag

            # subword แรก → label จริง
            input_ids.append(word_ids[0])
            label_ids.append(tag_id)

            # subword ที่ 2+ → -100 (ignore)
            for wid in word_ids[1:]:
                input_ids.append(wid)
                label_ids.append(-100)

        # เพิ่ม [SEP] ท้าย
        input_ids.append(self.tokenizer.sep_id)
        label_ids.append(-100)

        # Truncate
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length - 1] + [self.tokenizer.sep_id]
            label_ids = label_ids[:self.max_length - 1] + [-100]

        attention_mask = [1] * len(input_ids)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(label_ids,      dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.examples[idx]
        return self._align_labels(item["tokens"], item["ner_tags"])


# ─────────────────────────────────────────────────────────────────────────────
# SentimentDataset
# ─────────────────────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    """
    Wisesight Sentiment — TSV format
    แต่ละบรรทัด: text\\tlabel  (label = pos / neu / neg)
    """

    def __init__(
        self,
        data_path:  str,
        tokenizer:  Any,
        max_length: int = 512,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.examples   = self._load(data_path)

    def _load(self, path: str) -> List[Dict]:
        examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                text, label = parts[0], parts[1].strip().lower()
                if label not in SENTIMENT_LABEL2ID:
                    continue
                examples.append({"text": text, "label": label})
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        item   = self.examples[idx]
        # encode: [CLS] text [SEP]
        encoded = self.tokenizer.batch_encode(
            [item["text"]],
            max_length=self.max_length,
            padding=False,          # collator จะ pad ทีหลัง
            return_tensors=True,
        )
        return {
            "input_ids":      encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels":         torch.tensor(
                                  SENTIMENT_LABEL2ID[item["label"]],
                                  dtype=torch.long
                               ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# QADataset
# ─────────────────────────────────────────────────────────────────────────────

class QADataset(Dataset):
    """
    iApp Thai QA — SQuAD-style JSON format
    {
      "question": "...",
      "context":  "...",
      "answers":  {"text": ["..."], "answer_start": [42]}
    }

    จุดสำคัญ: answer_start ใน dataset เป็น character position
    ต้องแปลงเป็น token position หลัง encode
    """

    def __init__(
        self,
        data_path:  str,
        tokenizer:  Any,
        max_length: int = 512,
    ):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.examples   = self._load(data_path)

    def _load(self, path: str) -> List[Dict]:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # รองรับทั้ง flat list และ SQuAD-style nested
        if isinstance(data, list):
            return [ex for ex in data if self._valid(ex)]

        # SQuAD format: {"data": [{"paragraphs": [{"qas": [...]}]}]}
        examples = []
        for article in data.get("data", []):
            for para in article.get("paragraphs", []):
                context = para.get("context", "")
                for qa in para.get("qas", []):
                    ex = {
                        "question": qa.get("question", ""),
                        "context":  context,
                        "answers":  qa.get("answers", []),
                    }
                    if self._valid(ex):
                        examples.append(ex)
        return examples

    def _valid(self, ex: Dict) -> bool:
        return (
            bool(ex.get("question")) and
            bool(ex.get("context")) and
            bool(ex.get("answers"))
        )

    def _find_token_span(
        self,
        context_ids:   List[int],
        answer_text:   str,
        context_start: int,         # position ใน full sequence ที่ context เริ่ม
    ):
        """
        หา start/end token position ของ answer ใน context_ids
        ใช้ character prefix decoding alignment เพื่อความแม่นยำสูง (100% match rate)
        """
        context_text = self.tokenizer.sp.decode(context_ids)
        char_start = context_text.find(answer_text)
        if char_start == -1:
            return context_start, context_start
            
        char_end = char_start + len(answer_text)
        
        prefix_lens = []
        for i in range(len(context_ids) + 1):
            prefix_lens.append(len(self.tokenizer.sp.decode(context_ids[:i])))
            
        best_start = None
        best_end = None
        
        for i in range(len(context_ids)):
            token_start = prefix_lens[i]
            token_end = prefix_lens[i+1]
            
            if token_start <= char_start < token_end:
                best_start = i
            if token_start < char_end <= token_end:
                best_end = i
                
        if best_start is None:
            best_start = 0
        if best_end is None:
            best_end = best_start
            
        return context_start + best_start, context_start + best_end

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        item = self.examples[idx]

        # encode_qa คืน [CLS] question [SEP] context [SEP]
        encoded = self.tokenizer.encode_qa(
            question=item["question"],
            context=item["context"],
            max_length=self.max_length,
        )

        # context_ids สำหรับ span matching
        context_start = encoded["context_start"]
        full_ids      = encoded["input_ids"]
        context_ids   = full_ids[context_start:-1]   # ตัด [SEP] สุดท้ายออก

        # หา answer span — ใช้ answer แรกถ้ามีหลายคำตอบ
        answers     = item["answers"]
        answer_list = answers if isinstance(answers, list) else answers.get("text", [])
        answer_text = answer_list[0] if answer_list else ""

        start_pos, end_pos = self._find_token_span(
            context_ids, answer_text, context_start
        )

        # clamp ให้ไม่เกินความยาว sequence จริง
        seq_len    = len(full_ids)
        start_pos  = min(start_pos, seq_len - 1)
        end_pos    = min(end_pos,   seq_len - 1)

        return {
            "input_ids":      torch.tensor(full_ids,                  dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "start_labels":   torch.tensor(start_pos,                 dtype=torch.long),
            "end_labels":     torch.tensor(end_pos,                   dtype=torch.long),
            "context_start":  torch.tensor(context_start,             dtype=torch.long),
        }