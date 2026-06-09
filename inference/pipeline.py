# inference/pipeline.py
import os
import json
import torch
import yaml
from typing import List, Dict, Any, Optional

from tokenizer.thai_tokenizer import ThaiTokenizer
from model.encoder import ThaiTransformerEncoder, ModelConfig
from model.heads.ner_head import NERHead
from model.heads.sentiment_head import SentimentHead
from model.heads.qa_head import QAHead


# label maps สำหรับ decode output กลับเป็น string
NER_ID2LABEL = {
    0: "O",
    1: "B-PER", 2: "I-PER",
    3: "B-ORG",  4: "I-ORG",
    5: "B-LOC",  6: "I-LOC",
}
SENTIMENT_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


class ThaiNLPModel(torch.nn.Module):
    """รวม encoder + 3 heads เป็น module เดียว สำหรับ load/save"""
    def __init__(self, config: ModelConfig, num_ner_labels: int = 7):
        super().__init__()
        self.encoder        = ThaiTransformerEncoder(config)
        self.ner_head       = NERHead(config.d_model, num_ner_labels)
        self.sentiment_head = SentimentHead(config.d_model, num_classes=3)
        self.qa_head        = QAHead(config.d_model)


class ThaiNLPPipeline:
    """
    High-level inference class
    โหลด model ครั้งเดียวแล้วเรียก predict() ได้เรื่อยๆ
    """

    def __init__(self, model_dir: str, device: str = "auto"):
        # ── Device ───────────────────────────────────────────────────────
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ── Load config ───────────────────────────────────────────────────
        config_path = os.path.join(model_dir, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"ไม่พบ config.yaml ใน {model_dir}")
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)
        model_cfg = ModelConfig(**raw_config["model"])

        # ── Load tokenizer ────────────────────────────────────────────────
        self.tokenizer = ThaiTokenizer.from_pretrained(
            os.path.join(model_dir, "tokenizer")
        )

        # ── Load model ────────────────────────────────────────────────────
        self.model = ThaiNLPModel(model_cfg)
        ckpt_path  = os.path.join(model_dir, "checkpoint_best", "checkpoint.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ไม่พบ checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        print(f"pipeline ready on {self.device}")

    # ─────────────────────────────────────────────────────────────────────
    # predict — entry point หลัก
    # ─────────────────────────────────────────────────────────────────────

    def predict(
        self,
        text:     str,
        tasks:    List[str],
        question: Optional[str] = None,   # ต้องการสำหรับ QA
        context:  Optional[str] = None,   # ต้องการสำหรับ QA
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        text     : input text สำหรับ NER และ Sentiment
        tasks    : list ของ task ที่ต้องการ ["ner", "sentiment", "qa"]
        question : question string (เฉพาะ QA)
        context  : context string (เฉพาะ QA)

        Returns
        -------
        dict ที่มี key ตาม tasks ที่ขอ
        """
        results = {}

        with torch.no_grad():
            # ── NER ──────────────────────────────────────────────────────
            if "ner" in tasks:
                results["ner"] = self._predict_ner(text)

            # ── Sentiment ────────────────────────────────────────────────
            if "sentiment" in tasks:
                results["sentiment"] = self._predict_sentiment(text)

            # ── QA ───────────────────────────────────────────────────────
            if "qa" in tasks:
                if question is None or context is None:
                    results["qa"] = {"error": "QA ต้องการ question และ context"}
                else:
                    results["qa"] = self._predict_qa(question, context)

        return results

    # ─────────────────────────────────────────────────────────────────────
    # Task-specific predict methods
    # ─────────────────────────────────────────────────────────────────────

    def _encode(self, input_ids, attention_mask):
        """Shared encoder forward"""
        ids  = torch.tensor([input_ids],      dtype=torch.long).to(self.device)
        mask = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        hidden, _ = self.model.encoder(ids, mask)
        return hidden, mask

    def _predict_ner(self, text: str) -> List[Dict[str, str]]:
        """
        คืน list ของ {"token": str, "label": str}
        กรอง [CLS], [SEP], padding ออก และ merge subwords กลับเป็นคำ
        """
        encoded    = self.tokenizer.batch_encode(
            [text], max_length=512, padding=False, return_tensors=False
        )
        input_ids  = encoded["input_ids"][0]
        attn_mask  = encoded["attention_mask"][0]

        hidden, _  = self._encode(input_ids, attn_mask)
        logits     = self.model.ner_head(hidden)          # (1, T, num_labels)
        pred_ids   = logits[0].argmax(dim=-1).tolist()    # (T,)

        # decode tokens กลับเป็น string แล้ว zip กับ label
        pieces = self.tokenizer.sp.id_to_piece(input_ids)
        special = {
            self.tokenizer.cls_id,
            self.tokenizer.sep_id,
            self.tokenizer.pad_id,
        }

        entities = []
        current_word  = ""
        current_label = "O"

        for token_id, label_id in zip(input_ids, pred_ids):
            if token_id in special:
                continue

            piece = self.tokenizer.sp.id_to_piece([token_id])[0]
            label = NER_ID2LABEL.get(label_id, "O")

            # SentencePiece ใช้ "▁" นำหน้า subword แรกของคำ
            if piece.startswith("▁") or not current_word:
                # บันทึกคำก่อนหน้า (ถ้ามี)
                if current_word:
                    entities.append({
                        "token": current_word,
                        "label": current_label,
                    })
                current_word  = piece.lstrip("▁")
                current_label = label
            else:
                # subword ต่อเนื่อง — merge เข้ากับคำปัจจุบัน
                current_word += piece

        # บันทึกคำสุดท้าย
        if current_word:
            entities.append({"token": current_word, "label": current_label})

        return entities

    def _predict_sentiment(self, text: str) -> Dict[str, Any]:
        """
        คืน {"label": str, "confidence": float, "scores": dict}
        """
        encoded   = self.tokenizer.batch_encode(
            [text], max_length=512, padding=False, return_tensors=False
        )
        hidden, mask = self._encode(
            encoded["input_ids"][0],
            encoded["attention_mask"][0],
        )
        mask_tensor = torch.tensor(
            [encoded["attention_mask"][0]], dtype=torch.long
        ).to(self.device)

        logits = self.model.sentiment_head(hidden, mask_tensor)  # (1, 3)
        probs  = logits.softmax(dim=-1)[0].tolist()

        pred_id     = int(logits.argmax(dim=-1).item())
        pred_label  = SENTIMENT_ID2LABEL[pred_id]
        confidence  = round(probs[pred_id], 4)

        return {
            "label":      pred_label,
            "confidence": confidence,
            "scores": {
                SENTIMENT_ID2LABEL[i]: round(p, 4)
                for i, p in enumerate(probs)
            },
        }

    def _predict_qa(self, question: str, context: str) -> Dict[str, Any]:
        """
        คืน {"answer": str, "start": int, "end": int, "confidence": float}
        """
        encoded       = self.tokenizer.encode_qa(question, context, max_length=512)
        input_ids     = encoded["input_ids"]
        attn_mask     = encoded["attention_mask"]
        context_start = encoded["context_start"]

        hidden, _ = self._encode(input_ids, attn_mask)

        start_logits, end_logits = self.model.qa_head(
            hidden, context_start=context_start
        )   # (1, T) each

        # หา (start, end) ที่ให้ score สูงสุดโดย start ≤ end
        start_logits = start_logits[0]   # (T,)
        end_logits   = end_logits[0]     # (T,)
        seq_len      = len(input_ids)

        best_score = float("-inf")
        best_start = context_start
        best_end   = context_start

        # จำกัด span ไม่เกิน 30 tokens (คำตอบยาวเกินนี้ไม่สมเหตุสมผล)
        MAX_ANSWER_LEN = 30

        for s in range(context_start, seq_len):
            for e in range(s, min(s + MAX_ANSWER_LEN, seq_len)):
                score = start_logits[s].item() + end_logits[e].item()
                if score > best_score:
                    best_score = score
                    best_start = s
                    best_end   = e

        # decode answer กลับเป็น string
        answer_ids = input_ids[best_start:best_end + 1]
        answer     = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

        # confidence = softmax score ของ best span (normalize คร่าวๆ)
        start_probs = start_logits.softmax(dim=-1)
        end_probs   = end_logits.softmax(dim=-1)
        confidence  = round(
            (start_probs[best_start] * end_probs[best_end]).item(), 4
        )

        return {
            "answer":     answer,
            "start":      best_start,
            "end":        best_end,
            "confidence": confidence,
        }