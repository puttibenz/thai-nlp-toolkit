import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Any

class MultiTaskLoss(nn.Module):
    """
    Weighted sum of losses สำหรับ 3 tasks พร้อมกัน

    NER       — token-level CrossEntropy  (ignore padding label = -100)
    Sentiment — sentence-level CrossEntropy
    QA        — mean(start_loss, end_loss) โดย start/end ต่างก็เป็น CrossEntropy
    """
    def __init__(self, task_weights: Dict[str, float]):
        super().__init__()
        
        # task_weights มาจาก config เช่น {"ner": 1.0, "sentiment": 0.8, "qa": 1.2}
        self.task_weights = task_weights

        # ignore_index=-100 คือ convention มาตรฐาน:
        # label ที่เป็น -100 จะถูกข้ามออกจากการคำนวณ loss
        # ใช้ mark padding tokens และ tokens ที่ไม่ใช่ first subword ใน NER
        self.ner_loss_fn       = nn.CrossEntropyLoss(ignore_index=-100)
        self.sentiment_loss_fn = nn.CrossEntropyLoss()
        # QA ใช้ ignore_index=-100 เพื่อ handle กรณี answer ไม่อยู่ใน context
        self.qa_loss_fn        = nn.CrossEntropyLoss(ignore_index=-100)

    def ner_loss(
        self,
        logits: Tensor,   # (B, T, num_labels)
        labels: Tensor,   # (B, T)  — -100 สำหรับ padding และ non-first subwords
    ) -> Tensor:
        B, T, C = logits.shape
        # CrossEntropyLoss ต้องการ (N, C) และ (N,)
        # reshape: (B*T, C) และ (B*T,)
        return self.ner_loss_fn(
            logits.view(B * T, C),
            labels.view(B * T),
        )
    
    def sentiment_loss(
        self,
        logits: Tensor,   # (B, num_classes)
        labels: Tensor,   # (B,)
    ) -> Tensor:
        return self.sentiment_loss_fn(logits, labels)

    def qa_loss(
        self,
        start_logits: Tensor,   # (B, T)
        end_logits:   Tensor,   # (B, T)
        start_labels: Tensor,   # (B,)  — position ของ answer start
        end_labels:   Tensor,   # (B,)  — position ของ answer end
    ) -> Tensor:
        # QA loss = mean ของ start loss + end loss
        # ทั้งสองอย่างเป็น classification ว่า token ไหนเป็น start/end
        start_loss = self.qa_loss_fn(start_logits, start_labels)
        end_loss   = self.qa_loss_fn(end_logits,   end_labels)
        return (start_loss + end_loss) / 2

    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets:     Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Parameters
        ----------
        predictions : dict — output จาก model แต่ละ task
            ner       : Tensor (B, T, num_labels)
            sentiment : Tensor (B, num_classes)
            qa_start  : Tensor (B, T)
            qa_end    : Tensor (B, T)

        targets : dict — ground truth labels
            ner_labels       : Tensor (B, T)
            sentiment_labels : Tensor (B,)
            qa_start_labels  : Tensor (B,)
            qa_end_labels    : Tensor (B,)

        Returns
        -------
        dict with keys:
            total   : weighted combined loss (ใช้สำหรับ backward)
            ner     : raw NER loss (สำหรับ logging)
            sentiment : raw sentiment loss
            qa      : raw QA loss
        """
        losses = {}

        # ── NER loss ────────────────────────────────────────────────────
        if "ner" in predictions and "ner_labels" in targets:
            losses["ner"] = self.ner_loss(
                predictions["ner"],
                targets["ner_labels"],
            )

        # ── Sentiment loss ───────────────────────────────────────────────
        if "sentiment" in predictions and "sentiment_labels" in targets:
            losses["sentiment"] = self.sentiment_loss(
                predictions["sentiment"],
                targets["sentiment_labels"],
            )

        # ── QA loss ─────────────────────────────────────────────────────
        if "qa_start" in predictions and "qa_start_labels" in targets:
            losses["qa"] = self.qa_loss(
                predictions["qa_start"],
                predictions["qa_end"],
                targets["qa_start_labels"],
                targets["qa_end_labels"],
            )

        # ── Weighted sum ─────────────────────────────────────────────────
        # เฉพาะ tasks ที่มีใน batch นี้เท่านั้น
        # เพราะ multi-task training บางครั้ง batch มาจาก task เดียว
        total = sum(
            self.task_weights.get(task, 1.0) * loss
            for task, loss in losses.items()
        )

        losses["total"] = total
        return losses
