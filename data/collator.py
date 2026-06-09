from unittest import result

import torch
from typing import List, Dict, Any

class MultiTaskDataCollator:
    """
    Dynamic padding collator สำหรับ multi-task batches

    ปัญหาหลักที่ต้องแก้:
    1. แต่ละ sample ใน batch ยาวไม่เท่ากัน → pad ให้เท่ากับ longest ใน batch
    2. แต่ละ task มี label key ต่างกัน → detect task จาก keys และ pad label ให้ถูกต้อง
    3. NER labels ต้อง pad ด้วย -100 (ignore_index) ไม่ใช่ 0
    4. QA labels เป็น scalar (position) ไม่ต้อง pad
    """

    def __init__(self, tokenizer: Any, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}
        
        # ── Step 1: หา max length ใน batch นี้ ──────────────────────────
        # pad เท่ากับ longest ใน batch เท่านั้น ไม่ใช่ max_length เสมอ
        # ประหยัด compute ได้มากเมื่อ batch มี sequence สั้น
        max_len = min(
            max(item['input_ids'].size(0) for item in batch),
            self.max_length
        )

        # ── Step 2: Detect task จาก keys ของ sample แรก ─────────────────
        keys         = set(batch[0].keys())
        is_ner       = "labels"       in keys and "start_labels" not in keys
        is_qa        = "start_labels" in keys 
        is_sent = "labels"       in keys and not is_ner and not is_qa
        # fallback: ถ้า labels เป็น 1D scalar = sentiment
        if is_ner and batch[0]['labels'].dim() == 0:
            is_ner = False
            is_sent = True

        # ── Step 3: Pad input_ids และ attention_mask ─────────────────────
        pad_id = self.tokenizer.pad_id
        
        input_ids_list = []
        attention_mask_list = []

        for item in batch:
            ids = item['input_ids']
            mask = item['attention_mask']
            seq_len = ids.size(0)

            if seq_len > max_len:
                ids = ids[:max_len]
                mask = mask[:max_len]
            elif seq_len < max_len:
                # pad ท้าย
                pad_len = max_len - seq_len
                ids  = torch.cat([ids,  torch.full((pad_len,), pad_id,  dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad_len,            dtype=torch.long)])

            input_ids_list.append(ids)
            attention_mask_list.append(mask)
        result = {
            'input_ids': torch.stack(input_ids_list),          # (B, T)
            'attention_mask': torch.stack(attention_mask_list) # (B, T)
        } 

        # ── Step 4: Pad labels ตาม task ───────────────────────────────

        if is_ner:
            # NER labels: (B, T) — pad ด้วย -100 (ignore_index)
            # ห้าม pad ด้วย 0 เพราะ 0 = label "O" จะ mislead loss
            label_list = []
            for item in batch:
                lbl     = item['labels']
                seq_len = lbl.size(0)

                if seq_len > max_len:
                    lbl = lbl[:max_len]
                elif seq_len < max_len:
                    pad_len = max_len - seq_len
                    lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
                label_list.append(lbl)
            
            result['ner_labels'] = torch.stack(label_list) # (B, T)
         
        elif is_sent:
            # Sentiment labels: scalar per sample → stack เป็น (B,)
            result['sentiment_labels'] = torch.stack(
                [item['labels'] for item in batch]
            )

        elif is_qa:
            # QA labels: scalar position → stack เป็น (B,)
            # clamp ให้ไม่เกิน max_len ที่ truncate แล้ว
            result["qa_start_labels"] = torch.stack([
                item["start_labels"].clamp(max=max_len - 1) for item in batch
            ])
            result["qa_end_labels"] = torch.stack([
                item["end_labels"].clamp(max=max_len - 1) for item in batch
            ])
            result["context_start"] = torch.stack([
                item["context_start"] for item in batch
            ])

        return result