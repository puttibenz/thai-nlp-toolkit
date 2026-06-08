import os
import json
from typing import List, Dict, Optional
import sentencepiece as spm
import torch
from .preprocess import preprocess_thai   # จาก Phase 1


# Special token IDs — ตรงกับที่ตั้งค่าตอน SentencePiece training
PAD_ID  = 0
UNK_ID  = 1
BOS_ID  = 2
EOS_ID  = 3
CLS_ID  = 4   # user_defined_symbols ลำดับที่ 1
SEP_ID  = 5   # user_defined_symbols ลำดับที่ 2
MASK_ID = 6   # user_defined_symbols ลำดับที่ 3


class ThaiTokenizer:
    """
    Wraps SentencePiece BPE สำหรับภาษาไทย
    - preprocess text ก่อน encode ทุกครั้ง
    - รองรับ special tokens: [PAD], [UNK], [BOS], [EOS], [CLS], [SEP], [MASK]
    - batch_encode คืน torch.Tensor พร้อมใช้ใน DataLoader
    """

    def __init__(self, model_path: str):
        """
        Parameters
        ----------
        model_path : str
            path ถึง .model file จาก SentencePiece training
            เช่น "tokenizer/thai_bpe.model"
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ไม่พบ model file: {model_path}\n"
                f"รัน train_tokenizer.py ก่อนเพื่อสร้าง model"
            )

        self.model_path = model_path
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        # validate vocab size
        self.vocab_size = self.sp.get_piece_size()

        # special token ids (ดึงจาก model จริง ไม่ hardcode)
        self.pad_id  = self.sp.piece_to_id("[PAD]")
        self.unk_id  = self.sp.piece_to_id("[UNK]")
        self.bos_id  = self.sp.piece_to_id("[BOS]")
        self.eos_id  = self.sp.piece_to_id("[EOS]")
        self.cls_id  = self.sp.piece_to_id("[CLS]")
        self.sep_id  = self.sp.piece_to_id("[SEP]")
        self.mask_id = self.sp.piece_to_id("[MASK]")

    # ──────────────────────────────────────────────
    # Core encode / decode
    # ──────────────────────────────────────────────

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        text = preprocess_thai(text)
        if not text:
            return [self.cls_id, self.sep_id] if add_special_tokens else []

        ids = self.sp.encode(text, out_type=int)

        if add_special_tokens:
            ids = [self.cls_id] + ids + [self.sep_id]  # ← เปลี่ยนตรงนี้

        return ids

    def encode_qa(
    self,
    question: str,
    context: str,
    max_length: int = 512,
    ) -> dict:
        """
        QA ต้องการ: [CLS] question [SEP] context [SEP]
        และต้องรู้ว่า context เริ่มที่ position ไหน
        """
        q_ids = self.sp.encode(preprocess_thai(question), out_type=int)
        c_ids = self.sp.encode(preprocess_thai(context),  out_type=int)

        # [CLS] q_ids [SEP] c_ids [SEP]
        ids = [self.cls_id] + q_ids + [self.sep_id] + c_ids + [self.sep_id]

        # context เริ่มที่ position นี้ (QAHead ใช้ mask คำนวณ span)
        context_start = 1 + len(q_ids) + 1   # หลัง [CLS] + question + [SEP]

        # truncate ถ้ายาวเกิน
        if len(ids) > max_length:
            # ตัด context ก่อน ไม่ตัด question
            allowed_ctx = max_length - context_start - 1  # -1 สำหรับ [SEP] สุดท้าย
            c_ids = c_ids[:allowed_ctx]
            ids = [self.cls_id] + q_ids + [self.sep_id] + c_ids + [self.sep_id]

        attention_mask = [1] * len(ids)

        return {
            "input_ids":      ids,
            "attention_mask": attention_mask,
            "context_start":  context_start,   # QAHead ใช้ป้องกัน predict span ใน question
        }

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        แปลง list of token IDs → text

        Parameters
        ----------
        skip_special_tokens : bool
            ถ้า True จะกรอง PAD, BOS, EOS, CLS, SEP, MASK ออกก่อน decode
        """
        special = {
            self.pad_id, self.bos_id, self.eos_id,
            self.cls_id, self.sep_id, self.mask_id,
        }

        if skip_special_tokens:
            ids = [i for i in ids if i not in special]

        return self.sp.decode(ids)

    # ──────────────────────────────────────────────
    # Batch encode — ใช้ใน DataLoader
    # ──────────────────────────────────────────────

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 512,
        padding: bool = True,
        add_special_tokens: bool = True,
        return_tensors: bool = True,
    ) -> Dict:
        """
        Encode หลาย texts พร้อมกัน พร้อม padding และ truncation

        Returns
        -------
        dict with keys:
            input_ids      : (B, T) — token IDs
            attention_mask : (B, T) — 1=real token, 0=padding
        """
        all_ids = []
        for text in texts:
            ids = self.encode(text, add_special_tokens=add_special_tokens)

            # Truncate: ถ้ายาวเกิน max_length ให้ตัดท้าย
            # แต่รักษา EOS ไว้เสมอ
            if len(ids) > max_length:
                if add_special_tokens:
                    ids = ids[:max_length - 1] + [self.eos_id]
                else:
                    ids = ids[:max_length]

            all_ids.append(ids)

        # Padding: pad ให้ยาวเท่ากับ sequence ที่ยาวที่สุดใน batch
        if padding:
            max_len = max(len(ids) for ids in all_ids)
            padded_ids = []
            attention_masks = []

            for ids in all_ids:
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [self.pad_id] * pad_len)
                # attention_mask: 1 สำหรับ real token, 0 สำหรับ padding
                attention_masks.append([1] * len(ids) + [0] * pad_len)
        else:
            padded_ids = all_ids
            attention_masks = [[1] * len(ids) for ids in all_ids]

        if return_tensors:
            return {
                "input_ids":      torch.tensor(padded_ids,      dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }

        return {
            "input_ids":      padded_ids,
            "attention_mask": attention_masks,
        }

    # ──────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────

    def save(self, output_dir: str) -> None:
        """
        บันทึก tokenizer ลง directory
        สร้าง 2 ไฟล์:
          - thai_bpe.model  (SentencePiece binary)
          - tokenizer_config.json  (metadata)
        """
        os.makedirs(output_dir, exist_ok=True)

        # copy model file
        import shutil
        dst = os.path.join(output_dir, "thai_bpe.model")
        if os.path.abspath(self.model_path) != os.path.abspath(dst):
            shutil.copy2(self.model_path, dst)

        # บันทึก metadata
        config = {
            "vocab_size": self.vocab_size,
            "model_file": "thai_bpe.model",
            "special_tokens": {
                "pad_id":  self.pad_id,
                "unk_id":  self.unk_id,
                "bos_id":  self.bos_id,
                "eos_id":  self.eos_id,
                "cls_id":  self.cls_id,
                "sep_id":  self.sep_id,
                "mask_id": self.mask_id,
            }
        }
        with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"tokenizer saved → {output_dir}")

    @classmethod
    def from_pretrained(cls, directory: str) -> "ThaiTokenizer":
        """
        โหลด tokenizer จาก directory ที่ save() ไว้
        
        Usage:
            tok = ThaiTokenizer.from_pretrained("outputs/tokenizer")
        """
        config_path = os.path.join(directory, "tokenizer_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"ไม่พบ tokenizer_config.json ใน {directory}")

        with open(config_path) as f:
            config = json.load(f)

        model_path = os.path.join(directory, config["model_file"])
        return cls(model_path=model_path)

    # ──────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────

    def tokenize(self, text: str) -> List[str]:
        """แสดง subword pieces เป็น string — ใช้สำหรับ debug"""
        text = preprocess_thai(text)
        return self.sp.encode(text, out_type=str)

    def vocab_size_actual(self) -> int:
        return self.sp.get_piece_size()

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"ThaiTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"model='{os.path.basename(self.model_path)}')"
        )