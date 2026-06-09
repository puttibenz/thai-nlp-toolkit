import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# NER metrics — entity-level F1 (ไม่ใช่ token-level)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_entities(seq: List[str]) -> set:
    """
    แปลง BIO tag sequence เป็น set ของ (entity_type, start, end)

    ทำไมต้องใช้ entity-level F1 ไม่ใช่ token-level:
    - token-level จะให้ credit บางส่วนแม้ boundary ผิด
    - entity-level ถือว่าถูกต้องก็ต่อเมื่อ entity ทั้งตัวถูกต้อง
    - เป็น standard ของ CoNLL evaluation script
    """
    entities = set()
    current_type  = None
    current_start = None

    for i, tag in enumerate(seq):
        if tag.startswith("B-"):
            # เริ่ม entity ใหม่ — ปิด entity เก่าก่อน (ถ้ามี)
            if current_type is not None:
                entities.add((current_type, current_start, i - 1))
            current_type  = tag[2:]
            current_start = i

        elif tag.startswith("I-"):
            entity_type = tag[2:]
            # I- ที่ไม่ตามหลัง B- หรือ type ไม่ตรง = boundary error
            if current_type != entity_type:
                if current_type is not None:
                    entities.add((current_type, current_start, i - 1))
                current_type  = None
                current_start = None

        else:  # "O" หรือ tag อื่น
            if current_type is not None:
                entities.add((current_type, current_start, i - 1))
            current_type  = None
            current_start = None

    # ปิด entity สุดท้าย (ถ้า sequence จบโดยไม่มี O)
    if current_type is not None:
        entities.add((current_type, current_start, len(seq) - 1))

    return entities


def compute_ner_metrics(
    predictions: List[List[str]],   # [["B-PER","I-PER","O",...], ...]
    references:  List[List[str]],   # ground truth ในรูปแบบเดียวกัน
) -> Dict[str, float]:
    """
    Entity-level Precision, Recall, F1 โดยรวม และแยกต่อ entity type
    standard เดียวกับ CoNLL-2003 evaluation
    """
    assert len(predictions) == len(references), \
        f"จำนวน sequences ไม่ตรงกัน: {len(predictions)} vs {len(references)}"

    # tp, fp, fn รวมทุก entity type
    tp_total = fp_total = fn_total = 0

    # แยกต่อ entity type ด้วย
    per_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp":0,"fp":0,"fn":0})

    for pred_seq, ref_seq in zip(predictions, references):
        pred_entities = _extract_entities(pred_seq)
        ref_entities  = _extract_entities(ref_seq)

        tp_entities = pred_entities & ref_entities
        fp_entities = pred_entities - ref_entities
        fn_entities = ref_entities  - pred_entities

        tp_total += len(tp_entities)
        fp_total += len(fp_entities)
        fn_total += len(fn_entities)

        for (etype, *_) in tp_entities: per_type[etype]["tp"] += 1
        for (etype, *_) in fp_entities: per_type[etype]["fp"] += 1
        for (etype, *_) in fn_entities: per_type[etype]["fn"] += 1

    def _prf(tp, fp, fn) -> Tuple[float, float, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        return precision, recall, f1

    p, r, f1 = _prf(tp_total, fp_total, fn_total)

    result = {
        "ner_precision": round(p,  4),
        "ner_recall":    round(r,  4),
        "ner_f1":        round(f1, 4),
    }

    # เพิ่ม per-type metrics สำหรับ debug
    for etype, counts in per_type.items():
        ep, er, ef1 = _prf(counts["tp"], counts["fp"], counts["fn"])
        result[f"ner_f1_{etype}"] = round(ef1, 4)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment metrics — Accuracy + Macro F1
# ─────────────────────────────────────────────────────────────────────────────

def compute_sentiment_metrics(
    predictions: List[int],   # [2, 0, 1, 2, ...]
    references:  List[int],   # ground truth labels
) -> Dict[str, float]:
    """
    Accuracy และ Macro F1

    ทำไมต้องมีทั้งสอง:
    - Accuracy บอกว่าถูกกี่ %  แต่ถ้า class imbalance จะ mislead
      (เช่น 80% เป็น pos → predict pos ทุกอย่างก็ได้ accuracy 80%)
    - Macro F1 average F1 ทุก class เท่ากัน ไม่ว่าจะมีกี่ตัวอย่าง
      ทำให้ class เล็ก (เช่น neutral) มีน้ำหนักเท่า class ใหญ่
    """
    assert len(predictions) == len(references), \
        f"จำนวน samples ไม่ตรงกัน: {len(predictions)} vs {len(references)}"

    n = len(predictions)
    if n == 0:
        return {"sentiment_accuracy": 0.0, "sentiment_macro_f1": 0.0}

    # Accuracy
    correct  = sum(p == r for p, r in zip(predictions, references))
    accuracy = correct / n

    # Per-class TP/FP/FN
    classes  = sorted(set(references))
    f1_scores = []

    for cls in classes:
        tp = sum(p == cls and r == cls for p, r in zip(predictions, references))
        fp = sum(p == cls and r != cls for p, r in zip(predictions, references))
        fn = sum(p != cls and r == cls for p, r in zip(predictions, references))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "sentiment_accuracy":  round(accuracy,  4),
        "sentiment_macro_f1":  round(macro_f1,  4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# QA metrics — Exact Match + Token-level F1
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_answer(text: str) -> str:
    """
    Normalize คำตอบก่อนเปรียบเทียบ — standard จาก SQuAD evaluation script
    ลบ article, punctuation, extra whitespace, lowercase
    """
    # lowercase
    text = text.lower()
    # ลบ punctuation
    text = re.sub(r"[^\u0E00-\u0E7Fa-z0-9\s]", "", text)
    # normalize whitespace
    text = " ".join(text.split())
    return text


def _token_f1(pred: str, ref: str) -> float:
    """
    Token-level F1 ระหว่าง pred กับ ref
    นับ token ที่ overlap กัน (ไม่สนลำดับ)

    เป็น metric มาตรฐานของ SQuAD — ให้ partial credit
    เช่น ref="สมชาย ทำงาน" pred="สมชาย" → F1 = 0.67
    """
    pred_tokens = _normalize_answer(pred).split()
    ref_tokens  = _normalize_answer(ref).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    n_common = sum(common.values())

    if n_common == 0:
        return 0.0

    precision = n_common / len(pred_tokens)
    recall    = n_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_qa_metrics(
    predictions: List[Dict[str, Any]],   # [{"prediction_text": "สมชาย"}, ...]
    references:  List[Dict[str, Any]],   # [{"answers": {"text": ["สมชาย", ...]}}, ...]
) -> Dict[str, float]:
    """
    Exact Match (EM) และ Token-level F1 แบบ SQuAD

    EM = 1 ถ้า normalize(pred) == normalize(ref) ทุกตัวอักษร
    F1 = token overlap ระหว่าง pred กับ ref

    ถ้า reference มีหลายคำตอบ (multiple valid answers)
    จะเลือก max score จากทุกคำตอบ ตาม SQuAD convention
    """
    assert len(predictions) == len(references), \
        f"จำนวน samples ไม่ตรงกัน: {len(predictions)} vs {len(references)}"

    if not predictions:
        return {"qa_exact_match": 0.0, "qa_f1": 0.0}

    total_em = 0.0
    total_f1 = 0.0

    for pred_dict, ref_dict in zip(predictions, references):
        pred_text = pred_dict.get("prediction_text", "")

        # reference อาจมีหลายคำตอบที่ถูกต้อง
        ref_answers = ref_dict.get("answers", {})
        if isinstance(ref_answers, list):
            ref_texts = ref_answers
        else:
            ref_texts = ref_answers.get("text", [])

        if not ref_texts:
            continue

        # Exact Match: ใช้คำตอบที่ดีที่สุดจากทุก reference
        em = max(
            float(_normalize_answer(pred_text) == _normalize_answer(ref))
            for ref in ref_texts
        )

        # Token F1: ใช้คำตอบที่ให้ F1 สูงที่สุด
        f1 = max(_token_f1(pred_text, ref) for ref in ref_texts)

        total_em += em
        total_f1 += f1

    n = len(predictions)
    return {
        "qa_exact_match": round(total_em / n, 4),
        "qa_f1":          round(total_f1 / n, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined — ใช้ใน trainer.evaluate()
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    task:        str,
    predictions: Any,
    references:  Any,
) -> Dict[str, float]:
    """Router function — เรียก metric ที่ถูกต้องตาม task"""
    if task == "ner":
        return compute_ner_metrics(predictions, references)
    elif task == "sentiment":
        return compute_sentiment_metrics(predictions, references)
    elif task == "qa":
        return compute_qa_metrics(predictions, references)
    else:
        raise ValueError(f"task ไม่รู้จัก: '{task}'")