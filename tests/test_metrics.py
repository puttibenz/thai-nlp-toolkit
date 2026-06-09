from evaluation.metrics import (
    compute_ner_metrics,
    compute_sentiment_metrics,
    compute_qa_metrics,
)

# ── NER ─────────────────────────────────────────────────────────
pred_ner = [["B-PER","I-PER","O","B-LOC"],
            ["O","B-ORG","I-ORG","O"]]
ref_ner  = [["B-PER","I-PER","O","B-LOC"],
            ["O","B-ORG","O","O"]]          # I-ORG ผิด

m = compute_ner_metrics(pred_ner, ref_ner)
assert m["ner_f1"] > 0
assert m["ner_precision"] <= 1.0
assert m["ner_recall"] <= 1.0
# perfect ได้ถ้า pred ถูกทุก entity
perfect = compute_ner_metrics(ref_ner, ref_ner)
assert perfect["ner_f1"] == 1.0, f"perfect NER F1 ควรเป็น 1.0: {perfect}"
print(f"NER metrics: {m}")
print("NER OK")

# ── Sentiment ────────────────────────────────────────────────────
pred_sent = [2, 0, 1, 2, 0]
ref_sent  = [2, 0, 1, 0, 1]   # ผิด 2 ตัว

m = compute_sentiment_metrics(pred_sent, ref_sent)
assert m["sentiment_accuracy"] == 0.6   # ถูก 3/5
assert 0 < m["sentiment_macro_f1"] <= 1.0
# perfect
perfect = compute_sentiment_metrics(ref_sent, ref_sent)
assert perfect["sentiment_accuracy"]  == 1.0
assert perfect["sentiment_macro_f1"]  == 1.0
print(f"Sentiment metrics: {m}")
print("Sentiment OK")

# ── QA ───────────────────────────────────────────────────────────
pred_qa = [
    {"prediction_text": "พ.ศ. 2325"},
    {"prediction_text": "รัชกาลที่ 1"},
    {"prediction_text": "ผิดทั้งหมด"},
]
ref_qa = [
    {"answers": {"text": ["พ.ศ. 2325", "2325"]}},     # หลาย answer
    {"answers": {"text": ["รัชกาลที่ 1"]}},
    {"answers": {"text": ["สมชาย"]}},
]

m = compute_qa_metrics(pred_qa, ref_qa)
assert m["qa_exact_match"] > 0       # อย่างน้อย 2 ตัวถูก
assert m["qa_f1"] >= m["qa_exact_match"]  # F1 ≥ EM เสมอ

# normalize test: "พ.ศ. 2325" vs "พศ 2325" ควรได้ partial F1
from evaluation.metrics import _token_f1
f1 = _token_f1("พ.ศ. 2325", "พศ 2325")
assert f1 > 0, "token F1 ควรได้ partial credit"
print(f"QA metrics: {m}")
print("QA OK")