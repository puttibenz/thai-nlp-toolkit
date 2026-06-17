# scripts/evaluate.py
import os
import json
import yaml
import argparse
import logging
import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Thai NLP Toolkit model"
    )
    parser.add_argument("--model_dir", required=True,
                        help="directory ที่มี checkpoint_best/, tokenizer/, config.yaml")
    parser.add_argument("--task",   default="all",
                        choices=["ner","sentiment","qa","all"])
    parser.add_argument("--data_dir", default="./data/raw")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None,
                        help="บันทึกผล metrics เป็น JSON (optional)")
    return parser.parse_args()


def evaluate_ner(pipeline, data_dir, tokenizer):
    from data.datasets import NERDataset
    from data.collator import MultiTaskDataCollator
    from evaluation.metrics import compute_ner_metrics
    from model.heads.ner_head import NER_ID2LABEL

    test_path = os.path.join(data_dir, "ner_test.jsonl")
    if not os.path.exists(test_path):
        log.warning(f"ไม่พบ {test_path}")
        return {}

    dataset  = NERDataset(test_path, tokenizer, max_length=512)
    loader   = DataLoader(dataset, batch_size=32,
                          collate_fn=MultiTaskDataCollator(tokenizer),
                          shuffle=False)

    all_preds, all_refs = [], []

    pipeline.model.eval()
    with torch.no_grad():
        for batch in loader:
            ids   = batch["input_ids"].to(pipeline.device)
            mask  = batch["attention_mask"].to(pipeline.device)
            labels= batch["ner_labels"]                      # (B, T)

            hidden, _ = pipeline.model.encoder(ids, mask)
            logits    = pipeline.model.ner_head(hidden)      # (B, T, C)
            preds     = logits.argmax(dim=-1).cpu()          # (B, T)

            for pred_row, label_row in zip(preds, labels):
                pred_tags, ref_tags = [], []
                for p, l in zip(pred_row.tolist(), label_row.tolist()):
                    if l == -100:   # padding / non-first subword
                        continue
                    pred_tags.append(NER_ID2LABEL.get(p, "O"))
                    ref_tags.append(NER_ID2LABEL.get(l, "O"))
                all_preds.append(pred_tags)
                all_refs.append(ref_tags)

    metrics = compute_ner_metrics(all_preds, all_refs)
    log.info(f"NER  — F1: {metrics['ner_f1']:.4f} | "
             f"P: {metrics['ner_precision']:.4f} | "
             f"R: {metrics['ner_recall']:.4f}")
    return metrics


def evaluate_sentiment(pipeline, data_dir, tokenizer):
    from data.datasets import SentimentDataset
    from data.collator import MultiTaskDataCollator
    from evaluation.metrics import compute_sentiment_metrics

    test_path = os.path.join(data_dir, "sent_test.tsv")
    if not os.path.exists(test_path):
        log.warning(f"ไม่พบ {test_path}")
        return {}

    dataset = SentimentDataset(test_path, tokenizer, max_length=512)
    loader  = DataLoader(dataset, batch_size=64,
                         collate_fn=MultiTaskDataCollator(tokenizer),
                         shuffle=False)

    all_preds, all_refs = [], []

    pipeline.model.eval()
    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(pipeline.device)
            mask   = batch["attention_mask"].to(pipeline.device)
            labels = batch["sentiment_labels"]

            hidden, _ = pipeline.model.encoder(ids, mask)
            logits    = pipeline.model.sentiment_head(hidden, mask)
            preds     = logits.argmax(dim=-1).cpu()

            all_preds.extend(preds.tolist())
            all_refs.extend(labels.tolist())

    metrics = compute_sentiment_metrics(all_preds, all_refs)
    log.info(f"Sent — Acc: {metrics['sentiment_accuracy']:.4f} | "
             f"Macro-F1: {metrics['sentiment_macro_f1']:.4f}")
    return metrics


def evaluate_qa(pipeline, data_dir, tokenizer):
    from data.datasets import QADataset
    from data.collator import MultiTaskDataCollator
    from evaluation.metrics import compute_qa_metrics

    test_path = os.path.join(data_dir, "qa_test.json")
    if not os.path.exists(test_path):
        log.warning(f"ไม่พบ {test_path}")
        return {}

    dataset = QADataset(test_path, tokenizer, max_length=512)
    loader  = DataLoader(dataset, batch_size=16,
                         collate_fn=MultiTaskDataCollator(tokenizer),
                         shuffle=False)

    all_preds, all_refs = [], []

    pipeline.model.eval()
    with torch.no_grad():
        for batch in loader:
            ids    = batch["input_ids"].to(pipeline.device)
            mask   = batch["attention_mask"].to(pipeline.device)
            ctx    = batch.get("context_start")
            hidden, _ = pipeline.model.encoder(ids, mask)
            s_logits, e_logits = pipeline.model.qa_head(hidden, ctx)

            # Mask out padding positions using attention_mask
            if mask is not None:
                pad_mask = (mask == 0)
                s_logits = s_logits.masked_fill(pad_mask, torch.finfo(s_logits.dtype).min)
                e_logits = e_logits.masked_fill(pad_mask, torch.finfo(e_logits.dtype).min)

            # decode ทีละ sample
            for i in range(ids.size(0)):
                s = s_logits[i].argmax().item()
                e = e_logits[i].argmax().item()
                e = max(s, e)   # end ต้องไม่น้อยกว่า start

                answer_ids = batch["input_ids"][i][s:e+1].tolist()
                pred_text  = tokenizer.decode(answer_ids, skip_special_tokens=True)
                all_preds.append({"prediction_text": pred_text})

            # ref จาก dataset โดยตรง — ดึง original examples
            for idx in range(ids.size(0)):
                ex = dataset.examples[len(all_preds) - ids.size(0) + idx]
                answers = ex.get("answers", [])
                all_refs.append({"answers": {"text": answers}})

    # print top 5 samples for debugging
    log.info("=== Sample Predictions ===")
    for i in range(min(5, len(all_preds))):
        log.info(f"Ref:  {all_refs[i]['answers']['text']}")
        log.info(f"Pred: {all_preds[i]['prediction_text']}")
        log.info("-" * 30)

    metrics = compute_qa_metrics(all_preds, all_refs)
    log.info(f"QA   — EM: {metrics['qa_exact_match']:.4f} | "
             f"F1: {metrics['qa_f1']:.4f}")
    return metrics


def main():
    args = parse_args()

    # ── Load pipeline ──────────────────────────────────────────────────
    from inference.pipeline import ThaiNLPPipeline
    pipeline  = ThaiNLPPipeline(model_dir=args.model_dir, device=args.device)
    tokenizer = pipeline.tokenizer

    # ── Evaluate ───────────────────────────────────────────────────────
    all_metrics = {}
    tasks = ["ner","sentiment","qa"] if args.task == "all" else [args.task]

    fn_map = {
        "ner":       evaluate_ner,
        "sentiment": evaluate_sentiment,
        "qa":        evaluate_qa,
    }

    for task in tasks:
        log.info(f"── evaluating {task} ──")
        metrics = fn_map[task](pipeline, args.data_dir, tokenizer)
        all_metrics[task] = metrics

    # ── Summary ────────────────────────────────────────────────────────
    log.info("══ summary ══════════════════════")
    for task, m in all_metrics.items():
        log.info(f"{task}: { {k: f'{v:.4f}' for k, v in m.items()} }")

    # ── Save JSON ──────────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_metrics, f, indent=2)
        log.info(f"metrics saved → {args.output}")


if __name__ == "__main__":
    main()