# scripts/train.py
import os
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
        description="Train multi-task Thai NLP Toolkit"
    )
    parser.add_argument("--config",  default="./configs/base_config.yaml")
    parser.add_argument("--device",  default="auto",
                        choices=["auto","cuda","cpu","mps"])
    parser.add_argument("--resume",  default=None,
                        help="path ของ checkpoint ที่จะ resume จาก")
    parser.add_argument("--data_dir", default="./data/raw",
                        help="directory ที่มี dataset files")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.device != "auto":
        config["device"] = args.device

    log.info(f"config: {args.config}")
    log.info(f"data:   {args.data_dir}")

    # ── Tokenizer ─────────────────────────────────────────────────────
    from tokenizer.thai_tokenizer import ThaiTokenizer
    tok_dir = config["paths"].get("tokenizer_model", "tokenizer/")
    # ตัด .model ออกถ้า path ชี้ที่ไฟล์โดยตรง
    if tok_dir.endswith(".model"):
        tok_dir = os.path.dirname(tok_dir)
    tokenizer = ThaiTokenizer.from_pretrained(tok_dir)
    log.info(f"tokenizer: vocab_size={tokenizer.vocab_size:,}")

    # ── Model ─────────────────────────────────────────────────────────
    from model.encoder import ModelConfig
    from inference.pipeline import ThaiNLPModel

    model_cfg = ModelConfig(**config["model"])
    model     = ThaiNLPModel(model_cfg)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"model params: {n_params:,}")

    # ── Datasets ──────────────────────────────────────────────────────
    from data.datasets  import NERDataset, SentimentDataset, QADataset
    from data.collator  import MultiTaskDataCollator

    tc         = config["training"]
    batch_size = tc.get("batch_size", 32)
    max_len    = config["model"].get("max_seq_len", 512)
    collator   = MultiTaskDataCollator(tokenizer, max_length=max_len)

    def make_loader(dataset, bs, shuffle=True):
        return DataLoader(
            dataset, batch_size=bs,
            shuffle=shuffle, collate_fn=collator,
            num_workers=min(4, os.cpu_count()),
            pin_memory=torch.cuda.is_available(),
        )

    train_loaders, val_loaders = {}, {}

    # NER
    ner_train = os.path.join(args.data_dir, "ner_train.jsonl")
    ner_val   = os.path.join(args.data_dir, "ner_val.jsonl")
    if os.path.exists(ner_train):
        train_loaders["ner"] = make_loader(NERDataset(ner_train, tokenizer, max_len), batch_size)
        val_loaders["ner"]   = make_loader(NERDataset(ner_val,   tokenizer, max_len), batch_size*2, shuffle=False)
        log.info(f"NER: {len(train_loaders['ner'].dataset):,} train samples")
    else:
        log.warning(f"ไม่พบ {ner_train} — ข้าม NER task")

    # Sentiment
    sent_train = os.path.join(args.data_dir, "sent_train.tsv")
    sent_val   = os.path.join(args.data_dir, "sent_val.tsv")
    if os.path.exists(sent_train):
        train_loaders["sentiment"] = make_loader(SentimentDataset(sent_train, tokenizer, max_len), batch_size)
        val_loaders["sentiment"]   = make_loader(SentimentDataset(sent_val,   tokenizer, max_len), batch_size*2, shuffle=False)
        log.info(f"Sentiment: {len(train_loaders['sentiment'].dataset):,} train samples")
    else:
        log.warning(f"ไม่พบ {sent_train} — ข้าม Sentiment task")

    # QA
    qa_train = os.path.join(args.data_dir, "qa_train.json")
    qa_val   = os.path.join(args.data_dir, "qa_val.json")
    if os.path.exists(qa_train):
        train_loaders["qa"] = make_loader(QADataset(qa_train, tokenizer, max_len), batch_size // 2)
        val_loaders["qa"]   = make_loader(QADataset(qa_val,   tokenizer, max_len), batch_size,    shuffle=False)
        log.info(f"QA: {len(train_loaders['qa'].dataset):,} train samples")
    else:
        log.warning(f"ไม่พบ {qa_train} — ข้าม QA task")

    if not train_loaders:
        log.error("ไม่มี dataset เลย — รัน data/download.py ก่อน")
        return

    # ── Trainer ───────────────────────────────────────────────────────
    from training.trainer import MultiTaskTrainer

    trainer = MultiTaskTrainer(
        model        = model,
        tokenizer    = tokenizer,
        config       = config,
        train_loaders= train_loaders,
        val_loaders  = val_loaders,
    )

    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()