# scripts/train.py
import argparse
import os
import sys
import yaml
import pathlib
import logging
import torch
from torch.utils.data import DataLoader

# Add project root to sys.path
root = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from tokenizer.thai_tokenizer import ThaiTokenizer
from model.encoder import ModelConfig
from inference.pipeline import ThaiNLPModel
from data.datasets import NERDataset, SentimentDataset, QADataset
from data.collator import MultiTaskDataCollator
from training.trainer import MultiTaskTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train multi-task Thai NLP Toolkit model")
    parser.add_argument("--config", type=str, default="./configs/base_config.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cpu, cuda, mps)")
    parser.add_argument("--max_steps", type=int, default=None, help="Override maximum training steps")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from")
    
    args = parser.parse_args()
    log.info(f"Starting training process with config: {args.config} on device: {args.device}")

    # Load config
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at {args.config}")
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Set device in config
    if "training" not in config:
        config["training"] = {}
    config["training"]["device"] = args.device

    # Override max_steps if specified
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
        log.info(f"Overriding max_steps to: {args.max_steps}")
        # Ensure warmup_steps is less than max_steps
        warmup_steps = config["training"].get("warmup_steps", 1000)
        if warmup_steps >= args.max_steps:
            config["training"]["warmup_steps"] = max(1, args.max_steps // 2)
            log.info(f"Adjusted warmup_steps to: {config['training']['warmup_steps']} to be less than max_steps")

    paths = config["paths"]
    data_dir = paths["data_dir"]
    max_seq_len = config["model"].get("max_seq_len", 512)

    # Initialize tokenizer
    log.info(f"Loading tokenizer from: {paths['tokenizer_model']}")
    tokenizer = ThaiTokenizer(paths["tokenizer_model"])

    # Initialize datasets
    log.info(f"Loading datasets from: {data_dir}")
    ner_train = NERDataset(os.path.join(data_dir, "ner_train.jsonl"), tokenizer, max_length=max_seq_len)
    ner_val = NERDataset(os.path.join(data_dir, "ner_val.jsonl"), tokenizer, max_length=max_seq_len)

    sent_train = SentimentDataset(os.path.join(data_dir, "sent_train.tsv"), tokenizer, max_length=max_seq_len)
    sent_val = SentimentDataset(os.path.join(data_dir, "sent_val.tsv"), tokenizer, max_length=max_seq_len)

    qa_train = QADataset(os.path.join(data_dir, "qa_train.json"), tokenizer, max_length=max_seq_len)
    qa_val = QADataset(os.path.join(data_dir, "qa_val.json"), tokenizer, max_length=max_seq_len)

    log.info(f"Datasets loaded successfully:")
    log.info(f"  NER: {len(ner_train)} train, {len(ner_val)} val")
    log.info(f"  Sentiment: {len(sent_train)} train, {len(sent_val)} val")
    log.info(f"  QA: {len(qa_train)} train, {len(qa_val)} val")

    # Initialize loaders
    batch_size = config["training"].get("batch_size", 32)
    collator = MultiTaskDataCollator(tokenizer, max_length=max_seq_len)

    train_loaders = {
        "ner": DataLoader(ner_train, batch_size=batch_size, shuffle=True, collate_fn=collator),
        "sentiment": DataLoader(sent_train, batch_size=batch_size, shuffle=True, collate_fn=collator),
        "qa": DataLoader(qa_train, batch_size=batch_size, shuffle=True, collate_fn=collator)
    }

    val_loaders = {
        "ner": DataLoader(ner_val, batch_size=batch_size, shuffle=False, collate_fn=collator),
        "sentiment": DataLoader(sent_val, batch_size=batch_size, shuffle=False, collate_fn=collator),
        "qa": DataLoader(qa_val, batch_size=batch_size, shuffle=False, collate_fn=collator)
    }

    # Initialize model
    log.info("Initializing multi-task model...")
    model_config = ModelConfig(**config["model"])
    model = ThaiNLPModel(model_config)

    # Initialize trainer
    log.info("Initializing trainer...")
    trainer = MultiTaskTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_loaders=train_loaders,
        val_loaders=val_loaders
    )

    # Run training
    log.info("Starting training loop...")
    trainer.train(resume_from=args.resume_from)
    log.info("Training finished!")

if __name__ == "__main__":
    main()
