# scripts/evaluate.py
import argparse
import os
import sys
import yaml
import pathlib
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
root = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from tokenizer.thai_tokenizer import ThaiTokenizer
from model.encoder import ModelConfig
from inference.pipeline import ThaiNLPModel
from data.datasets import NERDataset, SentimentDataset, QADataset
from data.collator import MultiTaskDataCollator
from evaluation.metrics import compute_ner_metrics, compute_sentiment_metrics, compute_qa_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NER_LABEL2ID = {
    "O":     0,
    "B-PER": 1, "I-PER": 2,
    "B-ORG": 3, "I-ORG": 4,
    "B-LOC": 5, "I-LOC": 6,
}
NER_ID2LABEL = {v: k for k, v in NER_LABEL2ID.items()}

def evaluate_ner(model, dataloader, tokenizer, device, limit_batches=None):
    model.eval()
    all_preds = []
    all_refs = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating NER")):
            if limit_batches is not None and idx >= limit_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            hidden, _ = model.encoder(input_ids, attention_mask)
            logits = model.ner_head(hidden) # (B, T, num_labels)
            pred_ids = logits.argmax(dim=-1) # (B, T)
            
            for seq_preds, seq_refs in zip(pred_ids, batch["ner_labels"]):
                pred_seq = []
                ref_seq = []
                for p_id, r_id in zip(seq_preds, seq_refs):
                    r_id = r_id.item()
                    if r_id == -100:
                        continue
                    pred_seq.append(NER_ID2LABEL.get(p_id.item(), "O"))
                    ref_seq.append(NER_ID2LABEL.get(r_id, "O"))
                all_preds.append(pred_seq)
                all_refs.append(ref_seq)
                
    metrics = compute_ner_metrics(all_preds, all_refs)
    return metrics

def evaluate_sentiment(model, dataloader, device, limit_batches=None):
    model.eval()
    all_preds = []
    all_refs = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating Sentiment")):
            if limit_batches is not None and idx >= limit_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            hidden, _ = model.encoder(input_ids, attention_mask)
            logits = model.sentiment_head(hidden, attention_mask) # (B, num_classes)
            pred_ids = logits.argmax(dim=-1).tolist()
            ref_ids = batch["sentiment_labels"].tolist()
            
            all_preds.extend(pred_ids)
            all_refs.extend(ref_ids)
            
    metrics = compute_sentiment_metrics(all_preds, all_refs)
    return metrics

def evaluate_qa(model, dataloader, tokenizer, device, limit_batches=None):
    model.eval()
    all_preds = []
    all_refs = []
    MAX_ANSWER_LEN = 30
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating QA")):
            if limit_batches is not None and idx >= limit_batches:
                break
            input_ids_batch = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            context_starts = batch["context_start"].tolist()
            
            # Forward pass
            hidden, _ = model.encoder(input_ids_batch, attention_mask)
            
            for i in range(len(context_starts)):
                ctx_start = context_starts[i]
                hidden_single = hidden[i:i+1] # (1, T, D)
                start_logits, end_logits = model.qa_head(hidden_single, context_start=ctx_start)
                
                start_logits = start_logits[0] # (T,)
                end_logits = end_logits[0] # (T,)
                
                input_ids = batch["input_ids"][i].tolist()
                seq_len = len(input_ids)
                
                best_score = float("-inf")
                best_start = ctx_start
                best_end = ctx_start
                
                for s in range(ctx_start, seq_len):
                    for e in range(s, min(s + MAX_ANSWER_LEN, seq_len)):
                        score = start_logits[s].item() + end_logits[e].item()
                        if score > best_score:
                            best_score = score
                            best_start = s
                            best_end = e
                            
                pred_answer = tokenizer.decode(input_ids[best_start:best_end + 1], skip_special_tokens=True)
                
                ref_start = batch["qa_start_labels"][i].item()
                ref_end = batch["qa_end_labels"][i].item()
                ref_answer = tokenizer.decode(input_ids[ref_start:ref_end + 1], skip_special_tokens=True)
                
                all_preds.append({"prediction_text": pred_answer})
                all_refs.append({"answers": {"text": [ref_answer]}})
                
    metrics = compute_qa_metrics(all_preds, all_refs)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-task Thai NLP Toolkit model")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing config and checkpoints")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best", help="Checkpoint folder name under model_dir (e.g. checkpoint_best, checkpoint_final)")
    parser.add_argument("--task", type=str, default="all", choices=["ner", "sentiment", "qa", "all"],
                        help="Specific task to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cpu, cuda, mps)")
    parser.add_argument("--limit_batches", type=int, default=None, help="Limit number of batches to evaluate (for quick testing)")

    args = parser.parse_args()
    log.info(f"Evaluating task: {args.task} for model in: {args.model_dir} (checkpoint: {args.checkpoint}) on device: {args.device}")

    # Set up device
    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    elif device_name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        device_name = "cpu"
    device = torch.device(device_name)

    # Load config from checkpoint or model_dir
    ckpt_dir = os.path.join(args.model_dir, args.checkpoint)
    ckpt_file = os.path.join(ckpt_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint file not found at: {ckpt_file}")
        
    log.info(f"Loading checkpoint state from: {ckpt_file}")
    checkpoint_state = torch.load(ckpt_file, map_location="cpu")
    
    config = checkpoint_state.get("config")
    if config is None:
        # Fallback: try to load config.yaml from model_dir
        config_path = os.path.join(args.model_dir, "config.yaml")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found in checkpoint and no config.yaml found in {args.model_dir}")
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
            
    paths = config["paths"]
    data_dir = paths["data_dir"]
    max_seq_len = config["model"].get("max_seq_len", 512)

    # Initialize tokenizer
    tokenizer_model_path = paths["tokenizer_model"]
    if not os.path.exists(tokenizer_model_path):
        tokenizer_model_path = "./tokenizer/thai_bpe.model"
    log.info(f"Loading tokenizer from: {tokenizer_model_path}")
    tokenizer = ThaiTokenizer(tokenizer_model_path)

    # Initialize model
    model_config = ModelConfig(**config["model"])
    model = ThaiNLPModel(model_config)
    model.load_state_dict(checkpoint_state["model_state"])
    model.to(device)
    model.eval()
    log.info("Model loaded and moved to device successfully.")

    batch_size = config["training"].get("batch_size", 32)
    collator = MultiTaskDataCollator(tokenizer, max_length=max_seq_len)

    tasks_to_eval = ["ner", "sentiment", "qa"] if args.task == "all" else [args.task]

    for task in tasks_to_eval:
        log.info(f"Starting evaluation for task: {task.upper()}")
        if task == "ner":
            test_path = os.path.join(data_dir, "ner_test.jsonl")
            if not os.path.exists(test_path):
                log.error(f"Test file not found for NER: {test_path}")
                continue
            dataset = NERDataset(test_path, tokenizer, max_length=max_seq_len)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
            metrics = evaluate_ner(model, loader, tokenizer, device, limit_batches=args.limit_batches)
            print(f"=== NER Test Metrics ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
                
        elif task == "sentiment":
            test_path = os.path.join(data_dir, "sent_test.tsv")
            if not os.path.exists(test_path):
                log.error(f"Test file not found for Sentiment: {test_path}")
                continue
            dataset = SentimentDataset(test_path, tokenizer, max_length=max_seq_len)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
            metrics = evaluate_sentiment(model, loader, device, limit_batches=args.limit_batches)
            print(f"=== Sentiment Test Metrics ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
                
        elif task == "qa":
            test_path = os.path.join(data_dir, "qa_test.json")
            if not os.path.exists(test_path):
                log.error(f"Test file not found for QA: {test_path}")
                continue
            dataset = QADataset(test_path, tokenizer, max_length=max_seq_len)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
            metrics = evaluate_qa(model, loader, tokenizer, device, limit_batches=args.limit_batches)
            print(f"=== QA Test Metrics ===")
            for k, v in metrics.items():
                print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
