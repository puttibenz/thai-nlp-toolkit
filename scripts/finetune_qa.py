# scripts/finetune_qa.py
import os
import sys
import yaml
import json
import time
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from typing import Dict, List, Optional, Any

# Configure UTF-8 encoding for standard output/error on Windows
if sys.platform.startswith("win"):
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
    if sys.stderr.encoding != 'utf-8':
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

# Add project root to sys.path
import pathlib
root = pathlib.Path(__file__).resolve().parent
while root.parent != root:
    if (root / "requirements.txt").exists() or (root / "README.md").exists():
        sys.path.append(str(root))
        break
    root = root.parent

from tokenizer.thai_tokenizer import ThaiTokenizer
from model.encoder import ModelConfig
from inference.pipeline import ThaiNLPModel
from data.datasets import QADataset
from data.collator import MultiTaskDataCollator
from training.losses import MultiTaskLoss
from training.optimizer import get_optimizer, get_scheduler
from evaluation.metrics import compute_qa_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune QA head only or with top encoder layers"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Directory ของ pre-trained model outputs (ที่มี config.yaml, checkpoint_final, ฯลฯ)"
    )
    parser.add_argument(
        "--checkpoint_name", type=str, default="checkpoint_final",
        help="ชื่อ subfolder checkpoint ที่ต้องการโหลดมาเป็น base (default: checkpoint_final)"
    )
    parser.add_argument(
        "--config", default="./configs/qa_finetune_config.yaml",
        help="Path ของ QA fine-tuning config"
    )
    parser.add_argument(
        "--data_dir", default="./data/raw",
        help="Directory ที่เก็บ dataset json/jsonl/tsv"
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory สำหรับเซฟ output ใหม่ (default: จะสร้างใน model_dir/checkpoint_qa_finetuned)"
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "cpu", "mps"]
    )
    return parser.parse_args()


def get_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_name)


def freeze_model_except_qa(model: nn.Module):
    """Freeze ทั้งโมเดล ยกเว้น qa_head"""
    for name, param in model.named_parameters():
        if "qa_head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    log.info("Frozen all parameters except self.model.qa_head")


def unfreeze_top_encoder_layers(model: nn.Module, num_layers_to_unfreeze: int = 2):
    """ปลดล็อค encoder N block บนสุด"""
    # blocks ใน encoder อยู่ใน self.model.encoder.blocks (ModuleList)
    total_layers = len(model.encoder.blocks)
    unfreeze_from = total_layers - num_layers_to_unfreeze

    log.info(f"Unfreezing encoder blocks starting from block {unfreeze_from} (total={total_layers})")

    # ปลด blocks
    for idx in range(unfreeze_from, total_layers):
        for param in model.encoder.blocks[idx].parameters():
            param.requires_grad = True

    # ปลด norm/poolers ปลายทางถ้าต้องการ (เช่น final norm)
    if hasattr(model.encoder, "norm"):
        for param in model.encoder.norm.parameters():
            param.requires_grad = True

    log.info("Top encoder blocks are now trainable.")


def evaluate_qa_during_training(model, loader, device, tokenizer) -> Dict[str, float]:
    """คำนวณ EM และ F1 ของ QA เพื่อประเมินผลระหว่างเทรน"""
    model.eval()
    all_preds, all_refs = [], []
    global_idx = 0

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            ctx = batch.get("context_start")
            
            # Forward
            hidden, _ = model.encoder(ids, mask)
            s_logits, e_logits = model.qa_head(hidden, ctx)

            # Mask out padding positions using attention_mask
            if mask is not None:
                pad_mask = (mask == 0)
                s_logits = s_logits.masked_fill(pad_mask, torch.finfo(s_logits.dtype).min)
                e_logits = e_logits.masked_fill(pad_mask, torch.finfo(e_logits.dtype).min)

            MAX_ANSWER_LEN = 30
            for i in range(ids.size(0)):
                ctx_s = ctx[i].item() if ctx is not None else 0
                seq_len = (mask[i] == 1).sum().item()

                best_score = float("-inf")
                best_s, best_e = ctx_s, ctx_s

                # Constrained joint span search
                for s in range(ctx_s, seq_len):
                    for e in range(s, min(s + MAX_ANSWER_LEN, seq_len)):
                        score = s_logits[i, s].item() + e_logits[i, e].item()
                        if score > best_score:
                            best_score = score
                            best_s, best_e = s, e

                answer_ids = batch["input_ids"][i][best_s:best_e+1].tolist()
                pred_text = tokenizer.decode(answer_ids, skip_special_tokens=True)
                all_preds.append({"prediction_text": pred_text})

                # Reference
                ex = loader.dataset.examples[global_idx]
                answers = ex.get("answers", [])
                text_list = answers if isinstance(answers, list) else answers.get("text", [])
                all_refs.append({"answers": {"text": text_list}})
                global_idx += 1

    metrics = compute_qa_metrics(all_preds, all_refs)
    return metrics


def save_finetuned_checkpoint(model, optimizer, scheduler, scaler, step, best_metric, output_path, config):
    os.makedirs(output_path, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "global_step": step,
        "best_metric": best_metric,
        "config": config,
    }, os.path.join(output_path, "checkpoint.pt"))
    log.info(f"Finetuned checkpoint saved → {output_path}")


def main():
    args = parse_args()

    # ── Load Configs ──────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # โหลด config จาก base model directory
    base_config_path = os.path.join(args.model_dir, "config.yaml")
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"ไม่พบ config.yaml ใน {args.model_dir}")
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    # ── Device ────────────────────────────────────────────────────────
    device = get_device(args.device)
    log.info(f"Using device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────
    tok_dir = os.path.join(args.model_dir, "tokenizer")
    tokenizer = ThaiTokenizer.from_pretrained(tok_dir)
    log.info(f"Tokenizer loaded from {tok_dir}")

    # ── Datasets & Loaders ──────────────────────────────────────────────
    qa_train_path = os.path.join(args.data_dir, "qa_train.json")
    qa_val_path = os.path.join(args.data_dir, "qa_val.json")

    max_len = base_config["model"].get("max_seq_len", 512)
    collator = MultiTaskDataCollator(tokenizer, max_length=max_len)

    train_dataset = QADataset(qa_train_path, tokenizer, max_len)
    val_dataset = QADataset(qa_val_path, tokenizer, max_len)

    qafc = config["qa_finetune"]
    batch_size = qafc.get("batch_size", 16)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator,
        num_workers=min(4, os.cpu_count()), pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=collator,
        num_workers=min(4, os.cpu_count()), pin_memory=torch.cuda.is_available()
    )

    # ── Load Model ────────────────────────────────────────────────────
    model_cfg = ModelConfig(**base_config["model"])
    model = ThaiNLPModel(model_cfg)

    # โหลด checkpoint_final
    ckpt_dir = os.path.join(args.model_dir, args.checkpoint_name)
    ckpt_file = os.path.join(ckpt_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"ไม่พบ checkpoint: {ckpt_file}")
    
    log.info(f"Loading checkpoint from {ckpt_file} ...")
    ckpt = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # Loss Function (QA เท่านั้น)
    # ให้น้ำหนัก QA สูงสุด
    loss_fn = MultiTaskLoss(task_weights={"qa": 1.0}).to(device)

    # Output directory
    if args.output_dir is None:
        output_dir = os.path.join(args.model_dir, "checkpoint_qa_finetuned")
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # บันทึก config.yaml สำหรับ pipeline โหลดต่อได้
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(base_config, f)
    # คัดลอก tokenizer ไปด้วยเพื่อให้เป็น model folder ที่สมบูรณ์
    import shutil
    dest_tok_dir = os.path.join(output_dir, "tokenizer")
    if os.path.exists(dest_tok_dir):
        shutil.rmtree(dest_tok_dir)
    shutil.copytree(tok_dir, dest_tok_dir)

    # ── Phase 1: Train QA Head Only ───────────────────────────────────
    phase1_steps = qafc.get("phase1_steps", 5000)
    
    use_amp = base_config["training"].get("mixed_precision", True) and device.type == "cuda"
    scaler = GradScaler(device=device.type, enabled=use_amp)

    best_qa_f1 = 0.0
    global_step = 0
    grad_accum_steps = qafc.get("grad_accum_steps", 4)
    eval_every = qafc.get("eval_every", 500)

    start_time = time.time()
    
    # ตัวแปรสำหรับเก็บ optimizer และ scheduler ที่ทำงานเป็นตัวสุดท้าย เพื่อใช้เซฟ checkpoint ตอนท้าย
    active_optimizer = None
    active_scheduler = None

    def run_train_phase(max_steps, optimizer, scheduler, current_phase_name):
        nonlocal global_step, best_qa_f1
        model.train()
        step_in_phase = 0
        running_loss = 0.0
        n_updates = 0
        optimizer.zero_grad()

        iter_loader = iter(train_loader)
        save_every = qafc.get("save_every", 1000)

        while step_in_phase < max_steps * grad_accum_steps:
            try:
                batch = next(iter_loader)
            except StopIteration:
                iter_loader = iter(train_loader)
                batch = next(iter_loader)

            with autocast(device_type=device.type, enabled=use_amp):
                # Forward
                hidden, _ = model.encoder(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                start_logits, end_logits = model.qa_head(hidden, batch["context_start"].to(device))
                
                predictions = {"qa_start": start_logits, "qa_end": end_logits}
                targets = {
                    "qa_start_labels": batch["qa_start_labels"].to(device),
                    "qa_end_labels": batch["qa_end_labels"].to(device)
                }
                losses = loss_fn(predictions, targets)
                loss = losses["total"] / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += losses["total"].item()
            step_in_phase += 1

            if step_in_phase % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                # ดึงเฉพาะ params ที่ trainable ณ ขณะนั้นมาทำ clip
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                nn.utils.clip_grad_norm_(trainable_params, qafc.get("max_grad_norm", 1.0))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                n_updates += 1

                if global_step % 100 == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    avg_loss = running_loss / grad_accum_steps / 100
                    elapsed = (time.time() - start_time) / 60
                    log.info(
                        f"[{current_phase_name}] Step {global_step:>5}/{max_steps} | "
                        f"Loss {avg_loss:.4f} | LR {lr:.2e} | Time {elapsed:.1f}m"
                    )
                    running_loss = 0.0

                # Evaluation
                if global_step % eval_every == 0:
                    metrics = evaluate_qa_during_training(model, val_loader, device, tokenizer)
                    em = metrics["qa_exact_match"]
                    f1 = metrics["qa_f1"]
                    log.info(f"[{current_phase_name}] Eval at step {global_step}: EM={em:.4f} | F1={f1:.4f}")
                    
                    if f1 > best_qa_f1:
                        best_qa_f1 = f1
                        best_path = os.path.join(output_dir, "checkpoint_best")
                        save_finetuned_checkpoint(model, optimizer, scheduler, scaler, global_step, best_qa_f1, best_path, base_config)
                        log.info(f"New Best Checkpoint saved with F1={f1:.4f}")
                    
                    model.train()

                # บันทึก checkpoint เป็นระยะ
                if global_step % save_every == 0:
                    step_path = os.path.join(output_dir, f"checkpoint_step{global_step}")
                    save_finetuned_checkpoint(model, optimizer, scheduler, scaler, global_step, best_qa_f1, step_path, base_config)

    if phase1_steps > 0:
        log.info("=== Phase 1: Training QA Head Only ===")
        freeze_model_except_qa(model)

        phase1_lr = qafc.get("phase1_lr", 3e-4)
        phase1_warmup = qafc.get("phase1_warmup", 200)

        # Optimizer & Scheduler สำหรับ Phase 1
        trainable_params_p1 = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params_p1, lr=phase1_lr, weight_decay=0.01)
        scheduler = get_scheduler(optimizer, warmup_steps=phase1_warmup, max_steps=phase1_steps, schedule="cosine")
        
        active_optimizer = optimizer
        active_scheduler = scheduler
        
        # รัน Phase 1
        run_train_phase(phase1_steps, optimizer, scheduler, "Phase 1")
    else:
        log.info("=== Phase 1 skipped (steps == 0) ===")

    # ── Phase 2: Unfreeze Top Encoder Layers ─────────────────────────
    phase2_steps = qafc.get("phase2_steps", 10000)
    if phase2_steps > 0:
        log.info("=== Phase 2: Training QA Head + Top Encoder Layers ===")
        unfreeze_top_encoder_layers(model, qafc.get("unfreeze_layers", 2))

        # กำหนด parameter groups และ learning rates แยกกัน
        # Group 1: Encoder layers ที่ถูก unfreeze
        # Group 2: QA Head (lr สูงกว่า)
        encoder_params = []
        qa_head_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "qa_head" in name:
                    qa_head_params.append(param)
                else:
                    encoder_params.append(param)

        phase2_head_lr = qafc.get("phase2_head_lr", 1e-4)
        phase2_encoder_lr = qafc.get("phase2_encoder_lr", 5e-5)
        phase2_warmup = qafc.get("phase2_warmup", 500)

        param_groups = [
            {"params": qa_head_params, "lr": phase2_head_lr},
            {"params": encoder_params, "lr": phase2_encoder_lr}
        ]

        optimizer_p2 = torch.optim.AdamW(param_groups, weight_decay=0.01)
        scheduler_p2 = get_scheduler(optimizer_p2, warmup_steps=phase2_warmup, max_steps=phase2_steps, schedule="cosine")

        active_optimizer = optimizer_p2
        active_scheduler = scheduler_p2

        # รัน Phase 2
        run_train_phase(phase2_steps, optimizer_p2, scheduler_p2, "Phase 2")
    else:
        log.info("=== Phase 2 skipped (steps == 0) ===")

    # Save final checkpoint
    if active_optimizer is not None:
        final_path = os.path.join(output_dir, "checkpoint_final")
        save_finetuned_checkpoint(model, active_optimizer, active_scheduler, scaler, global_step, best_qa_f1, final_path, base_config)
        log.info(f"QA Fine-tuning complete! Best Validation F1: {best_qa_f1:.4f}")
    else:
        log.info("No training was performed (both phases set to 0 steps).")


if __name__ == "__main__":
    main()
