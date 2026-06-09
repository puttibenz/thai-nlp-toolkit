import os
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Any

from .losses import MultiTaskLoss
from .optimizer import get_optimizer, get_scheduler

log = logging.getLogger(__name__)

class MultiTaskTrainer:
    """
    Training loop สำหรับ multi-task NLP

    Features:
    - Mixed precision (torch.amp) — ใช้ GPU ได้เต็มประสิทธิภาพ
    - Gradient accumulation — effective batch size ใหญ่กว่า VRAM จริง
    - Interleaved task batches — หมุน NER → Sentiment → QA ทุก step
    - Checkpoint save/load — resume training ได้
    - Loss logging แยกต่าง task
    """

    def __init__(
            self,
            model,
            tokenizer,
            config,
            train_loaders: Dict[str, DataLoader],
            val_loaders: Dict[str, DataLoader],
    ):
        self.model       = model
        self.tokenizer   = tokenizer
        self.config      = config
        self.train_loaders = train_loaders
        self.val_loaders   = val_loaders

        # ── Device ───────────────────────────────────────────────────────
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        log.info(f'training on: {self.device}')

        # ── Training config ───────────────────────────────────────────────
        tc = config["training"]
        self.grad_accum_steps = tc.get("grad_accum_steps", 4)
        self.max_steps        = tc.get("max_steps", 50000)
        self.max_grad_norm    = tc.get("max_grad_norm", 1.0)
        self.eval_every       = config["evaluation"].get("eval_every", 500)
        self.save_every       = config["evaluation"].get("save_every", 1000)
        self.output_dir       = config["paths"].get("output_dir", "./outputs")
        self.use_amp          = tc.get("mixed_precision", True) and self.device.type == "cuda"
        os.makedirs(self.output_dir, exist_ok=True)

        # ── Optimizer + Scheduler ─────────────────────────────────────────
        self.optimizer = get_optimizer(
            model,
            learning_rate=tc["learning_rate"],
            weight_decay=tc.get("weight_decay", 0.01),
        )
        self.scheduler = get_scheduler(
            self.optimizer,
            warmup_steps=tc.get("warmup_steps", 1000),
            max_steps=self.max_steps,
            schedule="cosine",
        )

        # ── Loss function ─────────────────────────────────────────────────
        self.loss_fn = MultiTaskLoss(
            task_weights=tc.get("task_weights", {"ner": 1.0, "sentiment": 0.8, "qa": 1.2})
        )

        # ── Mixed precision scaler ────────────────────────────────────────
        # GradScaler ป้องกัน underflow ของ fp16 gradient
        self.scaler = GradScaler(enabled=self.use_amp)

         # ── State ─────────────────────────────────────────────────────────
        self.global_step   = 0
        self.best_metric   = float("inf")   # track best total loss สำหรับ save best

    # ─────────────────────────────────────────────────────────────────────
    # Forward pass — แยกตาม task
    # ─────────────────────────────────────────────────────────────────────
    
    def _forward(self, batch: Dict[str, torch.Tensor]) -> Dict:
        """
        ส่ง batch เข้า encoder แล้วเลือก head ตาม task
        detect task จาก keys ของ batch
        """
        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # encoder forward — คืน (hidden_states, attn_weights)
        hidden, _ = self.model.encoder(input_ids, attention_mask)

        predictions = {}
        targets     = {}

        if "ner_labels" in batch:
            predictions["ner"] = self.model.ner_head(hidden)
            targets["ner_labels"] = batch["ner_labels"].to(self.device)

        if "sentiment_labels" in batch:
            predictions["sentiment"] = self.model.sentiment_head(hidden, attention_mask)
            targets["sentiment_labels"] = batch["sentiment_labels"].to(self.device)

        if "qa_start_labels" in batch:
            context_start = batch.get("context_start")
            ctx_int = int(context_start[0].item()) if context_start is not None else None
            start_logits, end_logits = self.model.qa_head(hidden, ctx_int)
            predictions["qa_start"] = start_logits
            predictions["qa_end"]   = end_logits
            targets["qa_start_labels"] = batch["qa_start_labels"].to(self.device)
            targets["qa_end_labels"]   = batch["qa_end_labels"].to(self.device)

        return predictions, targets

    # ─────────────────────────────────────────────────────────────────────
    # train_epoch
    # ─────────────────────────────────────────────────────────────────────

    def train_epoch(self) -> Dict[str, float]:
        """
        Training loop 1 epoch — interleave batches จากทุก task

        แนวคิด interleaving:
        แทนที่จะ train task เดียวจนจบแล้วค่อยเปลี่ยน
        จะหมุน task ทุก step: NER → Sentiment → QA → NER → ...
        ทำให้ gradient update สม่ำเสมอและ backbone ไม่ overfit task ใดงาน task เดียว
        """
        self.model.train()

        # สร้าง iterator สำหรับแต่ละ task
        iterators = {
            task: iter(loader)
            for task, loader in self.train_loaders.items()
        }
        task_names = list(iterators.keys())

        # running loss สำหรับ logging
        running = {task: 0.0 for task in task_names}
        running["total"] = 0.0
        n_updates = 0

        self.optimizer.zero_grad()

        step_in_epoch = 0
        while True:
            # หมุน task แบบ round-robin
            task = task_names[step_in_epoch % len(task_names)]

            # ดึง batch — ถ้า iterator หมดให้ reset
            try:
                batch = next(iterators[task])
            except StopIteration:
                iterators[task] = iter(self.train_loaders[task])
                try:
                    batch = next(iterators[task])
                except StopIteration:
                    break   # loader ว่างจริงๆ

            # ── Forward + Loss ────────────────────────────────────────
            with autocast(enabled=self.use_amp):
                predictions, targets = self._forward(batch)
                losses = self.loss_fn(predictions, targets)
                # หาร grad_accum_steps เพื่อ normalize gradient
                loss = losses["total"] / self.grad_accum_steps

            # ── Backward ──────────────────────────────────────────────
            self.scaler.scale(loss).backward()

            # log running loss (ก่อน normalize)
            for key in running:
                if key in losses:
                    running[key] += losses[key].item()

            step_in_epoch += 1

            # ── Optimizer step (ทุก grad_accum_steps) ────────────────
            if step_in_epoch % self.grad_accum_steps == 0:
                # unscale ก่อน clip เพื่อให้ clip ทำงานบน gradient จริง
                self.scaler.unscale_(self.optimizer)
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                n_updates += 1

                # ── Logging ───────────────────────────────────────────
                if self.global_step % 100 == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    avg_total = running["total"] / max(n_updates, 1)
                    log.info(
                        f"step {self.global_step:>6} | "
                        f"loss {avg_total:.4f} | "
                        f"lr {lr:.2e} | "
                        f"grad_norm {grad_norm:.3f}"
                    )

                # ── Eval ──────────────────────────────────────────────
                if self.global_step % self.eval_every == 0:
                    self._run_eval()
                    self.model.train()

                # ── Save ──────────────────────────────────────────────
                if self.global_step % self.save_every == 0:
                    ckpt_path = os.path.join(
                        self.output_dir,
                        f"checkpoint_step{self.global_step}"
                    )
                    self.save_checkpoint(ckpt_path)

                # ── Max steps ─────────────────────────────────────────
                if self.global_step >= self.max_steps:
                    log.info(f"reached max_steps={self.max_steps} — stopping")
                    break

        # คืน average loss ของ epoch นี้
        return {
            key: val / max(n_updates, 1)
            for key, val in running.items()
        }

    # ─────────────────────────────────────────────────────────────────────
    # evaluate
    # ─────────────────────────────────────────────────────────────────────

    def evaluate(self, task: str) -> Dict[str, float]:
        """Evaluate task หนึ่ง task บน val_loader"""
        if task not in self.val_loaders:
            log.warning(f"ไม่มี val_loader สำหรับ task '{task}'")
            return {}

        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        with torch.no_grad():
            for batch in self.val_loaders[task]:
                with autocast(enabled=self.use_amp):
                    predictions, targets = self._forward(batch)
                    losses = self.loss_fn(predictions, targets)

                total_loss += losses["total"].item()
                n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        log.info(f"eval [{task}] loss: {avg_loss:.4f}")
        return {"loss": avg_loss}

    def _run_eval(self) -> Dict[str, float]:
        """Evaluate ทุก task และ log ผล"""
        all_metrics = {}
        for task in self.val_loaders:
            metrics = self.evaluate(task)
            all_metrics[task] = metrics

        # track total val loss สำหรับ save best
        total_val = sum(m.get("loss", 0) for m in all_metrics.values())
        if total_val < self.best_metric:
            self.best_metric = total_val
            best_path = os.path.join(self.output_dir, "checkpoint_best")
            self.save_checkpoint(best_path)
            log.info(f"new best val loss: {total_val:.4f} → saved")

        return all_metrics

    # ─────────────────────────────────────────────────────────────────────
    # Checkpoint
    # ─────────────────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        """
        บันทึก checkpoint ครบถ้วน — resume ได้จากจุดนี้เลย
        บันทึก: model weights, optimizer state, scheduler state,
                scaler state, global_step, best_metric
        """
        os.makedirs(path, exist_ok=True)
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state":    self.scaler.state_dict(),
            "global_step":     self.global_step,
            "best_metric":     self.best_metric,
            "config":          self.config,
        }, os.path.join(path, "checkpoint.pt"))
        log.info(f"checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        โหลด checkpoint และ restore ทุก state
        ใช้สำหรับ resume training หรือ fine-tuning
        """
        ckpt_file = os.path.join(path, "checkpoint.pt")
        if not os.path.exists(ckpt_file):
            raise FileNotFoundError(f"ไม่พบ checkpoint: {ckpt_file}")

        ckpt = torch.load(ckpt_file, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.global_step = ckpt["global_step"]
        self.best_metric = ckpt["best_metric"]

        log.info(
            f"checkpoint loaded ← {path} "
            f"(step={self.global_step}, best={self.best_metric:.4f})"
        )

    # ─────────────────────────────────────────────────────────────────────
    # train (full loop)
    # ─────────────────────────────────────────────────────────────────────

    def train(self, resume_from: Optional[str] = None) -> None:
        """
        Entry point หลัก — เรียก train_epoch วนจนครบ max_steps

        Usage:
            trainer.train()
            trainer.train(resume_from="outputs/checkpoint_step5000")
        """
        if resume_from:
            self.load_checkpoint(resume_from)

        log.info(f"starting training from step {self.global_step}")
        log.info(f"max_steps={self.max_steps} | "
                 f"grad_accum={self.grad_accum_steps} | "
                 f"amp={self.use_amp}")

        start_time = time.time()

        while self.global_step < self.max_steps:
            epoch_losses = self.train_epoch()

            elapsed = (time.time() - start_time) / 60
            log.info(
                f"step {self.global_step} | "
                f"elapsed {elapsed:.1f}m | "
                f"losses: { {k: f'{v:.4f}' for k, v in epoch_losses.items()} }"
            )

            if self.global_step >= self.max_steps:
                break

        # save final checkpoint
        final_path = os.path.join(self.output_dir, "checkpoint_final")
        self.save_checkpoint(final_path)
        log.info("training complete ✓")