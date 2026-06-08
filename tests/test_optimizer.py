import unittest
import torch
import torch.nn as nn
from training.optimizer import get_optimizer, get_scheduler

class TestOptimizer(unittest.TestCase):
    def test_optimizer_and_scheduler(self):
        # dummy model
        model = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        opt   = get_optimizer(model, learning_rate=3e-4, weight_decay=0.01)
        sch   = get_scheduler(opt, warmup_steps=100, max_steps=1000, schedule="cosine")

        # ตรวจ param groups
        self.assertEqual(len(opt.param_groups), 2)
        decay_wd    = opt.param_groups[0]["weight_decay"]
        no_decay_wd = opt.param_groups[1]["weight_decay"]
        self.assertEqual(decay_wd, 0.01, f"decay group ผิด: {decay_wd}")
        self.assertEqual(no_decay_wd, 0.0, f"no-decay group ผิด: {no_decay_wd}")

        # ตรวจ warmup: lr ต้องเพิ่มขึ้นใน warmup_steps แรก
        lrs = []
        for step in range(1000):
            sch.step()
            lrs.append(opt.param_groups[0]["lr"])

        # ช่วง warmup (step 0-99) lr ต้องเพิ่ม
        self.assertGreater(lrs[50], lrs[0], "warmup ไม่ทำงาน")
        # peak อยู่ที่ step 100
        self.assertGreaterEqual(lrs[99], lrs[50], "warmup ไม่ถึง peak")
        # หลัง warmup lr ต้องลด
        self.assertLess(lrs[999], lrs[100], "decay ไม่ทำงาน")
        # lr ไม่ต่ำกว่า min_lr_ratio * peak
        peak_lr = 3e-4
        self.assertGreaterEqual(lrs[999], 0.1 * peak_lr, "lr ต่ำกว่า min_lr_ratio")

        print(f"  lr at step   0: {lrs[0]:.2e}")
        print(f"  lr at step  50: {lrs[50]:.2e}  (warmup)")
        print(f"  lr at step 100: {lrs[100]:.2e}  (peak)")
        print(f"  lr at step 500: {lrs[500]:.2e}  (mid decay)")
        print(f"  lr at step 999: {lrs[999]:.2e}  (end)")
        print("optimizer + scheduler OK")

if __name__ == "__main__":
    unittest.main()
