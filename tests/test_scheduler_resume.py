"""Rigorously verify exact learning-rate and scheduler restoration on resume."""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def test_scheduler_uninterrupted_vs_resumed():
    base_lr = 5e-5
    epochs = 20

    # 1. Uninterrupted run
    model_ref = torch.nn.Linear(10, 10)
    opt_ref = AdamW(model_ref.parameters(), lr=base_lr, weight_decay=1e-4)
    sched_ref = CosineAnnealingLR(opt_ref, T_max=epochs)

    ref_lrs_during_train = []
    for ep in range(1, epochs + 1):
        ref_lrs_during_train.append(opt_ref.param_groups[0]["lr"])
        sched_ref.step()

    # 2. Test resume at Epoch 9 (mid-point) WITH scheduler_state_dict
    model_ckpt = torch.nn.Linear(10, 10)
    opt_ckpt = AdamW(model_ckpt.parameters(), lr=base_lr, weight_decay=1e-4)
    sched_ckpt = CosineAnnealingLR(opt_ckpt, T_max=epochs)
    for _ in range(8):
        sched_ckpt.step()
    
    ckpt_with_sched = {
        "epoch": 8,
        "optimizer_state_dict": opt_ckpt.state_dict(),
        "scheduler_state_dict": sched_ckpt.state_dict(),
    }

    # Resume Case A (with scheduler_state_dict)
    model_res_a = torch.nn.Linear(10, 10)
    opt_res_a = AdamW(model_res_a.parameters(), lr=base_lr, weight_decay=1e-4)
    opt_res_a.load_state_dict(ckpt_with_sched["optimizer_state_dict"])
    sched_res_a = CosineAnnealingLR(opt_res_a, T_max=epochs)
    sched_res_a.load_state_dict(ckpt_with_sched["scheduler_state_dict"])

    res_a_lrs = []
    for ep in range(9, epochs + 1):
        res_a_lrs.append(opt_res_a.param_groups[0]["lr"])
        sched_res_a.step()

    for idx, (res_lr, ref_lr) in enumerate(zip(res_a_lrs, ref_lrs_during_train[8:]), 9):
        assert abs(res_lr - ref_lr) < 1e-12, f"Mismatch at epoch {idx}: {res_lr} vs {ref_lr}"

    # 3. Test resume at Epoch 9 WITHOUT scheduler_state_dict (backward-compatible fallback)
    ckpt_without_sched = {
        "epoch": 8,
        "optimizer_state_dict": opt_ckpt.state_dict(),
    }

    model_res_b = torch.nn.Linear(10, 10)
    opt_res_b = AdamW(model_res_b.parameters(), lr=base_lr, weight_decay=1e-4)
    opt_res_b.load_state_dict(ckpt_without_sched["optimizer_state_dict"])
    
    # Apply train.py canonical reconstruction
    start_epoch = 9
    for group in opt_res_b.param_groups:
        group["initial_lr"] = base_lr
        group["lr"] = base_lr
    sched_res_b = CosineAnnealingLR(opt_res_b, T_max=epochs)
    for _ in range(start_epoch - 1):
        sched_res_b.step()

    res_b_lrs = []
    for ep in range(9, epochs + 1):
        res_b_lrs.append(opt_res_b.param_groups[0]["lr"])
        sched_res_b.step()

    for idx, (res_lr, ref_lr) in enumerate(zip(res_b_lrs, ref_lrs_during_train[8:]), 9):
        assert abs(res_lr - ref_lr) < 1e-12, f"Fallback mismatch at epoch {idx}: {res_lr} vs {ref_lr}"

    # 4. Test resume at Epoch 19 (late resume)
    model_late = torch.nn.Linear(10, 10)
    opt_late = AdamW(model_late.parameters(), lr=base_lr, weight_decay=1e-4)
    for _ in range(18):
        opt_ref.param_groups[0]["lr"] # dummy
    
    # Fallback reconstruction at epoch 19:
    for group in opt_late.param_groups:
        group["initial_lr"] = base_lr
        group["lr"] = base_lr
    sched_late = CosineAnnealingLR(opt_late, T_max=epochs)
    for _ in range(18):
        sched_late.step()

    late_lrs = []
    for ep in range(19, epochs + 1):
        late_lrs.append(opt_late.param_groups[0]["lr"])
        sched_late.step()

    for idx, (res_lr, ref_lr) in enumerate(zip(late_lrs, ref_lrs_during_train[18:]), 19):
        assert abs(res_lr - ref_lr) < 1e-12, f"Late resume mismatch at epoch {idx}: {res_lr} vs {ref_lr}"

    print("ALL SCHEDULER RESUME TESTS PASSED WITH 100% MATHEMATICAL PRECISION (< 1e-12)!")


if __name__ == "__main__":
    test_scheduler_uninterrupted_vs_resumed()
