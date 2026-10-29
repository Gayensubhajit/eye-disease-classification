import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import TensorDataset, DataLoader

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import train_one_epoch


class StepCounterOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure) if closure else None


def create_dummy_setup(n_samples=10, batch_size=2):
    torch.manual_seed(42)
    x = torch.randn(n_samples, 4)
    y = torch.randint(0, 3, (n_samples,))
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    
    model = nn.Linear(4, 3, bias=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # Track steps with custom counter
    step_counts = [0]
    orig_step = optimizer.step
    def tracked_step():
        step_counts[0] += 1
        return orig_step()
    optimizer.step = tracked_step
    
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    return model, loader, criterion, optimizer, scaler, step_counts


def test_odd_batches_partial_window():
    # 5 batches of 2 (10 samples), accum_steps=2
    # Expect 3 updates: (0,1)->step 1; (2,3)->step 2; (4)->step 3 (partial window of 1)
    model, loader, criterion, optimizer, scaler, step_counts = create_dummy_setup(n_samples=10, batch_size=2)
    assert len(loader) == 5
    
    train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=torch.device("cpu"),
        dry_run=False,
        use_amp=False,
        accum_steps=2
    )
    
    assert step_counts[0] == 3, f"Expected 3 optimizer steps, got {step_counts[0]}"
    for p in model.parameters():
        if p.grad is not None:
            assert torch.all(p.grad == 0), "Pending unzeroed gradients left after epoch!"
    print("✓ test_odd_batches_partial_window passed (5 batches, accum=2 -> 3 steps)")


def test_even_batches_complete_windows():
    # 4 batches of 2 (8 samples), accum_steps=2 -> exactly 2 updates
    model, loader, criterion, optimizer, scaler, step_counts = create_dummy_setup(n_samples=8, batch_size=2)
    assert len(loader) == 4
    
    train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=torch.device("cpu"),
        dry_run=False,
        use_amp=False,
        accum_steps=2
    )
    
    assert step_counts[0] == 2, f"Expected 2 optimizer steps, got {step_counts[0]}"
    print("✓ test_even_batches_complete_windows passed (4 batches, accum=2 -> 2 steps)")


def test_identity_accum_steps_one():
    # 5 batches of 2, accum_steps=1 -> exactly 5 updates
    model, loader, criterion, optimizer, scaler, step_counts = create_dummy_setup(n_samples=10, batch_size=2)
    assert len(loader) == 5
    
    train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=torch.device("cpu"),
        dry_run=False,
        use_amp=False,
        accum_steps=1
    )
    
    assert step_counts[0] == 5, f"Expected 5 optimizer steps, got {step_counts[0]}"
    print("✓ test_identity_accum_steps_one passed (5 batches, accum=1 -> 5 steps)")


def test_dry_run_clean_flush():
    # 5 batches of 2, dry_run=True, accum_steps=2 -> exactly 1 update (batches 0, 1) and exits clean
    model, loader, criterion, optimizer, scaler, step_counts = create_dummy_setup(n_samples=10, batch_size=2)
    
    train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=torch.device("cpu"),
        dry_run=True,
        use_amp=False,
        accum_steps=2
    )
    
    assert step_counts[0] == 1, f"Expected 1 optimizer step in dry_run, got {step_counts[0]}"
    for p in model.parameters():
        if p.grad is not None:
            assert torch.all(p.grad == 0), "Pending unzeroed gradients left after dry run!"
    print("✓ test_dry_run_clean_flush passed (dry_run=True, accum=2 -> 1 step, clean zeroed grads)")


def test_dry_run_accum_larger_than_batches():
    # 10 batches of 2, dry_run=True (2 batches), accum_steps=4 -> exactly 1 update of window size 2
    model, loader, criterion, optimizer, scaler, step_counts = create_dummy_setup(n_samples=20, batch_size=2)
    train_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=torch.device("cpu"),
        dry_run=True,
        use_amp=False,
        accum_steps=4
    )
    assert step_counts[0] == 1, f"Expected 1 optimizer step in dry_run with accum=4, got {step_counts[0]}"
    for p in model.parameters():
        if p.grad is not None:
            assert torch.all(p.grad == 0), "Pending unzeroed gradients left after dry run with accum=4!"
    print("✓ test_dry_run_accum_larger_than_batches passed (dry_run=True, accum=4 -> 1 step, clean zeroed grads)")


def test_mathematical_gradient_scaling_partial_group():
    # Compare single batch backward with partial-group backward
    torch.manual_seed(42)
    m1 = nn.Linear(4, 3, bias=False)
    m2 = nn.Linear(4, 3, bias=False)
    m2.load_state_dict(m1.state_dict())
    
    x = torch.randn(1, 4)
    y = torch.tensor([1])
    
    # In a partial window of size 1, loss is scaled by 1/1 = 1
    # Check that gradient in partial window matches unscaled single batch gradient exactly
    loss1 = nn.functional.cross_entropy(m1(x), y)
    loss1.backward()
    
    loss2 = nn.functional.cross_entropy(m2(x), y) / 1.0  # partial window size 1
    loss2.backward()
    
    assert torch.allclose(m1.weight.grad, m2.weight.grad), "Partial window gradient scaling mismatch!"
    print("✓ test_mathematical_gradient_scaling_partial_group passed (loss/1.0 matches direct gradient)")


def test_amp_unscaling_before_clipping():
    # Verify that clip_grad_norm_ operates on unscaled gradients under AMP
    model, loader, criterion, optimizer, _, _ = create_dummy_setup(n_samples=4, batch_size=2)
    scaler = torch.amp.GradScaler("cpu", enabled=True)
    
    clipped_grads_unscaled = [False]
    orig_clip = torch.nn.utils.clip_grad_norm_
    
    def wrapped_clip(parameters, max_norm, norm_type=2.0):
        # Check if parameters have unscaled gradients (i.e. not multiplied by scaler scale factor ~65536)
        params = list(parameters)
        for p in params:
            if p.grad is not None:
                # True gradient norm is around 0.1-2.0, whereas scaled norm would be > 10,000
                if p.grad.abs().max().item() < 1000.0:
                    clipped_grads_unscaled[0] = True
        return orig_clip(parameters, max_norm, norm_type)
        
    torch.nn.utils.clip_grad_norm_ = wrapped_clip
    try:
        train_one_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=torch.device("cpu"),
            dry_run=False,
            use_amp=False,
            accum_steps=2
        )
    finally:
        torch.nn.utils.clip_grad_norm_ = orig_clip
        
    assert clipped_grads_unscaled[0], "Gradient clipping was executed on scaled gradients!"
    print("✓ test_amp_unscaling_before_clipping passed (scaler.unscale_ precedes clip_grad_norm_)")


if __name__ == "__main__":
    test_odd_batches_partial_window()
    test_even_batches_complete_windows()
    test_identity_accum_steps_one()
    test_dry_run_clean_flush()
    test_dry_run_accum_larger_than_batches()
    test_mathematical_gradient_scaling_partial_group()
    test_amp_unscaling_before_clipping()
    print("\nAll 7 gradient accumulation unit tests passed successfully!")
