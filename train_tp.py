import os
import torch
import torch.nn as nn
from torch.optim import AdamW
import deepspeed
import argparse

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        # A single linear layer
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x)

def get_args():
    parser = argparse.ArgumentParser()
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--tp_size', type=int, default=1,
                        help='Tensor parallel size')
    parser.add_argument('--train_steps', type=int, default=10,
                        help='Number of training steps')
    return parser.parse_args()

def main():
    args = get_args()

    # Instantiate model and move parameters under DeepSpeed control
    model = SimpleModel()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # DeepSpeed initialization: returns engine wrapping model & optimizer
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=optimizer,
        config_params={
            "train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "tensor_parallel": {"tp_size": args.tp_size}
        }
    )

    # Training loop over random data
    for step in range(args.train_steps):
        # Create dummy input and target
        x = torch.randn(1, 8).to(model_engine.device)
        target = torch.randn(1, 8).to(model_engine.device)

        # Forward pass
        output = model_engine(x)

        # Compute simple MSE loss
        loss = nn.functional.mse_loss(output, target)

        # Backward pass (handles gradient accumulation, communication, etc.)
        model_engine.backward(loss)

        # Optimizer step (updates weights, zeroes grads, updates LR scheduler)
        model_engine.step()

        if model_engine.local_rank == 0:
            print(f"[Step {step:2d}] loss = {loss.item():.6f}")

if __name__ == "__main__":
    main()
