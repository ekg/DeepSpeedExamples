import os
import torch
import torch.nn as nn
import deepspeed
import argparse
from schedulefree import AdamWScheduleFree

def get_model():
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    # Use torch.compile to accelerate the model
    if hasattr(torch, 'compile') and callable(torch.compile):
        model = torch.compile(model)
        print("Model compiled with torch.compile")
    return model

def get_args():
    parser = argparse.ArgumentParser(
        description='DeepSpeed TP + ScheduleFree AdamW'
    )
    # Allow DeepSpeed's launcher to set the local rank
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    # Add DeepSpeed's built‑in arguments (e.g., --deepspeed_config)
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--tp_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--train_steps', type=int, default=10,
                        help='number of training steps')
    parser.add_argument('--port', type=int, default=29500,
                        help='port for distributed communication')
    return parser.parse_args()

def main():
    args = get_args()
    
    # Set custom port for distributed communication
    if args.port != 29500:
        os.environ['MASTER_PORT'] = str(args.port)
    
    # Instantiate model
    model = get_model()
    
    # 1) Instantiate Schedule‑Free AdamW (no external scheduler needed)
    base_opt = AdamWScheduleFree(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
        warmup_steps=0
    )

    # 2) DeepSpeed config for tensor parallelism
    ds_config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 1,
        "tensor_parallel": {
            "tp": {
                "tp_size": args.tp_size,
                "tp_grain_size": 64
            }
        }
    }

    # 3) Initialize DeepSpeed engine with our Schedule‑Free optimizer
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        optimizer=base_opt,
        config_params=ds_config
    )

    # 4) Put optimizer into train mode (required by Schedule‑Free)
    model_engine.optimizer.train()

    # 5) Training loop
    for step in range(args.train_steps):
        # Create dummy input and target
        x = torch.randn(4, 128, device=model_engine.device)
        y = torch.randint(0, 10, (4,), device=model_engine.device)

        # Forward pass
        logits = model_engine(x)
        
        # Compute loss
        loss = nn.functional.cross_entropy(logits, y)

        # Backward pass (handles gradient accumulation, communication, etc.)
        model_engine.backward(loss)

        # Optimizer step (updates weights, zeroes grads)
        model_engine.step()

        if model_engine.local_rank == 0:
            print(f"[Step {step:03d}] Loss = {loss.item():.4f}")
    
    # 6) Switch to eval mode before validation or checkpoint
    model_engine.optimizer.eval()

if __name__ == "__main__":
    main()
