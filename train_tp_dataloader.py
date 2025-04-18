import os, random, numpy as np
import torch, torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import deepspeed

# 1) Deterministic seeding
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

class RandomDataset(Dataset):
    """Generates random (input, target) pairs deterministically."""
    def __init__(self, length, in_dim, out_dim):
        self.length, self.in_dim, self.out_dim = length, in_dim, out_dim

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # each idx yields same random sample across workers
        g = torch.Generator().manual_seed(SEED + idx)
        x = torch.randn(self.in_dim, generator=g)
        y = torch.randint(0, self.out_dim, (), generator=g)
        return x, y

def get_args():
    parser = deepspeed.add_config_arguments(
        __import__('argparse').ArgumentParser()
    )
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--train_steps', type=int, default=100)
    parser.add_argument('--tp_size',     type=int, default=1)
    return parser.parse_args()

def main():
    args = get_args()

    # 2) Build DeepSpeed engine
    model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    ds_config = {
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 1,
        "tensor_parallel": { "tp": { "tp_size": args.tp_size } }
    }
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, optimizer=optimizer,
        config_params=ds_config
    )

    # 3) Prepare Dataset & Sampler
    dataset = RandomDataset(length=10000, in_dim=128, out_dim=10)
    sampler = DistributedSampler(
        dataset,
        num_replicas=model_engine.world_size(),
        rank=model_engine.global_rank,
        shuffle=True,
        seed=SEED
    )

    # 4) DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=4,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=lambda wid: torch.manual_seed(SEED + wid)
    )

    # 5) Training loop
    for step in range(args.train_steps):
        epoch = step // len(data_loader)
        sampler.set_epoch(epoch)  # Ensures deterministic shuffling per epoch

        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.to(model_engine.device), y.to(model_engine.device)
            logits = model_engine(x)
            loss = nn.functional.cross_entropy(logits, y)

            model_engine.backward(loss)
            model_engine.step()

            if model_engine.global_rank == 0 and batch_idx == 0:
                print(f"[Step {step:03d}] Loss: {loss.item():.4f}")
            
            # Break after one batch per step for this example
            break

if __name__ == "__main__":
    main()
