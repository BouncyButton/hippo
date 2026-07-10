import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


@dataclass
class Config:
    p: int = 17
    train_fraction: float = 0.40
    hidden_dim: int = 128
    stage1_steps: int = 1500
    stage2_steps: int = 4000
    lr: float = 1e-3
    lambdas: tuple = (0.0, 1e-3, 1e-2, 1e-1)
    seeds: tuple = (0, 1, 2)
    log_every: int = 10
    device: str = "cpu"


class ModularMLP(nn.Module):
    def __init__(self, p: int, hidden_dim: int):
        super().__init__()
        self.p = p
        self.net = nn.Sequential(
            nn.Linear(2 * p, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, p),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_onehot = F.one_hot(a, num_classes=self.p).float()
        b_onehot = F.one_hot(b, num_classes=self.p).float()
        x = torch.cat([a_onehot, b_onehot], dim=-1)
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataset(p: int, train_fraction: float, seed: int, device: str):
    a, b = torch.meshgrid(
        torch.arange(p),
        torch.arange(p),
        indexing="ij",
    )
    a = a.reshape(-1)
    b = b.reshape(-1)
    y = (a + b) % p

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(p * p, generator=generator)

    n_train = int(train_fraction * p * p)
    train_idx = permutation[:n_train]
    test_idx = permutation[n_train:]

    data = {
        "all_a": a.to(device),
        "all_b": b.to(device),
        "all_y": y.to(device),
        "train_a": a[train_idx].to(device),
        "train_b": b[train_idx].to(device),
        "train_y": y[train_idx].to(device),
        "test_a": a[test_idx].to(device),
        "test_b": b[test_idx].to(device),
        "test_y": y[test_idx].to(device),
    }
    return data


def cyclic_shift_probabilities(q: torch.Tensor, amount: int = 1) -> torch.Tensor:
    """
    If q[k] is the probability of class k, then the shifted distribution
    corresponds to adding 'amount' modulo p to the predicted class.
    """
    return torch.roll(q, shifts=amount, dims=-1)


def selection_constraint(
        model: nn.Module,
        p: int,
        device: str,
) -> torch.Tensor:
    """
    Enforces:
        f(a + 1, b) = f(a, b) + 1 mod p
        f(a, b + 1) = f(a, b) + 1 mod p

    The constraint is computed on the complete p x p input grid.
    """
    a, b = torch.meshgrid(
        torch.arange(p, device=device),
        torch.arange(p, device=device),
        indexing="ij",
    )
    a = a.reshape(-1)
    b = b.reshape(-1)

    q = F.softmax(model(a, b), dim=-1)
    q_a_shift = F.softmax(model((a + 1) % p, b), dim=-1)
    q_b_shift = F.softmax(model(a, (b + 1) % p), dim=-1)

    target_shift = cyclic_shift_probabilities(q, amount=1)

    loss_a = F.mse_loss(q_a_shift, target_shift)
    loss_b = F.mse_loss(q_b_shift, target_shift)
    return 0.5 * (loss_a + loss_b)


@torch.no_grad()
def accuracy(model: nn.Module, a: torch.Tensor, b: torch.Tensor, y: torch.Tensor) -> float:
    pred = model(a, b).argmax(dim=-1)
    return (pred == y).float().mean().item()


@torch.no_grad()
def functional_distance(
        model: nn.Module,
        reference_probabilities: torch.Tensor,
        all_a: torch.Tensor,
        all_b: torch.Tensor,
) -> float:
    q = F.softmax(model(all_a, all_b), dim=-1)
    return F.mse_loss(q, reference_probabilities).item()


@torch.no_grad()
def discrete_shift_violation(model: nn.Module, p: int, device: str) -> float:
    a, b = torch.meshgrid(
        torch.arange(p, device=device),
        torch.arange(p, device=device),
        indexing="ij",
    )
    a = a.reshape(-1)
    b = b.reshape(-1)

    pred = model(a, b).argmax(dim=-1)
    pred_a_shift = model((a + 1) % p, b).argmax(dim=-1)
    pred_b_shift = model(a, (b + 1) % p).argmax(dim=-1)

    expected = (pred + 1) % p
    violation_a = (pred_a_shift != expected).float().mean()
    violation_b = (pred_b_shift != expected).float().mean()
    return (0.5 * (violation_a + violation_b)).item()


def evaluate(
        model: nn.Module,
        data: Dict[str, torch.Tensor],
        p: int,
        device: str,
        reference_probabilities: torch.Tensor,
) -> Dict[str, float]:
    return {
        "train_loss": F.cross_entropy(model(data["train_a"], data["train_b"]), data["train_y"]).item(),
        "test_loss": F.cross_entropy(model(data["test_a"], data["test_b"]), data["test_y"]).item(),
        "train_acc": accuracy(model, data["train_a"], data["train_b"], data["train_y"]),
        "test_acc": accuracy(model, data["test_a"], data["test_b"], data["test_y"]),
        "constraint": selection_constraint(model, p, device).item(),
        "functional_distance": functional_distance(
            model,
            reference_probabilities,
            data["all_a"],
            data["all_b"],
        ),
        "shift_violation": discrete_shift_violation(model, p, device),
    }


def train_one_seed(config: Config, seed: int):
    set_seed(seed)
    data = make_dataset(config.p, config.train_fraction, seed, config.device)

    base_model = ModularMLP(config.p, config.hidden_dim).to(config.device)
    optimizer = torch.optim.Adam(base_model.parameters(), lr=config.lr)

    # Stage 1: supervised memorization only.
    for step in range(config.stage1_steps):
        optimizer.zero_grad()
        logits = base_model(data["train_a"], data["train_b"])
        supervised_loss = F.cross_entropy(logits, data["train_y"])
        supervised_loss.backward()
        optimizer.step()

    train_acc = accuracy(
        base_model,
        data["train_a"],
        data["train_b"],
        data["train_y"],
    )
    test_acc = accuracy(
        base_model,
        data["test_a"],
        data["test_b"],
        data["test_y"],
    )

    train_loss = F.cross_entropy(
        base_model(data["train_a"], data["train_b"]),
        data["train_y"],
    ).item()

    test_loss = F.cross_entropy(
        base_model(data["test_a"], data["test_b"]),
        data["test_y"],
    ).item()

    print(
        f"Seed {seed} after stage 1: "
        f"train_acc={train_acc:.3f}, test_acc={test_acc:.3f}"
        f"train_loss={train_loss:.3f}, test_loss={test_loss:.3f}"
    )

    with torch.no_grad():
        reference_probabilities = F.softmax(
            base_model(data["all_a"], data["all_b"]),
            dim=-1,
        ).detach()

    base_state = {
        name: tensor.detach().clone()
        for name, tensor in base_model.state_dict().items()
    }

    results = {}

    # Stage 2: branch from the same memorizing solution for each lambda.
    for constraint_weight in config.lambdas:
        model = ModularMLP(config.p, config.hidden_dim).to(config.device)
        model.load_state_dict(base_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        history = {
            "step": [],
            "train_acc": [],
            "test_acc": [],
            "constraint": [],
            "functional_distance": [],
            "shift_violation": [],
        }

        initial_metrics = evaluate(
            model,
            data,
            config.p,
            config.device,
            reference_probabilities,
        )
        for key in history:
            if key != "step":
                history[key].append(initial_metrics[key])
        history["step"].append(0)

        for step in range(1, config.stage2_steps + 1):
            optimizer.zero_grad()

            logits = model(data["train_a"], data["train_b"])
            supervised_loss = F.cross_entropy(logits, data["train_y"])
            constraint_loss = selection_constraint(model, config.p, config.device)

            total_loss = supervised_loss + constraint_weight * constraint_loss
            total_loss.backward()
            optimizer.step()

            if step % config.log_every == 0 or step == config.stage2_steps:
                metrics = evaluate(
                    model,
                    data,
                    config.p,
                    config.device,
                    reference_probabilities,
                )
                history["step"].append(step)
                for key in history:
                    if key != "step":
                        history[key].append(metrics[key])

        results[constraint_weight] = history

        print(
            f"  lambda={constraint_weight:g}: "
            f"train_acc={history['train_acc'][-1]:.3f}, "
            f"test_acc={history['test_acc'][-1]:.3f}, "
            f"constraint={history['constraint'][-1]:.6f}, "
            f"shift_violation={history['shift_violation'][-1]:.3f}"
        )

    return results


def aggregate_results(all_results: List[Dict[float, Dict[str, List[float]]]], lambdas):
    aggregated = {}

    for constraint_weight in lambdas:
        steps = np.array(all_results[0][constraint_weight]["step"])
        aggregated[constraint_weight] = {"step": steps}

        for metric in (
                "train_acc",
                "test_acc",
                "constraint",
                "functional_distance",
                "shift_violation",
        ):
            values = np.array([
                seed_result[constraint_weight][metric]
                for seed_result in all_results
            ])
            aggregated[constraint_weight][f"{metric}_mean"] = values.mean(axis=0)
            aggregated[constraint_weight][f"{metric}_std"] = values.std(axis=0)

    return aggregated


def plot_metric(aggregated, lambdas, metric: str, ylabel: str, output_path: str):
    plt.figure(figsize=(7, 4.5))

    for constraint_weight in lambdas:
        steps = aggregated[constraint_weight]["step"]
        mean = aggregated[constraint_weight][f"{metric}_mean"]
        std = aggregated[constraint_weight][f"{metric}_std"]

        label = rf"$\lambda={constraint_weight:g}$"
        plt.plot(steps, mean, label=label)
        plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Stage-2 gradient steps")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=17)
    parser.add_argument("--train-fraction", type=float, default=0.40)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--stage1-steps", type=int, default=1500)
    parser.add_argument("--stage2-steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 1e-3, 1e-2, 1e-1],
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="toy_constraint_grokking_results")
    args = parser.parse_args()

    config = Config(
        p=args.p,
        train_fraction=args.train_fraction,
        hidden_dim=args.hidden_dim,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        lr=args.lr,
        lambdas=tuple(args.lambdas),
        seeds=tuple(args.seeds),
        log_every=args.log_every,
        device=args.device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = [
        train_one_seed(config, seed)
        for seed in config.seeds
    ]

    aggregated = aggregate_results(all_results, config.lambdas)

    plot_metric(
        aggregated,
        config.lambdas,
        "train_acc",
        "Training accuracy",
        str(output_dir / "train_accuracy.png"),
    )
    plot_metric(
        aggregated,
        config.lambdas,
        "test_acc",
        "Test accuracy",
        str(output_dir / "test_accuracy.png"),
    )
    plot_metric(
        aggregated,
        config.lambdas,
        "constraint",
        "Selection-constraint loss",
        str(output_dir / "constraint_loss.png"),
    )
    plot_metric(
        aggregated,
        config.lambdas,
        "functional_distance",
        "Functional distance from stage-1 model",
        str(output_dir / "functional_distance.png"),
    )
    plot_metric(
        aggregated,
        config.lambdas,
        "shift_violation",
        "Discrete shift violation",
        str(output_dir / "shift_violation.png"),
    )

    print(f"\nSaved plots to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
