from typing import Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from IPython.display import clear_output
from pytorch_lightning import seed_everything


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    score_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    output_fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    seed: int = 42,
) -> pd.DataFrame:
    seed_everything(seed)
    metrics = []

    for epoch in tqdm(range(epochs)):
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                output = model(batch_X)
                val_losses.append(criterion(output, batch_y).item())
                val_preds.append(output_fn(output).cpu().numpy())
                val_labels.append(batch_y.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)

        row = {
            'epoch': epoch + 1,
            'train_loss': np.mean(train_losses),
            'val_loss': np.mean(val_losses),
        }
        for name, fn in score_fns.items():
            row[name] = fn(val_labels, val_preds)
        metrics.append(row)

        if epoch > 0:
            _plot_metrics(pd.DataFrame(metrics), score_fns)

    return pd.DataFrame(metrics)


def _plot_metrics(
    df: pd.DataFrame,
    score_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]],
) -> None:
    clear_output(wait=True)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.plot(df['epoch'], df['train_loss'], 'b-', label='train_loss')
    ax1.plot(df['epoch'], df['val_loss'], 'b--', label='val_loss')

    ax2 = ax1.twinx()
    ax2.set_ylabel('score')
    cmap = plt.get_cmap('Reds')
    colors = cmap(np.linspace(0.4, 0.8, len(score_fns)))
    for name, color in zip(score_fns.keys(), colors):
        ax2.plot(df['epoch'], df[name], color=color, label=name)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax1.set_xticks(df['epoch'])

    plt.tight_layout()
    plt.show()
