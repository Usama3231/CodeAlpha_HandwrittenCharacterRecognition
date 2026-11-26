from pathlib import Path
import argparse
import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from safe_datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SEED = 42


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, images):
        return self.network(images)


def load_mnist():
    dataset = fetch_openml(name="mnist_784", version=1, as_frame=False, parser="auto")
    images = dataset.data.astype(np.float32)
    labels = dataset.target.astype(np.int64)
    raw_images = images.copy()
    return images / 255.0, labels, raw_images, dataset


def make_loader(images, labels, batch_size, shuffle):
    tensor_images = torch.from_numpy(images).reshape(-1, 1, 28, 28)
    tensor_labels = torch.from_numpy(labels)
    return DataLoader(
        TensorDataset(tensor_images, tensor_labels),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.set_grad_enabled(training):
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * len(batch_labels)
            correct += int((outputs.argmax(dim=1) == batch_labels).sum().item())
            count += len(batch_labels)
    return total_loss / count, correct / count


def save_confusion_matrix(labels, predictions):
    matrix = confusion_matrix(labels, predictions, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("MNIST CNN confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for row in range(10):
        for column in range(10):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def save_sample(raw_image, label):
    sample = np.rint(raw_image).astype(int).reshape(28, 28).tolist()
    with open(DATA_DIR / "sample_input.json", "w", encoding="utf-8") as file:
        json.dump(sample, file, indent=2)
    return label


def main():
    parser = argparse.ArgumentParser(description="Train a CNN on the MNIST digits dataset.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Optional training subset size")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    device = torch.device("cpu")

    images, labels, raw_images, dataset = load_mnist()
    x_train, y_train = images[:60000], labels[:60000]
    x_test, y_test = images[60000:], labels[60000:]
    if args.limit:
        x_train, y_train = x_train[: args.limit], y_train[: args.limit]

    train_loader = make_loader(x_train, y_train, args.batch_size, True)
    test_loader = make_loader(x_test, y_test, args.batch_size, False)

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_accuracy = 0.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, optimizer, device, True
        )
        _, test_accuracy = run_epoch(model, test_loader, criterion, optimizer, device, False)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_accuracy, 4),
                "test_accuracy": round(test_accuracy, 4),
            }
        )
        if test_accuracy >= best_accuracy:
            best_accuracy = test_accuracy
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"loss={train_loss:.4f} train_acc={train_accuracy:.4f} test_acc={test_accuracy:.4f}"
        )

    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_images, _ in test_loader:
            outputs = model(batch_images.to(device))
            predictions.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
    predictions = np.asarray(predictions, dtype=np.int64)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(
            float(precision_score(y_test, predictions, average="macro", zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_test, predictions, average="macro", zero_division=0)), 4
        ),
        "f1_score": round(
            float(f1_score(y_test, predictions, average="macro", zero_division=0)), 4
        ),
    }

    torch.save(model.state_dict(), MODELS_DIR / "mnist_cnn.pt")
    save_confusion_matrix(y_test, predictions)
    sample_label = save_sample(raw_images[60000], y_test[0])

    summary = {
        "dataset": {
            "name": "MNIST handwritten digits (OpenML mnist_784)",
            "shape": list(dataset.data.shape),
            "train_size": len(y_train),
            "test_size": len(y_test),
            "classes": list(range(10)),
        },
        "model": {
            "name": "Convolutional Neural Network",
            "architecture": "Conv2d(32) - Pool - Conv2d(64) - Pool - Dense(128) - Output(10)",
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        "metrics": metrics,
        "sample": {"true_label": int(sample_label)},
    }

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    with open(RESULTS_DIR / "training_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print(f"Best test accuracy: {best_accuracy:.4f}")
    print(f"Final metrics: {metrics}")
    print(f"Saved model: {MODELS_DIR / 'mnist_cnn.pt'}")
    print(f"Saved metrics: {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
