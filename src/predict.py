from pathlib import Path
import argparse
import json

import numpy as np
import torch

from train_model import DigitCNN

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mnist_cnn.pt"


def load_image(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    image = np.asarray(data, dtype=np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image.reshape(1, 1, 28, 28)


def main():
    parser = argparse.ArgumentParser(description="Predict a handwritten digit from a 28x28 image.")
    parser.add_argument("json_file", help="Path to a JSON file containing a 28x28 image")
    args = parser.parse_args()

    model = DigitCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    image = torch.from_numpy(load_image(args.json_file))
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)[0].numpy()
    prediction = int(probabilities.argmax())

    print(f"Prediction: {prediction}")
    print(f"Confidence: {probabilities[prediction]:.4f}")
    print("Probabilities: " + ", ".join(f"{index}:{value:.4f}" for index, value in enumerate(probabilities)))


if __name__ == "__main__":
    main()
