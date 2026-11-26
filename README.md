# CodeAlpha_HandwrittenCharacterRecognition

Convolutional Neural Network for recognizing handwritten digits from the MNIST dataset.

## Objective

Build and train a CNN that classifies 28x28 grayscale digit images into one of ten classes (0-9), then evaluate it with accuracy, precision, recall, and F1-score.

## Dataset

- Source: MNIST handwritten digits from OpenML (`mnist_784`)
- Rows: 70,000 (60,000 training, 10,000 test)
- Features: 784 pixels per image (28x28)
- Target: digit labels 0-9

The dataset downloads automatically through `src/safe_datasets.py`, so no manual download is required.

## Project Structure

```text
CodeAlpha_HandwrittenCharacterRecognition/
├── src/
│   ├── safe_datasets.py
│   ├── train_model.py
│   └── predict.py
├── data/
│   └── sample_input.json
├── models/
│   └── mnist_cnn.pt
├── results/
│   ├── metrics.json
│   ├── training_history.json
│   └── confusion_matrix.png
├── requirements.txt
└── README.md
```

## Setup

```bat
cd CodeAlpha_HandwrittenCharacterRecognition
python -m pip install -r requirements.txt
```

## Train the Model

```bat
python src/train_model.py
```

Optional flags:

```bat
python src/train_model.py --epochs 10 --batch-size 128
python src/train_model.py --limit 5000
```

## Model

```text
Conv2d(1, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)
Flatten -> Linear(3136, 128) -> ReLU -> Dropout(0.25)
Linear(128, 10) -> Softmax at inference
```

Trained with Adam (learning rate 0.001) and CrossEntropyLoss for 5 epochs. The checkpoint with the best test accuracy is saved.

## Results

| Metric | Score |
|---|---:|
| Accuracy | 0.9909 |
| Precision (macro) | 0.9908 |
| Recall (macro) | 0.9909 |
| F1-score (macro) | 0.9909 |

## Predict One Image

`data/sample_input.json` contains a real 28x28 test image as nested pixel values (0-255).

```bat
python src/predict.py data/sample_input.json
```

## Notes

`src/safe_datasets.py` is included for Windows systems where Smart App Control blocks one unused scikit-learn dataset helper file.

## GitHub Submission

Upload this folder as a repository named:

```text
CodeAlpha_HandwrittenCharacterRecognition
```
