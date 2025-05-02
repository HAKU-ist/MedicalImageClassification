
# Tumor Image Classification with VGG-16

This project implements a deep learning pipeline for binary classification of medical images to determine the presence or absence of a tumor. The model is based on transfer learning using the VGG-16 convolutional neural network pretrained on ImageNet.

## Dataset Structure

The dataset is divided into training, validation, and test sets. Each contains two folders:

```
dataset/
├── train/
│   ├── 0/  # No Tumor
│   └── 1/  # Tumor
├── val/
│   ├── 0/
│   └── 1/
└── test/
    ├── 0/
    └── 1/
```

## How to Run

1. Install the required packages:

```bash
pip install -r requirements.txt
```

2. Place the dataset in the `dataset/` directory.

3. Run the training and evaluation script:

```bash
python code/train_vgg16.py
```

If a pretrained model (`best_model_vgg16.pth`) exists, it will skip training and go straight to evaluation.

## Model Performance

- **Test Accuracy:** 96.30%
- **Tumor Recall:** 95.4%
- **Tumor F1-score:** 96.3%
- Confusion matrix and classification report are automatically saved and printed.

## Visualization

TensorBoard logs are stored in the `runs/` directory. Launch TensorBoard with:

```bash
tensorboard --logdir=runs/
```

Open your browser and go to `http://localhost:6006`.

## Model File

Due to GitHub's 100MB file limit, the trained model is not included in the repository.


## License

This project is licensed under the MIT License.
