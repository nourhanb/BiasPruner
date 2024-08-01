# BiasPruner
This is the PyTorch implementation of our early-accept MICCAI 2024 paper:

*BiasPruner: Debiased Continual Learning for Medical Image Classification*

## Datasets

The script supports the following datasets:

- **Fitzpatrick**: A dataset with skin lesion images.
- **ham**: The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.
- **NIH**: The NIH Chest X-ray dataset, which includes X-ray images and associated pathology labels.

## Prerequisites

Before running the script, ensure that you have the following software and libraries installed:

- Python 3.6+
- PyTorch
- torchvision
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm

## Example
Here is an example of how to run the script with specific arguments:
`python main.py --dataset ham --num_classes 3 --learning_rate 0.01 --num_epochs 50 --patience 10`
