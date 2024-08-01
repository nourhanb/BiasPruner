# BiasPruner
This is the PyTorch implementation of our early-accept MICCAI 2024 paper: *BiasPruner: Debiased Continual Learning for Medical Image Classification*

## Abstract 
Continual Learning (CL) is crucial for enabling networks to dynamically adapt as they learn new tasks sequentially, accommodating new data and classes without catastrophic forgetting. Diverging from conventional perspectives on CL, our paper introduces a new perspective wherein forgetting could actually benefit the sequential learning paradigm. Specifically, we present BiasPruner, a CL framework that intentionally forgets spurious correlations in the training data that could lead to shortcut learning. Utilizing a new bias score that measures the contribution of each unit in the network to learning spurious features, BiasPruner prunes those units with the highest bias scores to form a debiased subnetwork preserved for a given task. As BiasPruner learns a new task, it constructs a new debiased subnetwork, potentially incorporating units from previous subnetworks, which improves adaptation and performance on the new task. During inference, BiasPruner employs a simple task-agnostic approach to select the best debiased subnetwork for predictions. We conduct experiments on three medical datasets for skin lesion classification and chest X-Ray classification and demonstrate that BiasPruner consistently outperforms SOTA CL methods in terms of classification performance and fairness. 

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
