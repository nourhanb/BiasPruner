# BiasPruner
This is the PyTorch implementation of our **early-accept** MICCAI 2024 paper: 

[BiasPruner: Debiased Continual Learning for Medical Image Classification](https://arxiv.org/pdf/2407.08609)

## Abstract 
Continual Learning (CL) is crucial for enabling networks to dynamically adapt as they learn new tasks sequentially, accommodating new data and classes without catastrophic forgetting. Diverging from conventional perspectives on CL, our paper introduces a new perspective wherein forgetting could actually benefit the sequential learning paradigm. Specifically, we present BiasPruner, a CL framework that intentionally forgets spurious correlations in the training data that could lead to shortcut learning. Utilizing a new bias score that measures the contribution of each unit in the network to learning spurious features, BiasPruner prunes those units with the highest bias scores to form a debiased subnetwork preserved for a given task. As BiasPruner learns a new task, it constructs a new debiased subnetwork, potentially incorporating units from previous subnetworks, which improves adaptation and performance on the new task. During inference, BiasPruner employs a simple task-agnostic approach to select the best debiased subnetwork for predictions. We conduct experiments on three medical datasets for skin lesion classification and chest X-Ray classification and demonstrate that BiasPruner consistently outperforms SOTA CL methods in terms of classification performance and fairness. 

<p align="center">
  <img src="overview.png" alt="alt text">
</p>

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

## How to Run

This script can perform model training, pruning, and fine-tuning using command-line arguments. Below are examples of how to run the script with different configurations.

### Train the Model

To train the model with a specified learning rate and number of epochs:

```bash
python main.py --train --learning_rate 0.001 --train_epochs 50 --weights_original /path/to/original_weights.pth
```
### Prune the Model
```bash
python main.py --prune --pruning_ratio 0.5 --weights_original /path/to/original_weights.pth
```

### Fine-tune the Model
```bash
python main.py --finetune --learning_rate 0.0005 --finetune_epochs 30 --weights_finetuned /path/to/finetuned_weights.pth

```

## Citation 
If you use this code in your research, please consider citing:

```text
@article{bayasi2024biaspruner,
  title={BiasPruner: Debiased Continual Learning for Medical Image Classification},
  author={Bayasi, Nourhan and Fayyad, Jamil and Bissoto, Alceu and Hamarneh, Ghassan and Garbi, Rafeef},
  journal={International Conference on Medical Image Computing and Computer-Assisted Intervention ({MICCAI})},
  year={2024}
}
```

