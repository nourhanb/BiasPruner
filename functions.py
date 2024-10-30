import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torch.nn as nn
from collections import Counter

def get_S_bc_S_ba2(train_loader, model, device):

    S_bc = set()
    S_ba = set()
    unknown = set()

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            img_idx, images, labels, bias = data
            # counts = Counter(img_idx)
            # duplicates = {element: count for element, count in counts.items() if count > 1}
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            outputs = torch.sigmoid(logits)
            predicted_labels = outputs.argmax(dim=1)

            for idx, label in enumerate(labels):
                sample_index = img_idx[idx]
                if predicted_labels[idx] == label.item():
                    S_ba.add(sample_index)
                elif predicted_labels[idx] != label.item():
                    S_bc.add(sample_index)
                else:
                    unknown.add(sample_index)

    return S_bc, S_ba


def get_conv2d_output(model, S, train_loader, device):
    # Initialize a dictionary to store the sum of filter activations for each layer
    filter_activations_sum = {}

    # Define the hook function to collect filter activations for each layer
    def getActivation(layer_name):
        def hook(self, input, output):
            nonlocal filter_activations_sum
            if layer_name in filter_activations_sum:
                filter_activations_sum[layer_name] += output.mean(dim=(2, 3)).detach().squeeze()
            else:
                filter_activations_sum[layer_name] = output.mean(dim=(2, 3)).detach().squeeze()
        return hook

    hooks = []  # List to store hooks for later removal

    for i in range(3):
        block = eval('model.layer%d' % (i + 1))
        for j in range(9):
            layer_name = f"layer{i + 1}_block{j}_relu1"
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(getActivation(layer_name))
            hooks.append(handler)  # Store the hook for later removal

    for sample_idx in S:
        # Load an image and label from the dataset
        _, image, label, bias = train_loader.dataset[sample_idx]
        image = image.to(device)
        image = image.unsqueeze(0)  # Add batch dimension
        # Forward pass through the model
        with torch.no_grad():
            output = model(image)

    # Calculate the average filter activations for each layer
    average_filter_activations = {layer: filter_activations_sum[layer] / len(S) for layer in filter_activations_sum}

    # Remove all the hooks after the forward pass is completed
    for hook in hooks:
        hook.remove()

    return average_filter_activations


class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, self.dataset[idx])


class GeneralizedCELoss(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        p = F.softmax(logits, dim=1)
        # p = logits
        if torch.isnan(p.mean()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        loss_weight = (Yg.squeeze().detach()**self.q) * self.q
        if torch.isnan(Yg.mean()):
            raise NameError('GCE_Yg')

        loss = F.cross_entropy(logits, targets, reduction='none') * loss_weight
        # Aggregate the per-sample losses to obtain a scalar loss
        loss = torch.mean(loss)

        return loss
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(y_label, y_predict):
    correct_predictions = sum(1 for true, pred in zip(y_label, y_predict) if true == pred)
    total_samples = len(y_label)
    accuracy = correct_predictions / total_samples
    return accuracy


class WCE(nn.Module):
    def __init__(self, alpha, beta):
        super(WCE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, targets, image_idx, image_loss_list):
        ce_loss = self.cross_entropy_loss(predictions, targets)

        # Extract GCE values for the corresponding samples
        gce_values = torch.tensor([image_loss_list[idx][1] for idx in image_idx], device=predictions.device)

        # Calculate individualized weights
        weights =  (torch.exp(self.alpha *gce_values)) #- self.beta
        # weights = self.alpha * (gce_values) #- self.beta


        # Apply the individualized weights to the cross entropy loss
        weighted_loss = weights * ce_loss

        return torch.mean(weighted_loss)
