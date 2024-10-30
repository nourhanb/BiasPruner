import torch
import numpy as np
from torch.autograd import Variable
from functions import calculate_accuracy

def test(model, testloader, device, weightpath=None):
    if weightpath:
        model.load_state_dict(torch.load(weightpath))
    model.eval()
    all_labels = []
    all_pred = []
    all_sensitive_attributes =[]
    with torch.no_grad():
        for i, data in enumerate(testloader):
            _, images, labels, bias, code = data
            N = images.size(0)
            images = Variable(images).to(device)
            outputs = model(images)
            # outputs = torch.sigmoid(outputs)
            outputs = torch.softmax(outputs, dim=1)
            prediction = outputs.max(1, keepdim=True)[1]
            all_labels.extend(labels.cpu().numpy())
            all_pred.extend(np.squeeze(prediction.cpu().numpy().T))
            all_sensitive_attributes.extend(bias.numpy())

    accuracy = calculate_accuracy(all_labels, all_pred)
    print(f"Total Accuracy of network {accuracy:.2f}%")
    return all_labels, all_pred, all_sensitive_attributes
