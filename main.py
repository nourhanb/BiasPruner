import argparse
import torch
import numpy as np
import random
import os
import pickle
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, f1_score, recall_score
import param
from attributes import get_bias

from functions import get_S_bc_S_ba2, get_conv2d_output, GeneralizedCELoss, WCE
from get_dataloaders import loaders
from models import *
from trainer import train_model, finetune_model
from test import test
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import demographic_parity_ratio, demographic_parity_difference
from my_fairlearn import *


def main(args):
    # 1) Define the seeds and device
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2) Define dataloaders
    train_loader, valid_loader, test_loader, samples_loader = loaders(param.NIH_base_meta, param.NIH_root, param.NIH_transform)

    # 3) Define model
    cfg = [16, 16, 16, 16, 16, 16, 16, 16, 16,
           32, 32, 32, 32, 32, 32, 32, 32, 32,
           64, 64, 64, 64, 64, 64, 64, 64, 64]

    model = resnet56_cifar(cfg=cfg)
    model = model.to(device)

    if args.train:
        # 4) Training loss and optimizer
        criterion = GeneralizedCELoss(q=0.7).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999))

        # 5) Training loop
        epoch_num = args.train_epochs
        image_loss_list = train_model(model, train_loader, valid_loader, criterion, optimizer, epoch_num, device,
                                      save_weights=True, original_model=True)

        with open('image_loss_list_nih.pkl', 'wb') as f:
            pickle.dump(image_loss_list, f)

        weightpath_original = args.weights_original
    else:
        print("Weights of Original model are loaded without training")
        weightpath_original = args.weights_original
        model.load_state_dict(torch.load(weightpath_original))
        model.to(device)

    if not args.train:
        with open('image_loss_list_nih.pkl', 'rb') as f:
            image_loss_list = pickle.load(f)

    # 6) Test model
    all_labels, all_preds, all_sensitive_attributes = test(model, test_loader, device, weightpath_original)

    if args.prune:
        # 7) Pruning
        # a) Get S_ba and S_bc
        S_bc, S_ba = get_S_bc_S_ba2(samples_loader, model, device)

        get_bias(S_ba, S_bc, samples_loader)

        # b) Get bias score
        bias_alligned = get_conv2d_output(model, S_ba, samples_loader, device)
        bias_conflicting = get_conv2d_output(model, S_bc, samples_loader, device)

        bias_score = {}

        if bias_alligned and bias_conflicting:  # Check if both dictionaries are not empty
            bias_score = {key: bias_alligned[key] - bias_conflicting[key] for key in bias_alligned}
        elif bias_alligned:  # If only bias_conflicting is empty
            bias_score = bias_alligned.copy()
        elif bias_conflicting:  # If only bias_alligned is empty
            bias_score = {key: -bias_conflicting[key] for key in bias_conflicting}

        # c) Calculate Threshold
        total_channel = 0
        index = 0
        for n in list(bias_score.values()):
            total_channel = total_channel + n.shape[0]
        print('total_channel:', total_channel)

        feature_s = torch.zeros(total_channel)
        for n in list(bias_score.values()):
            size = n.shape[0]
            feature_s[index:(index + size)] = n
            index = index + size

        y, i = torch.sort(feature_s, descending=True)
        thre_index = int(total_channel * args.pruning_ratio)
        thre = y[thre_index]

        # d) Prune
        pruned = 0
        cfg1 = []
        cfg_mask = []
        # i = 0
        for i in range(27):
            print('i=', i)
            feature_copy = list(bias_score.values())[i]
            mask = feature_copy.gt(thre).float()  # .cuda()    [1 1 1 0 0]   [3]
            if torch.sum(mask) == 0:
                cfg1.append(len(feature_copy))
                cfg_mask.append(torch.ones(len(feature_copy)).float())  # .cuda())
                print('total channel: {:d} \t remaining channel: {:d}'.
                      format(len(feature_copy), int(len(feature_copy))))
            else:
                pruned = pruned + mask.shape[0] - torch.sum(mask)
                cfg1.append(int(torch.sum(mask)))
                cfg_mask.append(mask.clone())
                print('total channel: {:d} \t remaining channel: {:d}'.
                      format(mask.shape[0], int(torch.sum(mask))))

        cfg_mask1 = []
        j = 0
        out = torch.ones(16).float()  # .cuda()#1
        cfg_mask1.append(out)

        print(cfg_mask[j])
        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(16).float())  # .cuda())#3

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(16).float().cuda())  # 5

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(16).float().cuda())  # 7

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(16).float().cuda())  # 9

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(16).float().cuda())  # 11

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(16).float().cuda())  # 13

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(16).float().cuda())  # 15

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(16).float().cuda())  # 17

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(16).float().cuda())  # 19

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(32).float().cuda())  # 21
        cfg_mask1.append(22)  # 22

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(32).float().cuda())  # 24

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(32).float().cuda())  # 26

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(32).float().cuda())  # 28

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(32).float().cuda())  # 30

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(32).float().cuda())  # 32

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(32).float().cuda())  # 34

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(32).float().cuda())  # 36

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(32).float().cuda())  # 38

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(64).float().cuda())  # 40
        cfg_mask1.append(41)  # 41

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(64).float().cuda())  # 43

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(64).float().cuda())  # 45

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(64).float().cuda())  # 47

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(64).float().cuda())  # 49

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(64).float().cuda())  # 51

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(64).float().cuda())  # 53

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(64).float().cuda())  # 55

        cfg_mask1.append(cfg_mask[j])
        j = j + 1
        cfg_mask1.append(torch.ones(64).float().cuda())  # 57

        pruned_ratio = pruned / total_channel
        print('pruned_ratio=', pruned_ratio)

        print('Pre-processing Successful!')

        # e) Save pruned model
        newmodel = resnet56_cifar(cfg=cfg1)
        newmodel.to(device)

        num_parameters = sum([param.nelement() for param in newmodel.parameters()])
        savepath = os.path.join(param.pruned_path, "prune3_task1.txt")
        with open(savepath, "w") as fp:
            fp.write("Configuration: \n" + str(cfg1) + "\n")
            fp.write("Number of parameters: \n" + str(num_parameters) + "\n")
            fp.write("pruned: \n" + str(pruned) + "\n")
            fp.write("pruned_ratio: \n" + str(pruned_ratio) + "\n")

        layer_id_in_cfg = 0
        start_mask = torch.ones(3)
        end_mask = cfg_mask1[layer_id_in_cfg]
        t = 1
        down = [22, 41]

        for [m0, m1] in zip(model.modules(), newmodel.modules()):
            if t in down:
                if isinstance(m0, nn.Conv2d):
                    m1.weight.data = m0.weight.data
                elif isinstance(m0, nn.BatchNorm2d):
                    m1.weight.data = m0.weight.data
                    m1.bias.data = m0.bias.data
                    m1.running_mean = m0.running_mean
                    m1.running_var = m0.running_var
                    layer_id_in_cfg += 1
                    t += 1
                    if layer_id_in_cfg < len(cfg_mask1):
                        end_mask = cfg_mask1[layer_id_in_cfg]
            elif isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                t += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask1):  # do not change in Final FC
                    end_mask = cfg_mask1[layer_id_in_cfg]
            elif isinstance(m0, nn.Conv2d):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()

        torch.save({'cfg': cfg1, 'state_dict': newmodel.state_dict()}, os.path.join(param.pruned_path, 'pruned0.1_50.pth.tar'))

    print("Pruning weights are loaded")
    prune_path = os.path.join(param.pruned_path, 'pruned0.1_50.pth.tar')
    checkpoint = torch.load(prune_path)
    pruned_model = resnet56_cifar(cfg=checkpoint['cfg'])
    pruned_model.load_state_dict(checkpoint['state_dict'])
    pruned_model.to(device)

    all_labels, all_preds, all_sensitive_attributes = test(pruned_model, test_loader, device)

    if args.finetune:
        print('Fine-Tuning now!')
        tune_criterion = nn.CrossEntropyLoss()
        finetune_optimizer = torch.optim.Adam(pruned_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.999))
        finetune_epochs = args.finetune_epochs

        finetune_model(pruned_model, train_loader, valid_loader, tune_criterion, finetune_optimizer, finetune_epochs,
                       device, image_loss_list=None, save_weights=True)

    wp = args.weights_finetuned
    all_labels, all_preds, all_sensitive_attributes = test(pruned_model, test_loader, device, wp)

    # Overall Accuracy
    overall_accuracy = accuracy_score(all_labels, all_preds)
    print(f'Overall Accuracy: {overall_accuracy:.4f}')

    # Evaluation on the test set
    print('\nTest Classification Report:')
    print(classification_report(all_labels, all_preds))

    dp_ratio = demographic_parity_ratio_mine(all_labels, all_preds, sensitive_features=all_sensitive_attributes)
    print(f'Demographic Parity ratio: {dp_ratio:.4f}')

    mf_acc = MetricFrame(metrics=accuracy_score, y_true=all_labels, y_pred=all_preds, sensitive_features=all_sensitive_attributes)
    print('Performance (accuracy) overall', mf_acc.overall)
    print('Performance (accuracy) by group', mf_acc.by_group)

    mf_balanced = MetricFrame(metrics=balanced_accuracy_score, y_true=all_labels, y_pred=all_preds, sensitive_features=all_sensitive_attributes)
    print('Performance (balanced accuracy) overall', mf_balanced.overall)
    print('Performance (balanced accuracy) by group', mf_balanced.by_group)

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_sensitive_attributes = np.array(all_sensitive_attributes)

    # Calculate Recall
    recall = recall_score(all_labels, all_preds, average='weighted')

    # Calculate F1 Score
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print("Recall:", recall)
    print("F1 Score:", f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training and Pruning')

    parser.add_argument('--train', action='store_true', help='Flag to indicate training mode')
    parser.add_argument('--prune', action='store_true', help='Flag to indicate pruning mode')
    parser.add_argument('--finetune', action='store_true', help='Flag to indicate finetuning mode')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--train_epochs', type=int, default=param.train_epochs, help='Number of training epochs')
    parser.add_argument('--finetune_epochs', type=int, default=param.finetune_epochs, help='Number of finetuning epochs')
    parser.add_argument('--pruning_ratio', type=float, default=0.1, help='Pruning ratio for model pruning')
    parser.add_argument('--weights_original', type=str, default='/home/jfayyad/PycharmProjects/BiasWashV2/weights/original_ham_task1.pth',
                        help='Path to the original model weights')
    parser.add_argument('--weights_finetuned', type=str, default=param.weights_finetuned,
                        help='Path to the finetuned model weights')

    args = parser.parse_args()
    main(args)
