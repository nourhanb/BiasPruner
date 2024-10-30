import torch
from tqdm import tqdm

import param
from train import train, validate, tune, validate_tune

def train_model( model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                epoch_num,
                device,
                save_weights =True,
                original_model = True
                ):
    best_val_acc = 0
    total_loss_train, total_acc_train = [],[]
    total_loss_val, total_acc_val = [],[]
    for epoch in tqdm(range(1, epoch_num+1)):
        loss_train, acc_train, image_loss_list = train(train_loader, model, criterion, optimizer, epoch, device,total_loss_train, total_acc_train)
        total_loss_train.append(loss_train)
        total_acc_train.append(acc_train)
        loss_val, acc_val = validate(val_loader, model, criterion, optimizer, epoch, device)
        total_loss_val.append(loss_val)
        total_acc_val.append(acc_val)
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            print('*****************************************************')
            print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
            print('*****************************************************')
            if save_weights:
                if original_model:
                    print('Saved weights for Original model: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
                    torch.save(model.state_dict(), param.weights_original)
                else:
                    print('Saved weights for Pruned model: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
                    torch.save(model.state_dict(), param.weights_pruned)

    return image_loss_list


def finetune_model( model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                epoch_num,
                device,
                image_loss_list= None,
                save_weights =True,
                ):
    best_val_acc = 0
    total_loss_train, total_acc_train = [],[]
    total_loss_val, total_acc_val = [],[]
    for epoch in tqdm(range(1, epoch_num+1)):
        loss_train, acc_train = tune(train_loader, model, criterion, optimizer, epoch, device,total_loss_train, total_acc_train, image_loss_list)
        total_loss_train.append(loss_train)
        total_acc_train.append(acc_train)
        acc_val = validate_tune(val_loader, model, criterion, optimizer, epoch, device)
        # total_loss_val.append(loss_val)
        total_acc_val.append(acc_val)
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            print('*****************************************************')
            print('best record: [epoch %d], [val acc %.5f]' % (epoch, acc_val))
            print('*****************************************************')
            if save_weights:
                print('Saved weights for tuned model: [epoch %d], [val acc %.5f]' % (epoch, acc_val))
                torch.save(model.state_dict(), param.weights_finetuned)
