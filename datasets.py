import torch
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from sklearn.model_selection import train_test_split
from DermoDatasets import load_data, get_fitz_dataloaders

def get_dataset(name):
    if name == "MNIST":

        transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
            ])

        trainset = MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
        ds_train, ds_val = train_test_split(trainset, test_size=0.15)
        ds_test = MNIST('~/.pytorch/MNIST_data/', download=True, train=False, transform=transform)


        trainloader = torch.utils.data.DataLoader(ds_train, batch_size=100, shuffle=True, num_workers=8)
        valloader = torch.utils.data.DataLoader(ds_val, batch_size=100, shuffle=True, num_workers=8)
        testloader = torch.utils.data.DataLoader(ds_test, batch_size=100, shuffle=True, num_workers=8)

    elif name == "CIFAR10":

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        trainset = CIFAR10('~/.pytorch/CIFAR_data/', download=True, train=True, transform=train_transform)
        ds_train, ds_val = train_test_split(trainset, test_size=0.15)
        ds_test = CIFAR10('~/.pytorch/MNIST_data/', download=True, train=False, transform=test_transform)


        trainloader = torch.utils.data.DataLoader(ds_train, batch_size=100, shuffle=True, num_workers=8)
        valloader = torch.utils.data.DataLoader(ds_val, batch_size=100, shuffle=True, num_workers=8)
        testloader = torch.utils.data.DataLoader(ds_test, batch_size=100, shuffle=True, num_workers=8)


    elif name == "HAM":

        meta_dir = '/ubc/ece/home/ra/grads/nourhanb/Documents/miccai 2024/HAM10000_metadata.csv'
        main_dir = "/ubc/ece/home/ra/grads/nourhanb/Documents/skin/HAM/images"
        input_size = 28

        trainloader, valloader, testloader = load_data(main_dir , meta_dir, input_size)

    elif name =='Fitzpatrick17':
        #root =  '/ubc/ece/home/ra/grads/nourhanb/Documents/skin/fitzpatrick17k/data/finalfitz17k'
        root =  '/ubc/ece/home/ra/grads/nourhanb/Downloads/archive/all_images'
        #root =  '/ubc/ece/home/ra/grads/nourhanb/Downloads/ISIC_2020_Training_JPEG/train'
        batch_size = 30
        shuffle = False
        input_size = 100

        trainloader,valloader,testloader = get_fitz_dataloaders(root, batch_size, shuffle, input_size, partial_skin_types=[], partial_ratio=1.0)

    elif name =='GbP':
        root = '/ubc/ece/home/ra/grads/nourhanb/Documents/miccai_2024/GbP/images-224/images-224'
        batch_size = 32
        shuffle = False
        input_size = 100

        trainloader,valloader,testloader = get_fitz_dataloaders(root, batch_size, shuffle, input_size, partial_skin_types=[], partial_ratio=1.0)

    else:
        print("This dataset is not implemented")


    return trainloader, valloader, testloader