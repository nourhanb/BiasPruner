import param
import torch
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
from torchvision import transforms, models
import pandas as pd
import os
from PIL import Image
from functions import IdxDataset

def custom_collate(batch):
    idx, images, labels, bias = zip(*batch)

    # Find the maximum number of channels in the batch
    max_channels = max(image.shape[0] for image in images)

    # Pad or duplicate channels to make all images have the same number of channels
    processed_images = []
    for image in images:
        if image.shape[0] < max_channels:
            # Pad channels with zeros
            pad_channels = max_channels - image.shape[0]
            image = torch.cat([image, torch.zeros(pad_channels, *image.shape[1:])], dim=0)
        elif image.shape[0] > max_channels:
            # Take only the first max_channels channels
            image = image[:max_channels, :, :]
        processed_images.append(image)
    return idx, torch.stack(processed_images), torch.tensor(labels), torch.tensor(bias)

# class CustomDataset(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.data_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#         self.label_mapping = {'akiec':0, 'bcc':1} #{'akiec': 0, 'bcc': 1}#, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
#         self.data_frame['label'] = self.data_frame['dx'].map(self.label_mapping)
#
#     def __len__(self):
#         return len(self.data_frame)
#
#     def __getitem__(self, idx):
#         #img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
#         img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 1] + '.jpg')
#         image = Image.open(img_name)
#
#         # Convert the image to grayscale and then to RGB
#         #image = transforms.Grayscale(num_output_channels=3)(image)
#
#         # Resize the image to a consistent size
#         image = self.transform(image)
#
#         label = self.data_frame.iloc[idx, 7]
#         bias_mapping = {'male': 0, 'female': 1}
#
#         bias = bias_mapping[self.data_frame.iloc[idx, 5]]
#
#
#         return idx, image, label, bias
#
#     def get_additional_info(self, idx, column_name):
#         return self.data_frame.loc[idx, column_name]


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.label_mapping = {'Atelectasis': 0, 'Consolidation': 1, 'Infiltration': 2, 'Pneumothorax': 3, 'Edema': 4, 'Emphysema': 5, 'Fibrosis': 6,
                              'Effusion': 7, 'Pneumonia': 8, 'Pleural_Thickening': 9, 'Cardiomegaly': 10, 'Nodule': 11, 'Mass': 12, 'Hernia': 13}
        self.data_frame['label'] = self.data_frame['FindingLabels'].map(self.label_mapping)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)

        # Convert the image to grayscale and then to RGB
        image = transforms.Grayscale(num_output_channels=3)(image)

        # Resize the image to a consistent size
        image = self.transform(image)

        label = self.data_frame.iloc[idx, -1]
        bias_mapping = {'M': 0, 'F': 1}

        bias = bias_mapping[self.data_frame.iloc[idx, 5]]

        return idx, image, label, bias


def loaders(csv_file, root_dir, transform):

    train_dataset = CustomDataset(csv_file=os.path.join(csv_file, 'train.csv'), root_dir=root_dir, transform=transform)
    val_dataset = CustomDataset(csv_file=os.path.join(csv_file, 'val.csv'), root_dir=root_dir, transform=transform)
    test_dataset = CustomDataset(csv_file=os.path.join(csv_file, 'test.csv'), root_dir=root_dir, transform=transform)



    # Calculate class weights for weighted sampler based on the training dataset
    class_counts_train = pd.Series(train_dataset.data_frame['label']).value_counts().to_dict()
    class_weights_train = {c: 1 / count for c, count in class_counts_train.items()}
    sample_weights_train = [class_weights_train[c] for c in train_dataset.data_frame['label']]

    # Use WeightedRandomSampler for imbalanced classes
    sampler = WeightedRandomSampler(sample_weights_train, len(sample_weights_train), replacement=True)

    # train_dataset = IdxDataset(train_dataset)

    # Create a DataLoader with the WeightedRandomSampler
    train_loader = DataLoader(train_dataset, batch_size=param.batch, sampler=sampler,  collate_fn=custom_collate, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=param.batch, shuffle=True, collate_fn=custom_collate, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=param.batch, shuffle=True, collate_fn=custom_collate, num_workers=4, pin_memory=True)
    sample_loader = DataLoader(train_dataset, batch_size=param.batch, collate_fn=custom_collate, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader, sample_loader
