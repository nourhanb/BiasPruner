from sklearn.model_selection import train_test_split
from torchvision import transforms
from glob import glob
import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
from collections import Counter
import torch
from torch.utils.data import Dataset
from PIL import Image

from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder


from torch.utils.data import WeightedRandomSampler

def load_data(main_dir, meta_dir, input_size):


    norm_mean, norm_std = compute_img_mean_std(main_dir, input_size)

    df_train, df_test = create_dataframes(main_dir, meta_dir)

    train_transform, test_transform = create_transformations(input_size, norm_mean, norm_std)

    training_val_set = HAM10000(df_train, transform=train_transform)

    training_set, val_set = train_test_split(training_val_set, test_size=0.15)

    train_loader = DataLoader(training_set, batch_size=50, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=50, shuffle=True, num_workers=8)

    testing_set = HAM10000(df_test, transform=test_transform)
    test_loader = DataLoader(testing_set, batch_size=50, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader


def compute_img_mean_std(image_paths, input_size):
    """
        computing the mean and std of three channel on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """
    # path = os.path.join(image_paths,'HAM10000_img')
    all_image_path_dmf = glob(os.path.join(image_paths,'*.jpg'))

    img_h, img_w = input_size, input_size
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(all_image_path_dmf))):
        img = cv2.imread(all_image_path_dmf[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))

    return means,stdevs


def create_dataframes(data_dir, meta_dir):
    # path = os.path.join(data_dir_dmf,'HAM10000_img')
    all_image_path_dmf = glob(os.path.join(data_dir,'*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path_dmf}
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    df_original = pd.read_csv(meta_dir)
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

    # this will tell us how many images are associated with each lesion_id
    df_undup = df_original.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    # here we identify lesion_id's that have duplicate images and those that have only one image.
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']
    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)

    print(df_original['duplicates'].value_counts())


    # now we filter out images that don't have duplicates
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']

    # now we create a test set using df because we are sure that none of these images have augmented duplicates in the train set
    y = df_undup['cell_type_idx']
    _, df_test = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)

    print(df_test.shape)

    # This set will be df_original excluding all rows that are in the test set
    # This function identifies if an image is part of the train or test set.
    def get_test_rows(x):
        # create a list of all the lesion_id's in the test set
        val_list = list(df_test['image_id'])
        if str(x) in val_list:
            return 'test'
        else:
            return 'train'

    # identify train and test rows
    # create a new colum that is a copy of the image_id column
    df_original['train_or_test'] = df_original['image_id']
    # apply the function to this new column
    df_original['train_or_test'] = df_original['train_or_test'].apply(get_test_rows)
    # filter out train rows
    df_train_DONT = df_original[df_original['train_or_test'] == 'train']

    print(df_train_DONT['cell_type_idx'].value_counts())
    print(df_train_DONT['cell_type'].value_counts())

    # Copy fewer class to balance the number of 7 classes
    data_aug_rate =[19,12,5,54,0,5,45] #[2,1,1,5,1,4,2] ## for HAM: [19,12,5,54,0,5,45]
    # for i in range(7):
    #     if data_aug_rate[i]:
    #         df_aug = df_train_DONT.loc[df_train_DONT['cell_type_idx'] == i,:]*(data_aug_rate[i]-1)
    #         df_train_DONT = pd.concat([df_train_DONT,df_aug], ignore_index=True)

    dfs_to_concat = []

    for i in range(7):
        if data_aug_rate[i]:
            condition = df_train_DONT['cell_type_idx'] == i
            selected_data = df_train_DONT[condition]
            repeated_data = [selected_data] * (data_aug_rate[i] - 1)
            dfs_to_concat.extend(repeated_data)

    # Concatenate the DataFrames in dfs_to_concat
    if dfs_to_concat:
        df_train_DONT = pd.concat([df_train_DONT] + dfs_to_concat, ignore_index=True)

    train_class_counts = df_train_DONT['cell_type_idx'].value_counts()
    print("Train Set:")
    print(train_class_counts)
    test_class_counts = df_test['cell_type_idx'].value_counts()
    print("Test Set:")
    print(test_class_counts)
    df_train_DONT = df_train_DONT.reset_index()
    df_test = df_test.reset_index()


    return df_train_DONT, df_test


def create_transformations(input_size, norm_mean, norm_std):
    # define the transformation of the train images.
    train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                            transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
    # define the transformation of the test images.
    test_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])

    return train_transform, test_transform


class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

        # Filter out rows with None image paths
        self.df = self.df.dropna(subset=['path'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        try:
            X = Image.open(self.df['path'].iloc[index])
        except (IOError, OSError) as e:
            print(f"Error opening image: {self.df['path'].iloc[index]} - {e}")
            return None, None

        y = torch.tensor(int(self.df['cell_type_idx'].iloc[index]))

        if self.transform:
            X = self.transform(X)

        return X, y

class SkinDataset():
    def __init__(self, df, is_bias, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.is_bias = is_bias 
        self.root_dir = root_dir
        self.transform = transform

        # Initialize label encoder
        self.label_encoder = LabelEncoder()
        self.df['benign_malignant'] = self.label_encoder.fit_transform(self.df['benign_malignant'])
        self.df['age'] = self.label_encoder.fit_transform(self.df['age'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.df.loc[self.df.index[idx], 'image_name'] + ".jpg")
        image = Image.open(img_name)

        # Use label encoder to convert label to numerical format
        if self.is_bias:
            label = self.df.loc[self.df.index[idx], 'benign_malignant']
        else:
            label = self.df.loc[self.df.index[idx], 'age']

        if self.transform:
            image = self.transform(image)

        # Convert the label to a PyTorch tensor
        label = torch.tensor(label)

        # Return separate tensors for image and label
        return image, label



def get_fitz_dataloaders(root, batch_size, shuffle, input_size, partial_skin_types=[], partial_ratio=1.0):

    #test_dir = '/ubc/ece/home/ra/grads/nourhanb/Documents/miccai_2024/GbP/age_ISIC_2020/flipped_unbiased_test_set.csv'
    #val_ratio = 0.1  

    df = pd.read_csv('/ubc/ece/home/ra/grads/nourhanb/Documents/miccai_2024/GbP/age_ISIC_2020/flipped_biased_train_set.csv')
    train_df, val_df = train_test_split(df, test_size=val_ratio, stratify=df['benign_malignant'])
    train_df.to_csv('/ubc/ece/home/ra/grads/nourhanb/Documents/miccai_2024/GbP/age_ISIC_2020/flipped_total_biased_train_set_split.csv', index=False)
    val_df.to_csv('/ubc/ece/home/ra/grads/nourhanb/Documents/miccai_2024/GbP/age_ISIC_2020/flipped_total_biased_val_set_split.csv', index=False)
    train_dir='/ubc/ece/home/ra/grads/nourhanb/Documents/miccai_2024/GbP/age_ISIC_2020/flipped_total_biased_train_set_split.csv'
    val_dir= '/ubc/ece/home/ra/grads/nourhanb/Documents/miccai_2024/GbP/age_ISIC_2020/flipped_total_biased_val_set_split.csv'

    # Extract the directory name and file name
    directory_name, file_name = os.path.split(train_dir)
    print("Directory Name:", directory_name)
    print("File Name:", file_name)
    val = pd.read_csv(val_dir)
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)


    transformed_train = SkinDataset(
        df=train,
        is_bias= True,
        root_dir=root,
        transform=transforms.Compose([
            #transforms.RandomRotation(degrees=15),
            #transforms.RandomHorizontalFlip(),
            #transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=(input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],  [0.229, 0.224, 0.225])
            #transforms.Normalize([0.485], [0.229])
        ])
    )

    transformed_val = SkinDataset(
        df=val,
        is_bias= True,
        root_dir=root,
        transform=transforms.Compose([
            transforms.Resize(size=(input_size, input_size)),
            #transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #transforms.Normalize([0.485], [0.229])
        ])
    )

    transformed_test = SkinDataset(
        df=test,
        is_bias= False,
        root_dir=root,
        transform=transforms.Compose([
            transforms.Resize(size=(input_size, input_size)),
            #transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #transforms.Normalize([0.485], [0.229])
        ])
    )
    
    
    #class_weights = calculate_class_weights(transformed_train, use_counter=True)
    #weights = [class_weights[label] for _, label in transformed_train]
    #weighted_sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        transformed_train,
        batch_size=batch_size,
        #sampler=weighted_sampler,
        drop_last=True,
        num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        transformed_val,
        batch_size=batch_size,
        shuffle=shuffle, drop_last=True,     num_workers=4, pin_memory=True  )

    test_loader = torch.utils.data.DataLoader(
        transformed_test,
        batch_size=batch_size,
        shuffle=False, drop_last=False,     num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader