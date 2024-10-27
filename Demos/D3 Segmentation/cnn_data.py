import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import cv2
import albumentations as A
from PIL import Image

IMAGE_PATH = "cnn_segmentation/data/dataset/semantic_drone_dataset/original_images/"
MASK_PATH = (
    "cnn_segmentation/data/dataset/semantic_drone_dataset/label_images_semantic/"
)


# Function to create a DataFrame containing image IDs from the dataset
def create_df():
    # Initialize empty list to store image names
    name = []
    # Walk through the image directory
    for dirname, _, filenames in os.walk(IMAGE_PATH):
        # For each file, extract the name without extension and add to list
        for filename in filenames:
            name.append(filename.split(".")[0])
    # Create and return a DataFrame with image IDs and sequential indices
    return pd.DataFrame({"id": name}, index=np.arange(0, len(name)))


def get_data_splits():
    # Create DataFrame containing image IDs
    df = create_df()

    # Split data into training+validation set (90%) and test set (10%)
    X_trainval, X_test = train_test_split(
        df["id"].values, test_size=0.1, random_state=19
    )

    # Further split training+validation set into training set (85%) and validation set (15%)
    X_train, X_val = train_test_split(X_trainval, test_size=0.15, random_state=19)

    # Return the three splits: training, validation and test sets
    return X_train, X_val, X_test


class DroneDataset(Dataset):
    def __init__(self, img_path, mask_path, X, mean, std, transform=None, patch=False):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform
        self.patches = patch
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Load image and convert from BGR to RGB color space
        img = cv2.imread(self.img_path + self.X[idx] + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load corresponding segmentation mask in grayscale
        mask = cv2.imread(self.mask_path + self.X[idx] + ".png", cv2.IMREAD_GRAYSCALE)

        # Apply data augmentation transforms if specified
        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug["image"])
            mask = aug["mask"]

        # Convert numpy array to PIL Image if no transforms applied
        if self.transform is None:
            img = Image.fromarray(img)

        # Convert image to tensor and normalize using ImageNet statistics
        t = T.Compose([T.ToTensor(), T.Normalize(self.mean, self.std)])
        img = t(img)

        # Convert mask to long tensor
        mask = torch.from_numpy(mask).long()

        # Split image and mask into patches if specified
        if self.patches:
            img, mask = self.tiles(img, mask)

        # Return image-mask pair
        return img, mask

    def tiles(self, img, mask):
        # Split image into overlapping patches of size 512x768
        # First unfold creates patches along height, second unfold along width
        img_patches = img.unfold(1, 512, 512).unfold(2, 768, 768)

        # Reshape patches into (3, num_patches, 512, 768) where 3 is number of channels
        img_patches = img_patches.contiguous().view(3, -1, 512, 768)

        # Rearrange dimensions to get (num_patches, 3, 512, 768) for batch processing
        img_patches = img_patches.permute(1, 0, 2, 3)

        # Split mask into corresponding patches of same size
        # Mask is single channel so only needs two unfolds
        mask_patches = mask.unfold(0, 512, 512).unfold(1, 768, 768)

        # Reshape into (num_patches, 512, 768)
        mask_patches = mask_patches.contiguous().view(-1, 512, 768)

        # Return image patches and corresponding mask patches
        return img_patches, mask_patches


class DroneTestDataset(Dataset):
    def __init__(self, img_path, mask_path, X, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path + self.X[idx] + ".jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_path + self.X[idx] + ".png", cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            aug = self.transform(image=img, mask=mask)
            img = Image.fromarray(aug["image"])
            mask = aug["mask"]

        if self.transform is None:
            img = Image.fromarray(img)

        mask = torch.from_numpy(mask).long()

        return img, mask


def get_data_loaders(batch_size=16):
    X_train, X_val, X_test = get_data_splits()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Describe the transformations applied to the training set
    t_train = A.Compose(
        [
            A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.GridDistortion(p=0.2),
            A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
            A.GaussNoise(),
        ]
    )

    # Describe the transformations applied to the validation set
    t_val = A.Compose(
        [
            A.Resize(704, 1056, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(),
            A.GridDistortion(p=0.2),
        ]
    )

    train_set = DroneDataset(
        IMAGE_PATH, MASK_PATH, X_train, mean, std, t_train, patch=False
    )
    val_set = DroneDataset(IMAGE_PATH, MASK_PATH, X_val, mean, std, t_val, patch=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    t_test = A.Resize(768, 1152, interpolation=cv2.INTER_NEAREST)
    test_set = DroneTestDataset(IMAGE_PATH, MASK_PATH, X_test, transform=t_test)

    return train_loader, val_loader, test_set
