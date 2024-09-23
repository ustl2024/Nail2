import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score
from skimage.transform import resize

class NailDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])
        self.label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.png') or f.endswith('.jpg')])

        if len(self.image_files) != len(self.label_files):
            raise ValueError(
                f"Mismatch between number of images ({len(self.image_files)}) and labels ({len(self.label_files)})")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if idx >= len(self.image_files):
            raise IndexError(f"Index {idx} out of range for image_files list with length {len(self.image_files)}")
        if idx >= len(self.label_files):
            raise IndexError(f"Index {idx} out of range for label_files list with length {len(self.label_files)}")

        img_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path).convert('L')

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Create datasets and dataloaders
train_dataset = NailDataset(r'D:\pythonProject\T3\archive\images', r'D:\pythonProject\T3\archive\labels',
                            transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
