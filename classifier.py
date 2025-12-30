import os
import random
from pathlib import Path
from PIL import Image
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from tqdm.auto import tqdm

torch.manual_seed(42)
torch.cuda.manual_seed(42)

label_map = {
    **{c: "dry" for c in ["dry_asphalt_severe", "dry_asphalt_slight", "dry_asphalt_smooth", "dry_concrete_severe", "dry_concrete_slight", "dry_concrete_smooth", "dry_gravel", "dry_mud"]},
    **{c: "wet" for c in ["wet_asphalt_severe", "wet_asphalt_slight", "wet_asphalt_smooth", "wet_concrete_severe", "wet_concrete_slight", "wet_concrete_smooth", "wet_gravel", "wet_mud"]},
    **{c: "standing_water" for c in ["water_asphalt_severe", "water_asphalt_slight", "water_asphalt_smooth", "water_concrete_severe", "water_concrete_slight", "water_concrete_smooth", "water_gravel", "water_mud"]},
    **{c: "snow" for c in ["fresh_snow", "melted_snow"]},
    **{c: "ice" for c in ["ice"]}
}

class_to_idx = {"dry": 0, "wet": 1, "standing_water": 2, "snow": 3, "ice": 4}
class_names = ["dry", "wet", "standing_water", "snow", "ice"]

BATCH_SIZE = 64
MAX_EPOCHS = 50

device = "cuda" if torch.cuda.is_available() else "cpu" # Y/N for Kaggle????

dataset_path_train = "/kaggle/input/rscd-dataset-1million/RSCD dataset-1million/train"
dataset_path_test = "/kaggle/input/rscd-dataset-1million/RSCD dataset-1million/test_50k"
dataset_path_vali = "/kaggle/input/rscd-dataset-1million/RSCD dataset-1million/vali_20k"

class RSCDDataset(Dataset):
    def __init__(self, root_dir, transform, label_map, class_names, class_to_idx, max_samples, balanced=True, samples_file=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.label_map = label_map
        self.class_names = class_names
        self.class_to_idx = class_to_idx
        self.max_samples = max_samples
        self.samples = []

        if samples_file is not None:
            with open(samples_file, 'r') as infile:
                self.samples = json.load(infile)
            return

        random.seed(42)

        samples_by_class = {class_name: [] for class_name in self.class_names}
    
        for original_label in self.root_dir.iterdir():
            if original_label.is_dir() and original_label.name in self.label_map:
                mapped_label = self.label_map[original_label.name]
                label_idx  = self.class_to_idx[mapped_label]

                for img_path in original_label.glob('*.jpg'):
                    samples_by_class[mapped_label].append((str(img_path), label_idx, mapped_label))
        
        if max_samples and balanced:
            num_classes = len(self.class_names)
            samples_per_class = max_samples // num_classes
            for class_name, class_samples in samples_by_class.items():
                n_to_take = min(samples_per_class, len(class_samples))
                selected = random.sample(class_samples, n_to_take)
                self.samples.extend(selected)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx, label_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB') # don't know why .convert() is used. look into?
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx

class RSCDDatasetTestVali(Dataset):
    def __init__(self, root_dir, transform, label_map, class_names, class_to_idx, max_samples, balanced=True, samples_file=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.label_map = label_map
        self.class_names = class_names
        self.class_to_idx = class_to_idx
        self.max_samples = max_samples
        self.samples = []
        self.samples_file = samples_file

        if self.samples_file is not None:
            with open(self.samples_file, 'r') as infile:
                self.samples = json.load(infile)
            return

        random.seed(42)

        samples_by_class = {class_name: [] for class_name in self.class_names}

        for image_path in self.root_dir.iterdir():
            if "dry" in str(image_path):
                samples_by_class["dry"].append((str(image_path), self.class_to_idx["dry"], "dry"))
            elif "wet" in str(image_path):
                samples_by_class["wet"].append((str(image_path), self.class_to_idx["wet"], "wet"))
            elif "water" in str(image_path):
                samples_by_class["standing_water"].append((str(image_path), self.class_to_idx["standing_water"], "standing_water"))
            elif "snow" in str(image_path):
                samples_by_class["snow"].append((str(image_path), self.class_to_idx["snow"], "snow"))
            elif "ice" in str(image_path):
                samples_by_class["ice"].append((str(image_path), self.class_to_idx["ice"], "ice"))

        if max_samples and balanced:
            num_classes = len(self.class_names)
            samples_per_class = max_samples // num_classes
            for class_name, class_samples in samples_by_class.items():
                n_to_take = min(samples_per_class, len(class_samples))
                selected = random.sample(class_samples, n_to_take)
                self.samples.extend(selected)
                
    def __len__(self):
            return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx, label_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB') # don't know why .convert() is used. look into?
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx 
            
def find_distribution(dataset):
    labels = [sample[1] for sample in dataset.samples]
    unique, counts = np.unique(labels, return_counts=True) # finds duplicates and removes them, returns unique labels and number of images in each label class
    print("Class distribution: ")
    for idx, count in zip(unique, counts):
        print(f"{idx}/{dataset.class_names[idx]}: {count} images ({count/len(labels)*100:.2f}%)")

    return dict(zip(unique, counts))

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(size=(224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])    
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

vali_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = RSCDDataset(dataset_path_train,  
                            transform=train_transform, 
                            label_map=label_map, 
                            class_names=class_names,
                            class_to_idx=class_to_idx,
                            max_samples=160000,
                            balanced=True,
                            samples_file="train_samples.json")

test_dataset = RSCDDatasetTestVali(dataset_path_test,
                          transform=test_transform,
                          label_map=label_map,
                          class_names=class_names,
                          class_to_idx=class_to_idx,
                          max_samples=20000,
                          balanced=True,
                          samples_file="test_samples.json")
                                   
vali_dataset = RSCDDatasetTestVali(dataset_path_vali,
                          transform=test_transform,
                          label_map=label_map,
                          class_names=class_names,
                          class_to_idx=class_to_idx,
                          max_samples=20000,
                          balanced=True,
                          samples_file="vali_samples.json")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
vali_dataloader = DataLoader(vali_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)

weights = models.Mobilenet_V3_Large_Weights.DEFAULT
model = models.mobilenet_v3_large(weights=weights).to(device)

for param in model.features.parameters():
    param.requires_grad=False

out_shape = len(class_names)

features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(features, out_shape)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

class EarlyStopping:
    def __init__(self, patience=5, delta=0.00001):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.wait_count = 0

    def checkEarlyStop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                return True
            return False

earlyStopping = EarlyStopping(patience=5, delta=0.00001)
history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': [] }

for epoch in range(MAX_EPOCHS):
    # TRAINING
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    train_progress = tqdm(train_dataloader, desc="Training", leave=False)
    for images, labels in train_progress:
        images, labels = images.to(device), labels.to(device)
        label_pred = model(images)
        loss = loss_fn(label_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        _, predicted = torch.max(label_pred.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        train_loss += loss.item()
        train_progress.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{train_correct / train_total:.2f}%'
        })

    train_loss = train_loss / len(train_dataloader)
    train_accuracy = train_correct / train_total

    # VALIDATING
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.inference_mode():
        val_progress = tqdm(vali_dataloader, desc="Validation", leave=False)
        for images, labels in val_progress:
            images, labels = images.to(device), labels.to(device)
            label_pred = model(images)
            loss = loss_fn(label_pred, labels)

            _, predicted = torch.max(label_pred.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_loss += loss.item()
            val_progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{val_correct / val_total:.2f}%'
            })

    val_loss = val_loss / len(vali_dataloader)
    val_accuracy = val_correct / val_total
    # KEEPING TRACK OF DATA FOR EARLY STOPPING
    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_accuracy)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_accuracy)

    if epoch == 0 or val_accuracy > max(history['val_accuracy'][:-1]):
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
        print("Best val accuracy model saved!")

    if earlyStopping.checkEarlyStop(val_loss):
        print(f"Final val accuracy: {val_accuracy}.")
        print(f"Final val loss: {val_loss}")
        break

print("Training complete!")
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
