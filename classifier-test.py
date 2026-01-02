import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import json
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from tqdm.auto import tqdm
import timm
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import F1Score
import matplotlib.pyplot as plt

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
MAX_EPOCHS = 10

device = "cuda" if torch.cuda.is_available() else "cpu" # Y/N for Kaggle????

dataset_path_train = "/kaggle/input/rscd-dataset-1million/RSCD dataset-1million/train"
dataset_path_test = "/kaggle/input/rscd-dataset-1million/RSCD dataset-1million/train"
dataset_path_vali = "/kaggle/input/rscd-dataset-1million/RSCD dataset-1million/train"

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

val_transform = transforms.Compose([
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
                            samples_file = '../input/file-path-ds/train_file.json')

val_dataset = RSCDDataset(dataset_path_train,  
                            transform=val_transform, 
                            label_map=label_map, 
                            class_names=class_names,
                            class_to_idx=class_to_idx,
                            max_samples=20000,
                            balanced=True,
                            samples_file = '../input/file-path-ds/val_file.json')

test_dataset = RSCDDataset(dataset_path_train,  
                            transform=test_transform, 
                            label_map=label_map, 
                            class_names=class_names,
                            class_to_idx=class_to_idx,
                            max_samples=20000,
                            balanced=True,
                            samples_file = '../input/file-path-ds/test_file.json')

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True, shuffle=True)


model = timm.create_model('tf_mobilenetv3_large_100.in1k', checkpoint_path='/kaggle/input/tf-mobilenet-v3/pytorch/tf-mobilenetv3-large-100/1/tf_mobilenetv3_large_100-427764d5.pth')

out_shape = len(class_names)

for param in model.parameters():
    param.requires_grad = True

num_blocks = len(model.blocks)
for i in range(num_blocks - 3):
    for param in model.blocks[i].parameters():
        param.requires_grad = False

for param in model.conv_stem.parameters():
    param.requires_grad = False

model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=out_shape, 
                    bias=True)).to(device)

model_path = '/kaggle/input/modelooo/best_model (1).pth'
state_dict = torch.load(model_path, map_location=device)

model.load_state_dict(state_dict)

model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=4e-5)

confmat = ConfusionMatrix(task="multiclass", num_classes=5).to(device)
f1 = F1Score(task="multiclass", num_classes=5, average="macro").to(device)

# TESTING
model.eval()
test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    test_progress = tqdm(test_dataloader, desc="Testing", leave=False)
    for images, labels in test_progress:
        images, labels = images.to(device), labels.to(device)
        label_pred = model(images)

        _, predicted = torch.max(label_pred.data, 1)
        confmat.update(predicted, labels)
        f1.update(predicted, labels)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        test_progress.set_postfix({
            'acc': f'{100 * (test_correct / test_total):.2f}%'
        })
f1_final = f1.compute()
print("Testing Complete!")
print(f"Accuracy was {100 * (test_correct / test_total):.2f}%")
print(f"F1 Score: {f1_final}")
print(f"Confusion Matrix: ")
fig, ax = confmat.plot()
plt.show()
cm = confmat.compute()
print(f"And numeric... {cm}!")
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
