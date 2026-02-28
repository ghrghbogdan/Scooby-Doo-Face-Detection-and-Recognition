import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import gc

CLASS_MAP = {
    'daphne': 0,
    'fred': 1,
    'shaggy': 2,
    'velma': 3,
    'unknown': 4
}

class ScoobyDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, split='train', transform=None):

        self.images_dir = images_dir
        self.transform = transform
        self.samples = [] 

        characters = ['daphne', 'fred', 'shaggy', 'velma']

        for char_name in characters:
            if split == 'train':
                filename = f"{char_name}_annotations.txt"
                file_path = os.path.join(annotations_dir, filename)
                
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 6:
                                img_name = parts[0]
                                x1, y1, x2, y2 = map(int, parts[1:5])
                                label_str = parts[5].lower()
                                
                                if label_str in CLASS_MAP:
                                    img_full_path = os.path.join(images_dir, char_name, img_name)
                                    if not os.path.exists(img_full_path):
                                        img_full_path = os.path.join(images_dir, img_name)
                                        
                                    if os.path.exists(img_full_path):
                                        self.samples.append((img_full_path, (x1, y1, x2, y2), CLASS_MAP[label_str]))

            elif split == 'val':
                filename = f"task2_{char_name}_gt_validare.txt"
                file_path = os.path.join(annotations_dir, filename)
                
                if os.path.exists(file_path):
                    current_label = CLASS_MAP[char_name]
                    
                    with open(file_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                img_name = parts[0]
                                x1, y1, x2, y2 = map(int, parts[1:5])
                                
                                img_full_path = os.path.join(images_dir, img_name)
                                
                                if os.path.exists(img_full_path):
                                    self.samples.append((img_full_path, (x1, y1, x2, y2), current_label))
                else:
                    print(f"Warning: Nu am gasit {filename} in {annotations_dir}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, bbox, label = self.samples[idx]
        
        image = Image.open(img_path).convert('RGB')
        face_crop = image.crop(bbox)
        
        if self.transform:
            face_crop = self.transform(face_crop)
        return face_crop, label

class CNN_v1(nn.Module):
    def __init__(self):
        super(CNN_v1, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
                 
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
   
            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout2d(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 5)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_cnn(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    model.to(device)
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 5
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for i, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = (correct_predictions / total_predictions) * 100
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1} Train Acc: {train_acc}, Val Acc: {val_acc}')
        
        if counter >= patience:
            print(f"Early stopping")
            break
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(16)
    print(f"Using device: {device}")
    
    train_transform, val_transform = get_transforms()
    
    train_root = os.path.join('..', 'antrenare')
    val_images_root = os.path.join('..', 'validare', 'validare') 
    val_annotations_root = os.path.join('..', 'validare')

    train_ds = ScoobyDataset(
        images_dir=train_root, 
        annotations_dir=train_root, 
        split='train', 
        transform=train_transform
    )

    val_ds = ScoobyDataset(
        images_dir=val_images_root, 
        annotations_dir=val_annotations_root, 
        split='val', 
        transform=val_transform
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    
    criterion = nn.CrossEntropyLoss()

    print("Training Model 1")
    model1 = CNN_v1()
    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'max', patience=5, factor=0.5)
    
    model1 = train_cnn(
        model1, train_loader, val_loader, 
        criterion, optimizer1, scheduler1, 
        num_epochs=100, device=device
    )

    torch.save(model1.state_dict(), 'model1.pth')
    print("\nModel 1 complete!")
    
    del optimizer1, scheduler1
    gc.collect()

if __name__ == "__main__":
    main()