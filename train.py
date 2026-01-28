import os
import torch
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import dataset

BATCH_SIZE = 8
EPOCHS = 20
LR = 0.0001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10 

def get_model():
    model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
    model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=(1, 1))
    return model.to(DEVICE)

def train():
    train_dataset = dataset.DualityDataset('./data/train/images', './data/train/masks')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs('output', exist_ok=True)
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for images, masks in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), './output/best_model.pth')

if __name__ == '__main__':
    train()
