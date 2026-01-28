import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import models
import dataset
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10

def compute_iou(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    intersection = np.diag(cm)
    ground_truth_set = cm.sum(axis=1)
    predicted_set = cm.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    iou = intersection / (union.astype(np.float32) + 1e-10)
    return np.nanmean(iou), iou

def evaluate():
    model = models.segmentation.deeplabv3_resnet50(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, NUM_CLASSES, kernel_size=(1, 1))
    model.load_state_dict(torch.load('./output/best_model.pth', map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    val_dataset = dataset.DualityDataset('./data/val/images', './data/val/masks')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_targets = []

    print("Running Evaluation...")
    with torch.no_grad():
        for images, masks in tqdm(val_loader):
            images = images.to(DEVICE)
            output = model(images)['out']
            pred_mask = torch.argmax(output, dim=1).cpu().numpy()
            all_preds.append(pred_mask)
            all_targets.append(masks.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    mIoU, per_class_iou = compute_iou(y_pred, y_true)
    
    print(f"Mean IoU: {mIoU:.4f}")
    for i, iou in enumerate(per_class_iou):
        print(f"{dataset.CLASSES[i]}: {iou:.4f}")

if __name__ == '__main__':
    evaluate()
