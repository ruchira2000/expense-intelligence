import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

RECEIPTS_DIR = Path("data/receipts")
LABELS_CSV = Path("data/labels/category_labels.csv")
MODEL_OUT = Path("model/receipt_classifier.pth")
CONF_MATRIX_OUT = Path("static/confusion_matrix.png")

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.0003


class ReceiptDataset(Dataset):
    def __init__(self, df, class_to_idx, transform):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = RECEIPTS_DIR / row["image_filename"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.class_to_idx[row["category"]]
        return image, label


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    df = pd.read_csv(LABELS_CSV)
    classes = sorted(df["category"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["category"], random_state=42
    )

    # class weighting: since Retail has far more examples than Travel & Utilities,
    # we tell the loss function to penalize mistakes on rare categories more heavily,
    # so the model can't just get lazy and guess Retail every time
    train_labels = train_df["category"].map(class_to_idx).values
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(classes)), y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print("Class weights:", dict(zip(classes, class_weights.tolist())))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = ReceiptDataset(train_df, class_to_idx, transform)
    val_ds = ReceiptDataset(val_df, class_to_idx, transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{EPOCHS}, loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Receipt Category Classification, Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONF_MATRIX_OUT)
    print(f"Saved confusion matrix to {CONF_MATRIX_OUT}")

    torch.save({"model_state": model.state_dict(), "classes": classes}, MODEL_OUT)
    print(f"Saved trained model to {MODEL_OUT}")


if __name__ == "__main__":
    main()