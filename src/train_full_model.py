import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from torch.utils.data import ConcatDataset

# ===============================
# 可修改参数
# ===============================

TRAIN_PATH = "frames/original/train"
TRAIN_PATH2 = "frames/crf28/train"
TRAIN_PATH3 = "frames/crf35/train"
TEST_PATH  = "frames/original/test"

MODEL_NAME = "mixed_model"

BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-4

# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# 数据增强
# ===============================

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# 加载数据
# ===============================

dataset1 = datasets.ImageFolder(TRAIN_PATH, transform=train_transform)
dataset2 = datasets.ImageFolder(TRAIN_PATH2, transform=train_transform)
dataset3 = datasets.ImageFolder(TRAIN_PATH3, transform=train_transform)

print("Train classes:", dataset1.class_to_idx, '\n', dataset2.class_to_idx, '\n', dataset3.class_to_idx)

train_dataset = ConcatDataset([dataset1, dataset2, dataset3])
test_dataset = datasets.ImageFolder(TEST_PATH, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Test classes:", test_dataset.class_to_idx)

# ===============================
# 构建模型（全模型训练）
# ===============================

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ===============================
# 训练
# ===============================

for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {running_loss/len(train_loader):.4f} "
          f"Train Acc: {train_acc:.4f}")

# ===============================
# 测试
# ===============================

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\n===== Test Results =====")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print("Confusion Matrix:")
print(cm)

# ===============================
# 保存模型和结果
# ===============================

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

torch.save(model.state_dict(), f"models/{MODEL_NAME}.pth")

df = pd.DataFrame({
    "label": all_labels,
    "pred": all_preds
})
df.to_csv(f"results/{MODEL_NAME}_predictions.csv", index=False)

with open(f"results/{MODEL_NAME}_metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1: {f1:.4f}\n")
    f.write(f"Confusion Matrix:\n{cm}\n")

print("\nModel and results saved.")