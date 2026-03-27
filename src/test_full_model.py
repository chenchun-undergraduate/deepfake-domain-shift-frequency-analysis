import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import os
from torch.utils.data import ConcatDataset

# ==========================
# 基本设置
# ==========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "models/crf28_full_model.(将训练场景随机打乱)pth"
TEST_PATH  = "frames/crf35/test"
TEST_PATH2 = "frames/crf28/test"
TEST_PATH3 = "frames/original/test"
RESULTS_DIR = "results"
BATCH_SIZE = 32

# ==========================
# 数据预处理
# ==========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset1 = datasets.ImageFolder(TEST_PATH, transform=transform)
dataset2 = datasets.ImageFolder(TEST_PATH2, transform=transform)
dataset3 = datasets.ImageFolder(TEST_PATH3, transform=transform)


test_dataset = ConcatDataset([dataset1, dataset2, dataset3])
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==========================
# 构建模型
# ==========================

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Testing on:", TEST_PATH)

# ==========================
# 测试
# ==========================

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ==========================
# 计算指标
# ==========================

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

# ==========================
# 保存预测结果 CSV
# ==========================

df = pd.DataFrame({
    "True_Label": all_labels,
    "Predicted_Label": all_preds
})

csv_path = os.path.join(RESULTS_DIR, "original-->mixed predictions.csv")
df.to_csv(csv_path, index=False)

print(f"Predictions saved to {csv_path}")

# ==========================
# 保存指标 TXT
# ==========================

metrics_path = os.path.join(RESULTS_DIR, "original-->mixed metrics.txt")

with open(metrics_path, "w") as f:
    f.write("===== Test Results =====\n")
    f.write(f"Accuracy : {acc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm))

print(f"Metrics saved to {metrics_path}")