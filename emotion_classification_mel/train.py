import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
from glob import glob
from emotion_classification_mel.model import EmotionCNN
from emotion_classification_mel.dataset import MelSpectrogramDataset
from tqdm import tqdm



# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 전체 파일 리스트
all_files = sorted(glob("./processed_data/mel_*.pt"))

# train/val split
train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)

# Dataloader 생성
train_dataset = MelSpectrogramDataset(train_files, use_path_list=True)
val_dataset = MelSpectrogramDataset(val_files, use_path_list=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 입력 데이터 확인
for mel, label in train_loader:
    print(mel.shape)  # (batch_size, 1, 128, 313)
    break

# 모델 및 학습 설정
model = EmotionCNN(num_classes=8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 훈련 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    train_loop = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{num_epochs}] Training", leave=False)

    for mel, label in train_loader:
        mel = mel.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(mel)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * mel.size(0)
        pred = output.argmax(dim=1)
        correct += (pred == label).sum().item()
        total += label.size(0)

        train_loop.set_postfix(loss=loss.item(), acc=correct/total)

    train_acc = correct / total
    train_loss /= total

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for mel, label in val_loader:
            mel = mel.to(device)
            label = label.to(device)

            output = model(mel)
            loss = criterion(output, label)

            val_loss += loss.item() * mel.size(0)
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

    val_acc = correct / total
    val_loss /= total

    print(f"[Epoch {epoch+1}] "
          f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} || "
          f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
