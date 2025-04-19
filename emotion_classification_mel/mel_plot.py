import torch
import matplotlib.pyplot as plt
import torchaudio.transforms

import os

def plot_mel_from_file(filepath):
    # 파일 이름에서 라벨 추출
    basename = os.path.basename(filepath)
    label_str = basename.split('_')[-1].replace(".pt", "")
    label = int(label_str)

    # Mel 불러오기
    mel = torch.load(filepath)  # shape: (1, 128, 128)
    mel = mel.squeeze().numpy()  # (128, 128)

    plt.figure(figsize=(8, 4))
    plt.imshow(mel, origin="lower", aspect="auto", cmap="magma")
    plt.title(f"Mel Spectrogram (Label: {label})")
    plt.xlabel("Time")
    plt.ylabel("Mel Frequency")
    plt.colorbar(label="Amplitude (dB)")
    plt.tight_layout()
    plt.show()

# 예시 경로: ./processed_data/mel_0000_label3.pt
plot_mel_from_file("./processed_data/mel_0034_4.pt")
