import os
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.functional as F_audio
from tqdm import tqdm

emotion_map = {
    '01': 0, '02': 1, '03': 2, '04': 3,
    '05': 4, '06': 5, '07': 6, '08': 7
}

root_dir = './Audio_Speech_Actors_01-24'
save_dir = './processed_data'
os.makedirs(save_dir, exist_ok=True)

sample_rate = 16000
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=128,
    n_fft=1024,
    hop_length=256
)

def fix_mel_spec(spec, target_shape=(128, 128)):
    # spec shape: (1, n_mels, time) 혹은 (n_mels, time)
    if spec.dim() == 3:
        spec = spec.squeeze(0)  # (n_mels, time)

    # 안전하게 unpack
    n_mels, time = spec.shape[-2:]

    if time < target_shape[1]:
        spec = F.pad(spec, (0, target_shape[1] - time))
    elif time > target_shape[1]:
        spec = spec[:, :target_shape[1]]

    return spec  # shape: (n_mels, time)


count = 0
for actor in sorted(os.listdir(root_dir)):
    actor_path = os.path.join(root_dir, actor)
    if not os.path.isdir(actor_path): continue

    for fname in os.listdir(actor_path):
        if not fname.endswith('.wav'):
            continue

        parts = fname.split('-')
        if len(parts) < 7:
            continue

        emotion_code = parts[2]
        label = emotion_map.get(emotion_code)
        if label is None:
            continue

        path = os.path.join(actor_path, fname)
        wav, sr = torchaudio.load(path)

        if sr != sample_rate:
            wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)

        max_len = sample_rate * 5
        if wav.shape[1] < max_len:
            wav = F.pad(wav, (0, max_len - wav.shape[1]))
        else:
            wav = wav[:, :max_len]

        mel = mel_transform(wav).squeeze(0)
        mel = F_audio.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=0.0)  # log scale 변환
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)  # 정규화는 유지

        save_path = os.path.join(save_dir, f'mel_{count:04d}_{label}.pt')
        torch.save(mel, save_path)
        count += 1
