import os
import torch
from torch.utils.data import Dataset

class MelSpectrogramDataset(Dataset):
    def __init__(self, data, use_path_list=False):
        """
        Args:
            data (str or list): 폴더 경로이거나, .pt 파일 경로 리스트
            use_path_list (bool): data가 파일 리스트인지 여부
        """
        if use_path_list:
            self.file_list = data
        else:
            self.file_list = [
                os.path.join(data, fname)
                for fname in os.listdir(data)
                if fname.endswith('.pt') and 'mel' in fname
            ]
        self.file_list.sort()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        mel = torch.load(path)

        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # (1, H, W)
        elif mel.shape[0] != 1:
            mel = mel[:1, :, :]  # 첫 채널만 가져옴

        # 파일명 예시: mel_00003_4.pt → label은 맨 끝
        label = int(os.path.basename(path).split('_')[-1].replace(".pt", ""))
        
        return mel, label
