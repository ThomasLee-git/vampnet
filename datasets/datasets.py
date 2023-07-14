import numpy as np
import torch
from torch.utils.data import Dataset


class PreprocessedMusicLmDataset(Dataset):
    def __init__(
        self, npz_path: str, target_frame_len: int, target_num_quantizers: int
    ) -> None:
        super().__init__()
        npz_data = np.load(npz_path)
        self.data = npz_data["data"]
        self.cumsum_addr = npz_data["cumsum_addr"]
        self.target_frame_len = target_frame_len
        self.target_num_quantizers = target_num_quantizers

    def __getitem__(self, index):
        while True:
            start = 0 if index == 0 else self.cumsum_addr[index - 1]
            end = self.cumsum_addr[index]
            tmp_data = self.data[start:end]
            # sampel frame idx
            num_frames, num_quantizers = tmp_data.shape
            if num_frames < self.target_frame_len:
                new_index = torch.randint(0, len(self), (1,)).item()
                err_msg = (
                    f"short sequence found at {index=} {num_frames=} try {new_index=}"
                )
                print(err_msg)
                index = new_index
            else:
                break
        frame_idx = 0
        if num_frames > self.target_frame_len:
            frame_upper_bound = num_frames - self.target_frame_len
            frame_idx = torch.randint(0, frame_upper_bound, (1,))
        # get data
        result = torch.from_numpy(
            tmp_data[
                frame_idx : frame_idx + self.target_frame_len,
                : self.target_num_quantizers,
            ]
        ).long()
        return result

    def __len__(self):
        return len(self.cumsum_addr)
