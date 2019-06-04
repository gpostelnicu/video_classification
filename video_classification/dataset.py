from typing import List

import os

import torch
from PIL import Image
from torch.utils import data


class VideoFramesDataset(data.Dataset):
    def __init__(self, base_dir: str, folders: List[str],
                 labels: List[int], num_frames: int, transform=None):
        self.base_dir = base_dir
        self.folders = folders
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform
        assert len(self.folders) == len(self.labels)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        folder = self.folders[index]
        label = self.labels[index]

        x = self._read_images(folder)
        y = torch.LongTensor([label - 1])  # Make output label 0-based.
        return x, y

    def _read_images(self, folder: str):
        x = []
        for i in range(self.num_frames):
            # Frame indexing starts at 1.
            fname = os.path.join(self.base_dir, folder, 'frame{:06d}.jpg'.format(i + 1))
            im = Image.open(fname)
            # TODO: make random components of transform constant for frame.
            if self.transform is not None:
                im = self.transform(im)
            x.append(im)

        x = torch.stack(x, dim=0)
        return x


def read_list_file(list_fname: str):
    """
    read_list_file reads an input text file of clip label data.
    """
    clips = []
    labels = []

    with open(list_fname) as fh:
        for line in fh.readlines():
            parts = line.split()
            clips.append(parts[0])
            labels.append(int(parts[1]))

    return clips, labels

