import random
from typing import List

import os

import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader


class SampledDataset(data.Dataset):
    def __init__(self, ds, sample_size):
        if sample_size < 0 or sample_size > len(ds):
            raise ValueError("Illegal value sample_size={} should be in range [0, {}]".format(
                sample_size, len(ds)))
        self.indices = random.sample(range(len(ds)), sample_size)
        self.ds = ds

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.ds[self.indices[item]]


def ds_islice(ds, stop, shuffle=False):
    class Ds(data.Dataset):
        def __init__(self, ds, stop):
            self.ds = ds
            self.stop = stop

        def __len__(self):
            return min(len(self.ds), self.stop)

        def __getitem__(self, item):
            return self.ds[item]

    if stop < 0:  # no-op
        return ds
    return Ds(ds, stop)


class VideoFramesDataset(data.Dataset):
    def __init__(self, base_dir: str, folders: List[str],
                 labels: List[int], num_files: int,
                 folder_frames: List[int], transform=None):
        self.base_dir = base_dir
        self.folders = folders
        self.labels = labels
        self.num_files = num_files
        self.folder_frames = folder_frames
        self.transform = transform
        assert len(self.folders) == len(self.labels)

    @staticmethod
    def from_list(base_dir: str, fname: str, num_frames: int, transform):
        """
        from_list creates a dataset from a file containing a space-separated list of folder,labelIndex,numImages.

        Args:
            fname: filename containing the list
            num_frames: number of frames to parse by folder
            transform: optional transform to apply to all images.
        Returns:
            VideoFramesDataset: created dataset.
        """
        folders = []
        labels = []
        folder_frames = []
        with open(fname) as fh:
            for line in fh.readlines():
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                if len(parts) != 3:
                    raise RuntimeError('Error parsing line {}'.format(line))
                folders.append(parts[0])
                labels.append(int(parts[1]))
                folder_frames.append(int(parts[2]))
        return VideoFramesDataset(base_dir=base_dir,
                                  folders=folders, labels=labels, folder_frames=folder_frames,
                                  num_files=num_frames, transform=transform)

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, index):
        folder = self.folders[index]
        label = self.labels[index]
        folder_frames = self.folder_frames[index]

        x = self._read_images(folder, folder_frames)
        y = torch.LongTensor([label - 1])  # Make output label 0-based.
        return x, y

    def _read_images(self, folder: str, folder_frames: int):
        x = []
        for i in range(min(self.num_files, folder_frames)):
            # Frame indexing starts at 1.
            fname = os.path.join(self.base_dir, folder, 'frame{:06d}.jpg'.format(i + 1))
            im = Image.open(fname)
            # TODO: make random components of transform constant for frame.
            if self.transform is not None:
                im = self.transform(im)
            x.append(im)

        x = torch.stack(x, dim=0)
        return x


def collate_fn(data):
    # Sort the data by the clip length (descending order).
    data.sort(key=lambda x: x[0].size(0), reverse=True)
    clips, labels = zip(*data)

    # Merge images (from tuple of 4D tensor to 5D tensor).
    lengths = [x.size(0) for x in clips]
    clips = torch.nn.utils.rnn.pad_sequence(clips)
    labels = torch.stack(labels, 0)

    return clips, labels, lengths


class LoaderMaker(object):
    def __init__(self, base_dir, batch_size, num_workers):
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def make(self, list_file, shuffle, transform):
        pass


class SimpleLoaderMaker(LoaderMaker):
    def __init__(self, base_dir, batch_size, num_workers, num_frames):
        super().__init__(base_dir, batch_size, num_workers)
        self.num_frames = num_frames

    def make(self, list_file, shuffle, transform):
        ds = VideoFramesDataset.from_list(
            base_dir=self.base_dir, fname=list_file, num_frames=self.num_frames, transform=transform
        )
        loader = DataLoader(
            dataset=ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_fn
        )
        return loader
