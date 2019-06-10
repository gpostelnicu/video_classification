from collections import Counter, namedtuple
import random
from typing import List

import os

import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import DataLoader


Sample = namedtuple('Sample', 'folder label start stop step weight'.split())


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


def ds_islice(ds, stop):
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
    def __init__(self, base_dir: str, samples: List[Sample], transform=None):
        self.base_dir = base_dir
        self.samples = samples
        self.transform = transform

    @staticmethod
    def from_list(base_dir: str, fname: str, max_samples_per_video: int, num_frames: int, transform):
        """
        from_list creates a dataset from a file containing a space-separated list of folder,labelIndex,numImages.

        Args:
            fname: filename containing the list
            max_samples_per_video: maximum number of samples to generate from a single video
            num_frames: number of frames to parse by folder
            transform: optional transform to apply to all images.
        Returns:
            VideoFramesDataset: created dataset.
        """
        samples = []
        with open(fname) as fh:
            for line in fh.readlines():
                vid_samples = []
                parts = line.strip().split()
                if len(parts) == 0:
                    continue
                if len(parts) != 3:
                    raise RuntimeError('Error parsing line {}'.format(line))
                folder = parts[0]
                label = int(parts[1])
                frames = int(parts[2])
                # Select random starting point
                for clip in range(min(frames - num_frames, max_samples_per_video)):
                    step = random.randint(1, max(1, min(10, frames // num_frames)))
                    start = random.randint(0, max(0, frames - num_frames * step))
                    stop = random.randint(num_frames, min(2 * num_frames, (frames - start) // step)) + start
                    vid_samples.append(
                        Sample(folder=folder, label=label, start=start, stop=stop, step=step, weight=1.))
                counter = Counter(vid_samples)
                for s, n in counter.items():
                    s.weight = n
                    samples.append(s._replace(weight=n))
        return VideoFramesDataset(base_dir=base_dir, samples=samples, transform=transform)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = sample.label

        x = self._read_images(sample.folder, sample.start, sample.stop, sample.step)
        y = torch.LongTensor([label - 1])  # Make output label 0-based.
        w = torch.Tensor([sample.weight])
        return x, y, w

    def _read_images(self, folder: str, start: int, stop: int, step: int):
        x = []
        for i in range(start, stop, step):
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
    clips, labels, weights = zip(*data)

    # Merge images (from tuple of 4D tensor to 5D tensor).
    lengths = [x.size(0) for x in clips]
    clips = torch.nn.utils.rnn.pad_sequence(clips, batch_first=True)
    labels = torch.stack(labels, 0)
    weights = torch.stack(weights, 0).squeeze()

    return clips, labels, lengths, weights


def loader_from_dataset(dataset: data.Dataset, batch_size: int, shuffle: bool, num_workers: int):
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    return loader
