"""Wildlife image loading and shared preprocessing for the CNN comparison."""
import hashlib
import io
import json
import os
import random
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2, InterpolationMode

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SOURCE_METADATA_SHA256 = "da83480cae34bbd3ec275461b560ce9343fe88b526844afbc88857d7c751d566"


def safe_relative(name):
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and '..' not in path.parts and ':' not in name and '\\' not in name


def rgb_image(image):
    image = ImageOps.exif_transpose(image)
    if 'A' in image.getbands() or (image.mode == 'P' and 'transparency' in image.info):
        rgba = image.convert('RGBA')
        background = Image.new('RGBA', rgba.size, (127, 127, 127, 255))
        return Image.alpha_composite(background, rgba).convert('RGB')
    return image.convert('RGB')


class CanonicalRGBResize:
    def __call__(self, image):
        image = rgb_image(image)
        w, h = image.size
        size = (max(256, round(w * 256 / min(w, h))), max(256, round(h * 256 / min(w, h))))
        if max(size) > 16384:
            raise ValueError('Image aspect ratio exceeds the preparation limit')
        return image.resize(size, Image.Resampling.BICUBIC) if image.size != size else image


class Letterbox224:
    def __call__(self, image):
        return ImageOps.pad(image, (224, 224), method=Image.Resampling.BICUBIC,
                            color=(124, 116, 104), centering=(0.5, 0.5))


def build_transform(training=False):
    steps = [CanonicalRGBResize(), Letterbox224()]
    if training:
        steps += [v2.RandomCrop(224, padding=8, fill=(124, 116, 104)),
                  v2.RandomHorizontalFlip(p=0.5),
                  v2.RandomRotation(10, interpolation=InterpolationMode.BILINEAR,
                                    expand=True, fill=(124, 116, 104)),
                  Letterbox224(),
                  v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)]
    steps += [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
    return v2.Compose(steps)


class WildlifeDataset(Dataset):
    def __init__(self, manifest, image_root=None, archive_path=None, training=False):
        if image_root is None and archive_path is None:
            raise ValueError('Provide image_root (the folder containing data/cache) or archive_path')
        columns = ['path', 'source_path', 'class_id', 'sample_id', 'split', 'file_sha256', 'cache_sha256']
        self.frame = pd.read_csv(manifest, usecols=columns, keep_default_na=False)
        if self.frame.empty:
            raise ValueError('Empty split file')
        if training and (set(self.frame.split) != {'train'} or self.frame.class_id.lt(0).any()):
            raise ValueError('Training requires train.csv.gz with known class labels only')
        self.image_root = Path(image_root).resolve() if image_root else None
        self.archive_path = str(Path(archive_path).resolve()) if archive_path else None
        self.transform = build_transform(training)
        self._archive = None
        self._archive_pid = None
        if not all(safe_relative(p) for p in self.frame.path) or not all(safe_relative(p) for p in self.frame.source_path):
            raise ValueError('Unsafe image path in split file')

    def __len__(self):
        return len(self.frame)

    def _open_archive(self):
        # Each DataLoader worker owns its ZIP handle, including Windows spawn workers.
        if self._archive is None or self._archive_pid != os.getpid():
            self.close()
            archive = ZipFile(self.archive_path)
            try:
                if hashlib.sha256(archive.read('metadata.csv')).hexdigest() != SOURCE_METADATA_SHA256:
                    raise ValueError('This ZIP is a different dataset release from the fixed splits')
            except Exception:
                archive.close()
                raise
            self._archive = archive
            self._archive_pid = os.getpid()
        return self._archive

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = (self.image_root / row.path).resolve() if self.image_root else None
        if path is not None and not path.is_relative_to(self.image_root):
            raise ValueError('Image path resolves outside image_root')
        if path is not None and path.is_file():
            data, expected = path.read_bytes(), row.cache_sha256
        elif self.archive_path:
            data, expected = self._open_archive().read(row.source_path), row.file_sha256
        else:
            raise FileNotFoundError(f'Missing prepared image: {path}. Supply the prepared cache or archive_path.')
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Image fingerprint mismatch: {row.sample_id}')
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            tensor = self.transform(image)
        return tensor, int(row.class_id), row.sample_id

    def close(self):
        if self._archive is not None:
            self._archive.close()
            self._archive = None
            self._archive_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_archive'] = state['_archive_pid'] = None
        return state


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loaders(project_root, image_root=None, archive_path=None, batch_size=32, num_workers=0, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    directory = Path(project_root) / 'data/prepared/open_set'
    class_map = json.loads((directory / 'class_to_idx.json').read_text())
    loaders = {}
    for split in ['train', 'val']:
        dataset = WildlifeDataset(directory / f'{split}.csv.gz', image_root, archive_path, training=split == 'train')
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=split == 'train',
                                   num_workers=num_workers, worker_init_fn=seed_worker,
                                   generator=torch.Generator().manual_seed(seed),
                                   pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0)
    return loaders, class_map
