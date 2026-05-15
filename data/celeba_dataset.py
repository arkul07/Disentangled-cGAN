"""Custom Dataset for CelebA-style datasets.

The baseline uses the CelebA image folder plus the attribute CSV in `data/archive`.
Only the selected attributes are loaded, and labels are mapped from -1/1 to 0/1.
"""
from typing import List, Tuple, Optional
import csv
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CelebADataset(Dataset):
    """Dataset expecting an image folder and an attributes file.

    Selected attributes are specified in `selected_attrs`.
    """

    def __init__(
        self,
        images_dir: str,
        attr_file: str,
        selected_attrs: Optional[List[str]] = None,
        image_size: int = 64,
        max_images: Optional[int] = None,
        subset_seed: int = 42,
        training: bool = True,
    ):
        super().__init__()
        self.images_dir = images_dir
        self.attr_file = attr_file
        self.selected_attrs = selected_attrs or ["Smiling", "Male", "Eyeglasses", "Blond_Hair", "Young"]
        self.image_size = image_size
        self.max_images = max_images
        self.subset_seed = subset_seed
        self.training = training

        self.data: List[Tuple[str, List[int]]] = []
        self._parse_attributes()

        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
        if training:
            transform_list.insert(1, transforms.RandomHorizontalFlip())
        self.transform = transforms.Compose(transform_list)

    def _parse_attributes(self) -> None:
        if not os.path.exists(self.attr_file):
            raise FileNotFoundError(f"Attributes file not found: {self.attr_file}")

        with open(self.attr_file, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            if not header:
                raise ValueError(f"Could not read header from attribute file: {self.attr_file}")

            image_key = header[0]
            missing = [attr for attr in self.selected_attrs if attr not in header]
            if missing:
                raise ValueError(
                    "Missing selected attributes in CSV header: " + ", ".join(missing)
                )

            for row in reader:
                fname = row[image_key]
                attrs = [1 if row[attr].strip() == "1" else 0 for attr in self.selected_attrs]
                self.data.append((fname, attrs))

        if self.max_images is not None and len(self.data) > self.max_images:
            rng = random.Random(self.subset_seed)
            self.data = rng.sample(self.data, self.max_images)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        fname, attrs = self.data[idx]
        img_path = os.path.join(self.images_dir, fname)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        attrs = torch.tensor(attrs, dtype=torch.float32)
        return img, attrs
