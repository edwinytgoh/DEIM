# SPDX-License-Identifier: MIT
"""
DEIM adapter for the SARTileDetectionBase dataset.
This class bridges the generic SAR dataset with the specific requirements of the DEIM training and evaluation pipeline.
"""
from __future__ import annotations

from typing import Any, Tuple

import torch
from torchvision.ops import box_convert

from engine.core import register
from engine.data._misc import convert_to_tv_tensor
from engine.data.dataset._dataset import DetDataset
from engine.data.dataset.sar_dataset import (
    SARTileDetectionBase,
)  # Assuming both files are in the same directory


@register()
class DEIM_SAR_Dataset(SARTileDetectionBase, DetDataset):
    __inject__ = ["transforms"]

    def __init__(self, img_folder: str, ann_file: str, transforms: Any = None, **kw):
        # Pass `image_input_path` to the base class for its file discovery logic
        super().__init__(ann_file=ann_file, image_input_path=img_folder, **kw)
        self.transforms = transforms
        self._epoch = -1

    def set_epoch(self, epoch: int) -> None:
        """Required by the DEIM training loop to set the current epoch."""
        self._epoch = epoch

    @property
    def epoch(self) -> int:
        """Required by DEIM transforms for epoch-based policies."""
        return self._epoch

    def load_item(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Loads the raw data from the base dataset and formats it for DEIM.
        This method should NOT apply augmentations; that is handled by __getitem__.
        """
        # 1. Get the raw PIL image and target from the base class
        img, target = SARTileDetectionBase.__getitem__(self, idx)
        w, h = img.size

        # 2. Convert bounding boxes from [x,y,w,h] to [x1,y1,x2,y2]
        xywh = target.pop("boxes")
        xyxy = (
            box_convert(xywh, "xywh", "xyxy")
            if xywh.shape[0] > 0
            else torch.empty((0, 4))
        )

        # 3. Build the final target dictionary with all keys expected by DEIM
        target = {
            "boxes": convert_to_tv_tensor(xyxy, "boxes", spatial_size=(h, w)),
            "labels": target["labels"],
            "area": (xywh[:, 2] * xywh[:, 3]) if xywh.shape[0] > 0 else torch.empty(0),
            "iscrowd": torch.zeros(xywh.size(0), dtype=torch.int64),
            "orig_size": torch.tensor([h, w]),
            "image_id": target["image_id"],
        }
        return img, target

    def __getitem__(self, idx: int):
        """
        This is the main entry point for the DataLoader. It follows the DEIM
        `DetDataset` pattern: load raw item, then apply transforms.
        """
        # Call our overridden load_item to get the DEIM-formatted data
        img, target = self.load_item(idx)

        # Apply transforms as defined in the YAML config
        if self.transforms:
            img, target, _ = self.transforms(img, target, self)

        return img, target
