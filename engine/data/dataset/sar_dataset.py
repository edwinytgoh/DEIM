# SPDX-License-Identifier: MIT
"""
SARTileDetectionBase
--------------------
COCO-compliant sliding-window loader for Sentinel-1 SAR GeoTIFFs.

Key features
============
1. **Flexible image discovery**
   • *image_input_path = None*  → read every `file_name` in the COCO JSON.
   • *image_input_path = "/folder"* → scan that folder for .tif(f) files whose basenames appear in the JSON.
   • *image_input_path = "/path/to/scene.tif"* → process exactly that file (must be listed in JSON).
   Relative `file_name`s are automatically resolved against the JSON's directory.

2. **Direct CocoDetection inheritance** - lets any library calling
   `get_coco_api_from_dataset()` short-circuit instantly.

3. **SAR-specific guarantees** - nodata == 0 preserved; mono band replicated
   to 3-channel RGB so pretrained backbones work out-of-the-box.
"""

from __future__ import annotations

import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio import windows
from torch.utils.data import Dataset, get_worker_info
from torchvision.datasets import CocoDetection
from tqdm.auto import tqdm

MAX_NODATA_FRACTION = 0.4


# ───────────────────────── helper functions ──────────────────────────
def _nodata_val(src: rasterio.DatasetReader) -> float:
    if src.nodata is not None:
        return src.nodata
    sample = src.read(
        1, window=windows.Window(0, 0, 64, 64), boundless=True, fill_value=0
    )
    return -9999.0 if np.any(sample < -9000) else 0.0


def _process_scene(
    tif_path: str, win: int, stride: int, ann_file: str, coco_dataset: dict
):
    """
    Stand-alone so it can run in a worker process.
    Re-creates only the cheap state it needs; avoids pickling the whole Dataset.
    """

    from pycocotools.coco import COCO

    # light-weight re-construction
    coco = COCO()
    coco.dataset = coco_dataset
    coco.createIndex()

    # empty shell of the base dataset
    dataset = object.__new__(SARTileDetectionBase)  # no __init__ called
    dataset.win = win
    dataset.stride = stride
    dataset.coco = coco
    dataset.ann_file = ann_file

    # 1) init raster (sets src & nodata_val)
    dataset._initialize_raster_properties(Path(tif_path))

    # 2) compute windows (via your optimized numpy-based logic)
    initial_windows = dataset._compute_initial_windows_optimized_numpy()

    # 3) load JSON annotations & image info
    basename = dataset._load_coco_annotations_and_image_info(Path(tif_path))
    if basename is None:
        return [], [], [], []  # no matching COCO entry

    # 4) filter windows+labels via your existing logic
    kept_windows, kept_labels = dataset._build_filtered_labels_and_windows(
        initial_windows
    )

    # repeat paths & image-ids
    kept_paths = [tif_path] * len(kept_windows)
    kept_ids = [dataset.img_id] * len(kept_windows)

    return kept_windows, kept_labels, kept_paths, kept_ids

    # return out_win, out_lbl, [tif_path] * len(out_win), [img_id] * len(out_win)


# ─────────────────────────── dataset class ───────────────────────────
class SARTileDetectionBase(CocoDetection):
    def __init__(
        self,
        ann_file: str,
        image_input_path: Optional[str] = None,
        *,
        window_size: int = 640,
        stride: int = 320,
        return_masks: bool = False,
    ) -> None:
        # initialise CocoDetection ─ root may be empty if we search by ourselves
        super().__init__(image_input_path or "", ann_file)

        self.win = window_size
        self.stride = stride
        self.ann_file = ann_file

        # —— discover GeoTIFF paths (three modes) ————————————————
        if image_input_path is None:
            # derive absolute paths from JSON entries
            tif_paths = []
            for img_info in self.coco.dataset.get("images", []):
                fname = img_info["file_name"]
                p = Path(fname)
                if not p.is_absolute():
                    p = Path(ann_file).parent / p.name
                if p.exists():
                    tif_paths.append(str(p))
                else:
                    print(f"Warning: {fname} not found; skipped.")
        else:
            p = Path(image_input_path)
            if p.is_dir():
                all_tifs = sorted(p.glob("*.tif")) + sorted(p.glob("*.tiff"))
                names_in_json = {
                    Path(im["file_name"]).name for im in self.coco.dataset["images"]
                }
                tif_paths = [str(tp) for tp in all_tifs if tp.name in names_in_json]
                missing = [
                    n for n in names_in_json if n not in {tp.name for tp in all_tifs}
                ]
                if missing:
                    print(f"Warning: these JSON images missing in {p}: {missing}")
            else:
                if not p.exists():
                    raise FileNotFoundError(f"{p} does not exist.")
                if p.name not in {
                    Path(im["file_name"]).name for im in self.coco.dataset["images"]
                }:
                    raise ValueError(f"{p.name} not listed in {ann_file}.")
                tif_paths = [str(p)]

        if not tif_paths:
            raise RuntimeError("No GeoTIFFs matched the COCO annotation set.")

        # —— tile enumeration (parallel) ————————————————
        self._windows, self._labels, self._paths, self._img_ids = [], [], [], []
        with ProcessPoolExecutor(max_workers=min(os.cpu_count() // 2 or 1, 32)) as pool:
            futures = [
                pool.submit(
                    _process_scene,
                    tp,
                    self.win,
                    self.stride,
                    ann_file,
                    self.coco.dataset,
                )
                for tp in tif_paths
            ]
            for f in tqdm(
                as_completed(futures), desc="Indexing SAR tiles", total=len(futures)
            ):
                w, l, pth, ids = f.result()
                self._windows.extend(w)
                self._labels.extend(l)
                self._paths.extend(pth)
                self._img_ids.extend(ids)

        if not self._windows:
            raise RuntimeError("No valid tiles found across provided scenes.")

        # —— misc helpers ————————————————————————————
        self.n = len(self._windows)
        self.indices = np.arange(self.n, dtype=np.int32)
        self._src_cache: Dict[int, Dict[str, Tuple[rasterio.DatasetReader, float]]] = {}

        cat_ids = self.coco.getCatIds()
        self.names = [cat["name"] for cat in self.coco.loadCats(cat_ids)]

        print(
            f"""
        Dataset Summary:
        - Total .tif files processed: {len(set(self._img_ids))}
        - Total valid windows: {self.n}
        - Window size: {self.win}x{self.win}
        - Stride: {self.stride}
        - Number of categories: {len(self.names)}
        - Category names: {self.names}
        """
        )

    # ─────────────────────── helper methods (copied from original) ──────────────────────
    def _initialize_raster_properties(self, geotiff_path: Path) -> None:
        """Open a GeoTIFF once and cache its nodata value."""
        self.src = rasterio.open(str(geotiff_path))
        self.nodata_val = _nodata_val(self.src)

    def _compute_initial_windows_optimized_numpy(self) -> List[windows.Window]:
        """
        Fast tiling using an integral image over the nodata mask.
        Returns every window whose nodata-pixel fraction < MAX_NODATA_FRACTION.
        """
        h, w = self.src.height, self.src.width
        win, stride = self.win, self.stride
        if h < win or w < win:
            return []

        mask = (self.src.read(1) == self.nodata_val).astype(np.uint32)
        S = np.pad(mask.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

        ys = np.arange(0, h - win + 1, stride)
        xs = np.arange(0, w - win + 1, stride)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        y1, x1 = yy + win, xx + win
        nodata_cnt = S[yy, xx] + S[y1, x1] - S[y1, xx] - S[yy, x1]
        keep = nodata_cnt.astype(np.float32) / (win**2) < MAX_NODATA_FRACTION

        return [
            windows.Window(int(x), int(y), win, win) for y, x in zip(yy[keep], xx[keep])
        ]

    def _load_coco_annotations_and_image_info(
        self, geotiff_path: Path
    ) -> Optional[str]:
        """
        Fetch the COCO `image_id` and its annotations that correspond to this GeoTIFF.
        """
        basename = geotiff_path.name
        img_ids = [
            img["id"]
            for img in self.coco.dataset["images"]
            if Path(img["file_name"]).name == basename
        ]
        if not img_ids:
            return None

        self.img_id = img_ids[0]
        self.anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[self.img_id]))
        return basename

    def _build_filtered_labels_and_windows(
        self, initial_windows: List[windows.Window]
    ) -> Tuple[List[windows.Window], List[np.ndarray]]:
        """
        Keep only those windows that fully contain at least one annotation.
        Returns the surviving windows and their per-window label arrays:
        [[class, x_rel, y_rel, w, h], …] in **pixel** coordinates.
        """
        kept_windows, kept_labels = [], []
        for win in initial_windows:
            x0_win, y0_win = win.col_off, win.row_off
            boxes_in_window = []
            for ann in self.anns:
                x_abs, y_abs, w_abs, h_abs = ann["bbox"]
                if (
                    x_abs >= x0_win
                    and y_abs >= y0_win
                    and x_abs + w_abs <= x0_win + self.win
                    and y_abs + h_abs <= y0_win + self.win
                ):
                    boxes_in_window.append(
                        [
                            ann["category_id"] - 1,  # zero-based
                            x_abs - x0_win,
                            y_abs - y0_win,
                            w_abs,
                            h_abs,
                        ]
                    )

            if boxes_in_window:
                kept_windows.append(win)
                kept_labels.append(np.asarray(boxes_in_window, np.float32))

        return kept_windows, kept_labels

    # ——————— Dataset API —————————
    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        ridx = self.indices[idx]
        path, win = self._paths[ridx], self._windows[ridx]

        src, nodata = self._get_src(path)
        tile = src.read(1, window=win, boundless=True, fill_value=nodata).astype(
            np.float32
        )
        tile[tile == nodata] = 0.0

        # replicate mono → RGB for ImageNet-pretrained backbones
        img = Image.fromarray(
            (np.repeat(tile[None, ...], 3, 0).transpose(1, 2, 0) * 255).astype(np.uint8)
        )

        lbl = torch.from_numpy(self._labels[ridx])
        target = {
            "boxes": lbl[:, 1:],
            "labels": lbl[:, 0].long(),
            "image_id": torch.tensor([self._img_ids[ridx]]),
        }
        return img, target

    # ——————— helpers ———————————
    def _get_src(self, path: str) -> Tuple[rasterio.DatasetReader, float]:
        wid = get_worker_info().id if get_worker_info() else 0
        if wid not in self._src_cache:
            self._src_cache[wid] = {}
        if path not in self._src_cache[wid]:
            ds = rasterio.open(path)
            self._src_cache[wid][path] = (ds, _nodata_val(ds))
        return self._src_cache[wid][path]
