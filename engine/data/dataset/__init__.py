"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# from ._dataset import DetDataset
from .coco_dataset import (
    CocoDetection,
    mscoco_category2label,
    mscoco_category2name,
    mscoco_label2category,
)
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
from .deim_sar_dataset import *
from .sar_dataset import *
from .voc_detection import VOCDetection
from .voc_eval import VOCEvaluator
