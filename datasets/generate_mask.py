"""
Generate masks from voxel and segmentation to skip rays
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from datasets import PhototourismDataset
from datasets.mask_utils import label_id_mapping_ade20k

root_dir = Path("/mnt/hdd/3d_recon/neural_recon_w/nepszinhaz_all_disk/split_0")
# root_dir = Path("/mnt/hdd/3d_recon/neural_recon_w/heritage-recon/brandenburg_gate")  # todo compare to orig
semantic_map_path = "semantic_maps"

kwargs = {
    "root_dir": str(root_dir),
    "img_downscale": 1,
    "val_num": 1,
    "semantic_map_path": semantic_map_path,
    "with_semantics": True,

}

val_dataset = PhototourismDataset(split="test_train", use_cache=False, **kwargs)

excluded_labels = ["desk",
                   "blanket",
                   "bed ",
                   "tray",
                   "computer",
                   "person",
                   "swimming pool",
                   "plate",
                   "basket",
                   "glass",
                   "car",
                   "minibike",
                   "food",
                   "land",
                   "bicycle",
                   ]

(root_dir / "masks").mkdir(exist_ok=True)
(root_dir / "masks_vis").mkdir(exist_ok=True)

for data in tqdm(val_dataset):
    # print(data)

    h, w = data['img_wh']
    shape = (w, h)
    voxel_mask = data['mask'].detach().cpu().numpy().reshape(shape)
    semantic_mask = data['semantics'].detach().cpu().numpy().reshape(shape)

    mask = voxel_mask
    for label_name in excluded_labels:
        mask[semantic_mask == label_id_mapping_ade20k[label_name]] = False

    # plt.imshow(mask, cmap='gray')
    # plt.show()

    image_name = data["image_name"]
    np.save((root_dir / "masks" / image_name).with_suffix('.npy'), mask)
    Image.fromarray(mask).save((root_dir / "masks_vis" / image_name).with_suffix('.png'))
