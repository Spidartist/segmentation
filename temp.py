import mmcv
import mmengine.fileio as fileio
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt

img_bytes = fileio.get("/mnt/tuyenld/data/endoscopy/public_dataset/TrainDataset/masks/6.png", backend_args=None)
gt_semantic_seg = mmcv.imfrombytes(
    img_bytes, flag='unchanged',
    backend='pillow').squeeze().astype(np.uint8)
np.set_printoptions(threshold=sys.maxsize)
print(gt_semantic_seg[:,40])
plt.imshow(gt_semantic_seg)
plt.show()
plt.savefig('myfilename.png')