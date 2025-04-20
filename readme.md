# Mask R-CNN for Object Detection and Segmentation

## Overview

This repository contains a PyTorch implementation of Mask R-CNN for the Computational Methods for Single-Cell Biology [MA5617] course project (WS 2024-2025) at TUM. The implementation is based on references from [aotumanbiu/Pytorch-Mask-RCNN](https://github.com/aotumanbiu/Pytorch-Mask-RCNN) and PyTorch's official Mask R-CNN implementation.

## Features

- Complete Mask R-CNN implementation in PyTorch
- Modified RPN module with DropBlock for uncertainty quantification


## Dataset

The dataset in this repository includes only one example image per class due to data confidentiality requirements. The dataset is automatically split into training, validation, and test sets.

You can use your own data for training the Mask R-CNN model by replacing the files in the SPenn\SPenn folder. Make sure to:
1. Save images in the PNGImages folder
2. Save corresponding masks in the PedMasks folder
3. Keep consistent naming between image and mask files

If you need to adjust the class names and count, modify the CLASSES dictionary in the PennFudanDataset class:

```python
self.CLASSES = {
    'EOS': 1,
    'LYT': 2,
    'MON': 3,
    'MYO': 4,
    'NGB': 5,
    'NGS': 6,
    'EBO': 7,
    'BAS': 8
}  # Background is 0, target classes start from 1
```

Additionally, ensure your image files follow the naming format `XXX_mn`, where `XXX` represents the class name and `mn` is a numeric identifier.


## Implementation Details

### DropBlock in RPN

We've enhanced the Region Proposal Network (RPN) with a DropBlock mechanism to introduce uncertainty quantification. Unlike traditional dropout, DropBlock drops contiguous regions of feature map which is more effective for convolutional networks, especially in the context of object detection tasks.

## Usage

The notebook test.ipynb can be accessed and run directly in Google Colab:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uJGmcKVbAAeFquPzD65RAxW-25kme8YQ)
## References

- [aotumanbiu/Pytorch-Mask-RCNN](https://github.com/aotumanbiu/Pytorch-Mask-RCNN)
- [PyTorch official Mask R-CNN implementation](https://pytorch.org/vision/stable/models.html#mask-r-cnn)
