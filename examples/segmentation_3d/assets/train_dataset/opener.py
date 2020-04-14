import os
from glob import glob

from monai.data import NiftiDataset
from monai.transforms import Compose, AddChannel, ScaleIntensity, ToTensor, RandSpatialCrop, RandRotate90
import torch
from torch.utils.data import DataLoader

import substratools as tools

class MonaiTrainOpener(tools.Opener):
    device = torch.device("cpu")

    def _get_loader(self, folders):
        images = []
        segs = []
        for folder in folders:
            images += glob(os.path.join(folder, '*_im.nii.gz'))
            segs += glob(os.path.join(folder, '*_seg.nii.gz'))
        images = sorted(images)
        segs = sorted(segs)

        imtrans = Compose([
            ScaleIntensity(),
            AddChannel(),
            RandSpatialCrop((96, 96, 96), random_size=False),
            RandRotate90(prob=0.5, spatial_axes=(0, 2)),
            ToTensor()
        ])
        segtrans = Compose([
            AddChannel(),
            RandSpatialCrop((96, 96, 96), random_size=False),
            RandRotate90(prob=0.5, spatial_axes=(0, 2)),
            ToTensor()
        ])

        ds = NiftiDataset(images, segs, transform=imtrans, seg_transform=segtrans)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())

        return loader

    def _get_X_iterator(self, loader):
        for data in loader:
            yield data[0].to(self.device)

    def _get_y_iterator(self, loader):
        for data in loader:
            yield data[1].to(self.device)

    def get_X(self, folders):
        loader = self._get_loader(folders)
        return self._get_X_iterator(loader)

    def get_y(self, folders):
        loader = self._get_loader(folders)
        return self._get_y_iterator(loader)

    def save_predictions(self, y_pred, path):
        raise NotImplementedError

    def get_predictions(self, path):
        raise NotImplementedError

    def fake_X(self):
        raise NotImplementedError

    def fake_y(self):
        raise NotImplementedError
