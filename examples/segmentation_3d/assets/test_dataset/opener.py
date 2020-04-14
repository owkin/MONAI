from glob import glob
import os

from monai.data import NiftiDataset, NiftiSaver, load_nifti
from monai.transforms import Compose, AddChannel, ScaleIntensity, ToTensor, AsChannelFirst
import torch
from torch.utils.data import DataLoader

import substratools as tools

class MonaiTestOpener(tools.Opener):
    device = torch.device("cpu")

    def _get_loader(self, folders):
        images = []
        segs = []
        for folder in folders:
            images += glob(os.path.join(folder, '*_im.nii.gz'))
            segs += glob(os.path.join(folder, '*_seg.nii.gz'))
        images = sorted(images)
        segs = sorted(segs)

        imtrans = Compose([ScaleIntensity(), AddChannel(), ToTensor()])
        segtrans = Compose([AddChannel(), ToTensor()])

        ds = NiftiDataset(images, segs, transform=imtrans, seg_transform=segtrans, image_only=False)
        loader = DataLoader(ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())

        return loader

    def _get_X_iterator(self, loader):
        for data in loader:
            yield (data[0].to(self.device), data[2])

    def _get_y_iterator(self, loader):
        for data in loader:
            yield (data[1].to(self.device), data[2])

    def get_X(self, folders):
        loader = self._get_loader(folders)
        return self._get_X_iterator(loader)

    def get_y(self, folders):
        loader = self._get_loader(folders)
        return self._get_y_iterator(loader)

    def save_predictions(self, y_pred, path):
        saver = NiftiSaver(output_dir=path)
        for outputs, metadata in y_pred:
            saver.save_batch(outputs, metadata)

    def _get_predictions_iterator(self, segs):
        segtrans = Compose([AsChannelFirst(), ToTensor(), AddChannel()])
        for seg in segs:
            data = load_nifti(seg, image_only=False)
            yield(segtrans(data[0]), data[1])

    def get_predictions(self, path):
        segs = sorted(glob(os.path.join(path, '*/*_seg.nii.gz')))
        return self._get_predictions_iterator(segs)

    def fake_X(self):
        raise NotImplementedError

    def fake_y(self):
        raise NotImplementedError
