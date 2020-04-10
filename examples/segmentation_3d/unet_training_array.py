# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import tempfile
import shutil
from glob import glob
import logging
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import NiftiDataset, create_test_image_3d, sliding_window_inference
from monai.transforms import Compose, AddChannel, ScaleIntensity, RandUniformPatch, RandRotate90, ToTensor
from monai.metrics import compute_meandice
from monai.visualize.img2tensorboard import plot_2d_or_3d_image

monai.config.print_config()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# create a temporary directory and 40 random image, mask paris
tempdir = tempfile.mkdtemp()
print('generating synthetic data to {} (this may take a while)'.format(tempdir))
for i in range(40):
    im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1)

    n = nib.Nifti1Image(im, np.eye(4))
    nib.save(n, os.path.join(tempdir, 'im%i.nii.gz' % i))

    n = nib.Nifti1Image(seg, np.eye(4))
    nib.save(n, os.path.join(tempdir, 'seg%i.nii.gz' % i))

images = sorted(glob(os.path.join(tempdir, 'im*.nii.gz')))
segs = sorted(glob(os.path.join(tempdir, 'seg*.nii.gz')))

# define transforms for image and segmentation
train_imtrans = Compose([
    ScaleIntensity(),
    AddChannel(),
    RandUniformPatch((96, 96, 96)),
    RandRotate90(prob=0.5, spatial_axes=(0, 2)),
    ToTensor()
])
train_segtrans = Compose([
    AddChannel(),
    RandUniformPatch((96, 96, 96)),
    RandRotate90(prob=0.5, spatial_axes=(0, 2)),
    ToTensor()
])
val_imtrans = Compose([
    ScaleIntensity(),
    AddChannel(),
    ToTensor()
])
val_segtrans = Compose([
    AddChannel(),
    ToTensor()
])

# define nifti dataset, data loader
check_ds = NiftiDataset(images, segs, transform=train_imtrans, seg_transform=train_segtrans)
check_loader = DataLoader(check_ds, batch_size=10, num_workers=2, pin_memory=torch.cuda.is_available())
im, seg = monai.utils.misc.first(check_loader)
print(im.shape, seg.shape)

# create a training data loader
train_ds = NiftiDataset(images[:20], segs[:20], transform=train_imtrans, seg_transform=train_segtrans)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=8, pin_memory=torch.cuda.is_available())
# create a validation data loader
val_ds = NiftiDataset(images[-20:], segs[-20:], transform=val_imtrans, seg_transform=val_segtrans)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())

# create UNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
model = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)
loss_function = monai.losses.DiceLoss(do_sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

# start a typical PyTorch training
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
writer = SummaryWriter()
for epoch in range(5):
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch + 1, 5))
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print("%d/%d, train_loss:%0.4f" % (step, epoch_len, loss.item()))
        writer.add_scalar('train_loss', loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print("epoch %d average loss:%0.4f" % (epoch + 1, epoch_loss))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            metric_sum = 0.
            metric_count = 0
            val_images = None
            val_labels = None
            val_outputs = None
            for val_data in val_loader:
                val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
                value = compute_meandice(y_pred=val_outputs, y=val_labels, include_background=True,
                                         to_onehot_y=False, add_sigmoid=True)
                metric_count += len(value)
                metric_sum += value.sum().item()
            metric = metric_sum / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), 'best_metric_model.pth')
                print('saved new best metric model')
            print("current epoch %d current mean dice: %0.4f best mean dice: %0.4f at epoch %d"
                  % (epoch + 1, metric, best_metric, best_metric_epoch))
            writer.add_scalar('val_mean_dice', metric, epoch + 1)
            # plot the last model output as GIF image in TensorBoard with the corresponding image and label
            plot_2d_or_3d_image(val_images, epoch + 1, writer, index=0, tag='image')
            plot_2d_or_3d_image(val_labels, epoch + 1, writer, index=0, tag='label')
            plot_2d_or_3d_image(val_outputs, epoch + 1, writer, index=0, tag='output')
shutil.rmtree(tempdir)
print('train completed, best_metric: %0.4f  at epoch: %d' % (best_metric, best_metric_epoch))
writer.close()
