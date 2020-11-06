import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from skimage.transform import resize
import nibabel as nib

import os
import re
import csv

import midl
from midl.utils.image_processing import write_raw


def infer(model,
          test_loader):

    # Load shape dict
    f = open('./test_shape_info.csv', 'r')
    rdr = csv.reader(f)
    shape_dict = {rows[0]: rows[1:] for rows in rdr}
    f.close()
    print(shape_dict)

    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            image = (torch.from_numpy(np.expand_dims(sample['image'], axis=1)).float()).to(device)

            out = model.forward(image)

            pred = F.softmax(out, dim=1)
            pred = pred.permute(0, 2, 3, 4, 1).contiguous()
            pred = torch.argmax(pred, dim=-1)
            print(pred.shape)
            pred = np.squeeze(pred.to('cpu').numpy())

            # out_prob = np.squeeze(out_prob.to('cpu').numpy())
            # out_label = np.squeeze(out.to('cpu').numpy())
            # out_label = out_label.to('cpu')

            #pred = pred.astype(np.int16)

            filename = sample['filename'][0].split('.')
            filename = filename[0].split('_')
            filename = '_'.join(filename[0:3]) + "_label.nii.gz"

            dim = [int(x) for x in shape_dict[filename]]
            print(dim)

            pred = resize(pred, dim, order=0, preserve_range=True, anti_aliasing=False).astype(np.int16)

            # Label value recovery
            org_value = [0, 205, 420, 500, 550, 600, 820, 850]
            for i in range(len(org_value)):
                pred[pred == i] = org_value[i]

            # Save nifti
            nifti_image = nib.Nifti1Image(pred.transpose((2, 1, 0)), affine=np.eye(4))
            nib.save(nifti_image, os.path.join('.', '%s' % filename))
            # write_raw(pred, os.path.join('.', '%s' % filename))


if __name__ == "__main__":
    device = torch.device('cuda')

    test_ds = midl.ds.MMWHS2017Dataset(128, 128, 64,
                                       'D:\data\Cardiac\MMWHS2017\ct_test_raw\image')

    test_loader = torch.utils.data.DataLoader(test_ds, 1)

    model = midl.models.UNet(dim=3, in_channels=1, n_classes=8)
    model = model.to(device)

    checkpoint = torch.load('./models/valbest.pth')
    model.load_state_dict(checkpoint['net'])

    infer(model, test_loader)
