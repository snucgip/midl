import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

import midl
from midl.layers.losses.DiceLoss import DiceLoss

import pickle

min_loss = float("inf")
min_val_loss = float("inf")


def train(epoch,
          model,
          train_loader,
          optimizer):

    model.train()
    total_loss = 0.0

    for batch_idx, sample in enumerate(train_loader):

        optimizer.zero_grad()

        image = (torch.from_numpy(np.expand_dims(sample['image'], axis=1)).float()).to(device)
        label = (sample['label'].long()).to(device)

        out = model.forward(image)

        pred = F.softmax(out, dim=1)

        weights = torch.tensor([1.0] * 8).to(device)
        labels_one_hot = F.one_hot(label, num_classes=8)
        labels_one_hot = labels_one_hot.permute(0, 4, 1, 2, 3).contiguous()
        loss = metric(pred, labels_one_hot, weights)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Progress bar
        data_cnt = len(train_loader.dataset)
        done_cnt = min((batch_idx + 1) * train_loader.batch_size, data_cnt)
        rate = done_cnt / data_cnt
        bar = ('=' * int(rate * 32) + '>').ljust(32, '.')
        idx = str(done_cnt).rjust(len(str(data_cnt)), ' ')
        print('\rTrain\t : {}/{}: [{}]'.format(
            idx,
            data_cnt,
            bar
        ), end='')
    print(' epoch: ' + str(epoch) + ' loss: ' + str(total_loss))

    if total_loss < min_loss:
        state = {
            'epoch': epoch,
            'net': model.state_dict()
        }
        torch.save(state, "./models/best.pth")


def val(epoch,
        model,
        val_loader,
        scheduler):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):

            image = (torch.from_numpy(np.expand_dims(sample['image'], axis=1)).float()).to(device)
            label = (sample['label'].long()).to(device)

            # Val
            out = model.forward(image)

            pred = F.softmax(out, dim=1)

            weights = torch.tensor([1.0] * 8).to(device)
            labels_one_hot = F.one_hot(label, num_classes=8)
            labels_one_hot = labels_one_hot.permute(0, 4, 1, 2, 3).contiguous()
            loss = metric(pred, labels_one_hot, weights)

            total_loss += loss.item()

            # Progress bar
            data_cnt = len(val_loader.dataset)
            done_cnt = min((batch_idx + 1) * val_loader.batch_size, data_cnt)
            rate = done_cnt / data_cnt
            bar = ('=' * int(rate * 32) + '>').ljust(32, '.')
            idx = str(done_cnt).rjust(len(str(data_cnt)), ' ')
            print('\rVal\t : {}/{}: [{}]'.format(
                idx,
                data_cnt,
                bar
            ), end='')

        print(' epoch: ' + str(epoch) + ' loss: ' + str(total_loss))

        if total_loss < min_val_loss:
            state = {
                'epoch': epoch,
                'net': model.state_dict()
            }
            torch.save(state, "./models/valbest.pth")

        if epoch % 2 == 0:
            state = {
                'epoch': epoch,
                'net': model.state_dict()
            }
            torch.save(state, "./models/{}.pth".format(str(epoch)))

        scheduler.step(total_loss)


if __name__ == "__main__":
    device = torch.device('cuda')

    train_ds = midl.ds.MMWHS2017Dataset(128, 128, 64,
                                        'D:\data\Cardiac\MMWHS2017\ct_train_raw\image',
                                        'D:\data\Cardiac\MMWHS2017\ct_train_raw\label')
    train_loader = torch.utils.data.DataLoader(train_ds, 1)

    val_ds = midl.ds.MMWHS2017Dataset(128, 128, 64,
                                      'D:\data\Cardiac\MMWHS2017\ct_val_raw\image',
                                      'D:\data\Cardiac\MMWHS2017\ct_val_raw\label')
    val_loader = torch.utils.data.DataLoader(val_ds, 1)

    model = midl.models.UNet(dim=3, in_channels=1, n_classes=8)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.1, verbose=True, eps=1e-8)

    metric = DiceLoss()

    for epoch in range(300):
        train(epoch, model, train_loader, optimizer)
        val(epoch, model, val_loader, scheduler)
