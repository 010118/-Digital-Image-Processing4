import os
import cv2
import sys
import re
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime
from tabulate import tabulate

import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset

import pandas as pd

from torch.utils.tensorboard import SummaryWriter

from A6_submission import Classifier, Params


class NotMNIST_RGB(Dataset):
    def __init__(self, fname='train_data_rgb.npz'):
        train_data = np.load(fname, allow_pickle=True)
        """_train_images is a 4D array of size n_images x 28 x 28 x 3"""
        self._train_images, self._train_labels = train_data['images'], train_data[
            'labels']  # type: np.ndarray, np.ndarray
        """
        pixel values converted to floating-point numbers and normalized to be between 0 and 1 to make them 
        suitable for processing in CNNs
        """
        self._train_images = self._train_images.astype(np.float32) / 255.0
        """switch images from n_images x 28 x 28 x 3 to n_images x 3 x 28 x 28 since CNNs expect the channels to be 
        the first dimension"""
        self._train_images = np.transpose(self._train_images, (0, 3, 1, 2))

        self._train_labels = self._train_labels.astype(np.int64)
        self._n_train = self._train_images.shape[0]

    def __len__(self):
        return self._n_train

    def __getitem__(self, idx):
        assert idx < self._n_train, "Invalid idx: {} for n_train: {}".format(idx, self._n_train)

        images = self._train_images[idx, ...]
        labels = self._train_labels[idx]
        return images, labels


def evaluate(classifier, data_loader, criterion_cls, vis, writer, iteration, device):
    mean_loss_sum = 0
    _psnr_sum = 0
    total_images = 0
    correct_images = 0

    n_batches = 0
    _pause = 1

    # set CNN to evaluation mode
    classifier.eval()

    total_test_time = 0

    # disable gradients computation
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            start_t = time.time()

            outputs = classifier(inputs)

            end_t = time.time()
            test_time = end_t - start_t

            total_test_time += test_time

            loss = criterion_cls(outputs, targets)

            mean_loss = loss.item()

            mean_loss_sum += mean_loss

            _, predicted = outputs.max(1)
            total_images += targets.size(0)

            is_correct = predicted.eq(targets)

            correct_images += is_correct.sum().item()

            n_batches += 1

            if vis:
                inputs_np = inputs.detach().cpu().numpy()
                concat_imgs = []
                for i in range(data_loader.batch_size):
                    input_img = inputs_np[i, ...].squeeze()

                    """switch image from 3 x 28 x 28 to 28 x 28 x 3 since opencv expects channels 
                    to be on the last axis
                    copy is needed to resolve an opencv issue with memory layout:
                    https://stackoverflow.com/questions/23830618/python-opencv-typeerror-layout-of-the-output-array
                    -incompatible-with-cvmat
                    """
                    input_img = np.transpose(input_img, (1, 2, 0)).copy()

                    target = targets[i]
                    output = predicted[i]
                    _is_correct = is_correct[i].item()
                    if _is_correct:
                        col = (0, 1, 0)
                    else:
                        col = (0, 0, 1)

                    pred_img = np.zeros_like(input_img)
                    _text = '{}'.format(chr(65 + int(output)))
                    cv2.putText(pred_img, _text, (8, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, col, 1, cv2.LINE_AA)

                    label_img = np.zeros_like(input_img)
                    _text = '{}'.format(chr(65 + int(target)))
                    cv2.putText(label_img, _text, (8, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (1, 1, 1), 1, cv2.LINE_AA)

                    concat_img = np.concatenate((input_img, label_img, pred_img), axis=0)
                    concat_imgs.append(concat_img)

                vis_img = np.concatenate(concat_imgs, axis=1)

                if writer is not None:
                    vis_img_uint8 = (vis_img * 255.0).astype(np.uint8)
                    vis_img_tb = cv2.cvtColor(vis_img_uint8, cv2.COLOR_BGR2RGB)
                    """tensorboard expects channels in the first axis"""
                    vis_img_tb = np.transpose(vis_img_tb, axes=[2, 0, 1])
                    writer.add_image('evaluation', vis_img_tb, iteration)

                if vis == 2:
                    cv2.imshow('Press Esc to exit, Space to resume, any other key for next batch', vis_img)
                    k = cv2.waitKey(1 - _pause)
                    if k == 27:
                        sys.exit(0)
                    elif k == ord('q'):
                        vis = 0
                        cv2.destroyWindow('concat_imgs')
                        break
                    elif k == 32:
                        _pause = 1 - _pause

    overall_mean_loss = mean_loss_sum / n_batches
    acc = 100. * float(correct_images) / float(total_images)

    test_speed = float(total_images) / total_test_time

    if vis == 2:
        cv2.destroyWindow('concat_imgs')

    return overall_mean_loss, acc, total_images, test_speed


def main():
    params = Params()

    # optional command line argument parsing
    try:
        import paramparse
    except ImportError:
        pass
    else:
        paramparse.process(params)

    # init device
    if params.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print('Running on GPU: {}'.format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        print('Running on CPU')

    # load dataset
    train_set = NotMNIST_RGB()

    num_train = len(train_set)
    indices = list(range(num_train))

    assert params.val_ratio > 0, "Zero validation ratio is not allowed "
    split = int(np.floor((1.0 - params.val_ratio) * num_train))

    train_idx, val_idx = indices[:split], indices[split:]

    n_train = len(train_idx)
    n_val = len(val_idx)

    print('Training samples: {}\n'
          'Validation samples: {}\n'
          ''.format(
        n_train, n_val,
    ))
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=params.train_batch_size,
                                                   sampler=train_sampler, num_workers=params.n_workers)
    val_dataloader = torch.utils.data.DataLoader(train_set, batch_size=params.val_batch_size,
                                                 sampler=val_sampler, num_workers=params.n_workers)

    # create modules
    classifier = Classifier().to(device)

    assert isinstance(classifier, nn.Module), 'classifier must be an instance of nn.Module'

    classifier.init_weights()

    # create losses
    criterion = torch.nn.CrossEntropyLoss().to(device)

    parameters = classifier.parameters()

    # create optimizer
    if params.optim_type == 0:
        optimizer = torch.optim.SGD(parameters, lr=params.lr,
                                    momentum=params.momentum,
                                    weight_decay=params.weight_decay)
    elif params.optim_type == 1:
        optimizer = torch.optim.Adam(parameters, lr=params.lr,
                                     weight_decay=params.weight_decay,
                                     eps=params.eps,
                                     )
    else:
        raise IOError('Invalid optim_type: {}'.format(params.optim_type))

    weights_dir = os.path.dirname(params.weights_path)
    weights_name = os.path.basename(params.weights_path)

    weights_dir = os.path.abspath(weights_dir)

    if not os.path.isdir(weights_dir):
        os.makedirs(weights_dir)

    tb_path = os.path.join(weights_dir, 'tb')
    if not os.path.isdir(tb_path):
        os.makedirs(tb_path)
    writer = SummaryWriter(log_dir=tb_path)

    print(f'Saving tensorboard summary to: {tb_path}')
    if params.vis:
        print('warning: writing visualization images to tensorboard is enabled which might slow it down')

    start_epoch = 0
    max_train_acc = max_val_acc = 0
    min_train_loss = min_val_loss = np.inf
    max_train_acc_epoch = max_val_acc_epoch = 0
    min_train_loss_epoch = min_val_loss_epoch = 0
    val_loss = val_acc = -1

    # load weights
    if params.load_weights:
        matching_ckpts = [k for k in os.listdir(weights_dir) if
                          os.path.isfile(os.path.join(weights_dir, k)) and
                          k.startswith(weights_name)]
        if not matching_ckpts:
            msg = 'No checkpoints found matching {} in {}'.format(weights_name, weights_dir)
            if params.load_weights == 1:
                raise IOError(msg)
            print(msg)
        else:
            matching_ckpts.sort(key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

            weights_path = os.path.join(weights_dir, matching_ckpts[-1])

            chkpt = torch.load(weights_path, map_location=device)  # load checkpoint

            print('Loading weights from: {} with:\n'
                  '\tepoch: {}\n'
                  '\ttrain_loss: {}\n'
                  '\ttrain_acc: {}\n'
                  '\tval_loss: {}\n'
                  '\tval_acc: {}\n'
                  '\ttimestamp: {}\n'.format(
                weights_path, chkpt['epoch'],
                chkpt['train_loss'], chkpt['train_acc'],
                chkpt['val_loss'], chkpt['val_acc'],
                chkpt['timestamp']))

            classifier.load_state_dict(chkpt['classifier'])
            optimizer.load_state_dict(chkpt['optimizer'])

            max_val_acc = chkpt['val_acc']
            min_val_loss = chkpt['val_loss']

            max_train_acc = chkpt['train_acc']
            min_train_loss = chkpt['train_loss']

            min_train_loss_epoch = min_val_loss_epoch = max_train_acc_epoch = max_val_acc_epoch = chkpt['epoch']
            start_epoch = chkpt['epoch'] + 1

    if params.load_weights != 1:
        # start / continue training
        for epoch in range(start_epoch, params.n_epochs):
            # set CNN to training mode
            classifier.train()

            train_loss = 0
            train_total = 0
            train_correct = 0
            batch_idx = 0

            save_weights = 0

            for batch_idx, (inputs, targets) in tqdm(enumerate(train_dataloader)):
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                outputs = classifier(inputs)

                loss = criterion(outputs, targets)

                mean_loss = loss.item()
                train_loss += mean_loss

                loss.backward()
                optimizer.step()

                _, predicted = outputs.max(1)
                train_total += targets.size(0)
                train_correct += predicted.eq(targets).sum().item()

            mean_train_loss = train_loss / (batch_idx + 1)

            train_acc = 100. * train_correct / train_total

            # write training data for tensorboard
            writer.add_scalar('train/loss', train_loss, epoch)
            writer.add_scalar('train/acc', train_acc, epoch)

            if epoch % params.val_gap == 0:

                val_loss, val_acc, _, val_speed = evaluate(
                    classifier, val_dataloader, criterion, params.vis, writer, epoch, device)
                print(f'validation speed: {val_speed:.4f} images / sec')

                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    max_val_acc_epoch = epoch
                    if params.save_criterion == 0:
                        save_weights = 1

                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    min_val_loss_epoch = epoch
                    if params.save_criterion == 1:
                        save_weights = 1

                if train_acc > max_train_acc:
                    max_train_acc = train_acc
                    max_train_acc_epoch = epoch
                    if params.save_criterion == 2:
                        save_weights = 1

                if train_loss < min_train_loss:
                    min_train_loss = train_loss
                    min_train_loss_epoch = epoch
                    if params.save_criterion == 3:
                        save_weights = 1

                # write validation data for tensorboard
                writer.add_scalar('validation/loss', val_loss, epoch)
                writer.add_scalar('validation/acc', val_acc, epoch)

                rows = ('training', 'validation')
                cols = ('loss', 'acc', 'min_loss (epoch)', 'max_acc (epoch)')

                status_df = pd.DataFrame(
                    np.zeros((len(rows), len(cols)), dtype=object),
                    index=rows, columns=cols)

                status_df['loss']['training'] = mean_train_loss
                status_df['acc']['training'] = train_acc
                status_df['min_loss (epoch)']['training'] = '{:.3f} ({:d})'.format(min_train_loss, min_train_loss_epoch)
                status_df['max_acc (epoch)']['training'] = '{:.3f} ({:d})'.format(max_train_acc, max_train_acc_epoch)

                status_df['loss']['validation'] = val_loss
                status_df['acc']['validation'] = val_acc
                status_df['min_loss (epoch)']['validation'] = '{:.3f} ({:d})'.format(min_val_loss,
                                                                                     min_val_loss_epoch)
                status_df['max_acc (epoch)']['validation'] = '{:.3f} ({:d})'.format(max_val_acc,
                                                                                    max_val_acc_epoch)
                print('Epoch: {}'.format(epoch))
                print(tabulate(status_df, headers='keys', tablefmt="orgtbl", floatfmt='.3f'))

            # Save checkpoint.
            if save_weights:
                model_dict = {
                    'classifier': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_loss': mean_train_loss,
                    'train_acc': train_acc,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'epoch': epoch,
                    'timestamp': datetime.now().strftime("%y/%m/%d %H:%M:%S"),
                }
                weights_path = '{}.{:d}'.format(params.weights_path, epoch)
                print('Saving weights to {}'.format(weights_path))
                torch.save(model_dict, weights_path)

    if params.enable_test:
        test_set = NotMNIST_RGB('test_data_rgb.npz')
        test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=params.test_batch_size,
                                                      num_workers=params.n_workers)
    else:
        test_dataloader = val_dataloader

    _, test_acc, n_test, test_speed = evaluate(
        classifier, test_dataloader, criterion, vis=0, device=device, writer=None, iteration=0)

    min_speed = 25
    min_marks = 50
    max_marks = 100
    min_acc = 80
    max_acc = 95

    if test_speed < min_speed or test_acc < min_acc:
        marks = 0
    elif test_acc >= max_acc:
        marks = 100
    else:
        marks = min_marks + (max_marks - min_marks) * (test_acc - min_acc) / (max_acc - min_acc)

    print(f'test accuracy: {test_acc:.4f}%')
    print(f'test speed: {test_speed:.4f} images / sec')
    print(f'marks: {marks:.2f}%')


if __name__ == '__main__':
    main()
