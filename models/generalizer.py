from tqdm import tqdm

from logging import getLogger
from collections import OrderedDict

from PIL import Image

import torch
import torch.nn as nn
import torchvision

logger = getLogger(__name__)

class Generalizer(object):
    def __init__(self, **kwargs):
        self.device = kwargs['device']
        self.network = kwargs['network']
        self.optimizer = kwargs['optimizer']
        self.criterion = kwargs['criterion']
        self.train_loader, self.test_loader = kwargs['data_loaders']
        self.metrics = kwargs['metrics']
        self.writer = kwargs['writer']
        self.save_ckpt_interval = kwargs['save_ckpt_interval']
        self.ckpt_dir = kwargs['ckpt_dir']

    def train(self, n_epochs, start_epoch=0):

        min_test_loss = 1e9

        for epoch in range(start_epoch, n_epochs):
            logger.info(f'\n\n==================== Epoch: {epoch} ====================')
            logger.info('### train:')
            self.network.train()

            train_loss = 0
            n_total = 0

            with tqdm(self.train_loader, ncols=100) as pbar:
                for idx, (inputs, targets) in enumerate(pbar):
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.network(inputs)

                    loss = self.criterion(outputs, inputs)

                    loss.backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    train_loss += loss.item()

                    n_total += targets.size(0)

                    ### logging train loss
                    pbar.set_postfix(OrderedDict(
                        epoch="{:>10}".format(epoch),
                        loss="{:.4f}".format(train_loss / n_total)))

            # logging train loss in ternsorboard and save as csv
            self.metrics.logging(epoch, train_loss/n_total, mode='train')
            self.metrics.save_csv(epoch, train_loss/n_total, mode='train')

            # save checkpoint
            if epoch % self.save_ckpt_interval == 0:
                logger.info('saving checkpoint...')
                self._save_ckpt(epoch, train_loss/(idx+1))

            # save image in tensorboard
            self._save_images(epoch, inputs.cpu()[:2], outputs.detach().cpu()[:2], prefix='train')

            ### test
            logger.info('### test:')
            test_loss = self.test(epoch)

            if test_loss < min_test_loss:
                logger.info(f'saving best checkpoint (epoch: {epoch})...')
                min_test_loss = test_loss
                self._save_ckpt(epoch, train_loss/(idx+1), mode='best')

    def test(self, epoch, inference=False):
        self.network.eval()
    
        test_loss = 0
        n_total = 0

        with torch.no_grad():
            with tqdm(self.test_loader, ncols=100) as pbar:
                    for idx, (inputs, targets) in enumerate(pbar):

                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)

                        outputs = self.network(inputs)

                        loss = self.criterion(outputs, inputs)

                        self.optimizer.zero_grad()

                        test_loss += loss.item()

                        n_total += targets.size(0)

                        ### logging test loss
                        pbar.set_postfix(OrderedDict(
                            epoch="{:>10}".format(epoch),
                            loss="{:.4f}".format(test_loss / n_total)))

            # logging train loss in ternsorboard and save as csv
            self.metrics.logging(epoch, test_loss/n_total, mode='test')
            self.metrics.save_csv(epoch, test_loss/n_total, mode='test')

            # save image in tensorboard
            self._save_images(epoch, inputs.cpu()[:2], outputs.detach().cpu()[:2], prefix='val')

        return test_loss / n_total

    def _save_ckpt(self, epoch, loss, mode=None, zfill=4):
        if isinstance(self.network, nn.DataParallel):
            network = self.network.module
        else:
            network = self.network

        if mode == 'best':
            ckpt_path = self.ckpt_dir / 'best_loss_ckpt.pth'
        else:
            ckpt_path = self.ckpt_dir / f'epoch{str(epoch).zfill(zfill)}_ckpt.pth'

        torch.save({
            'epoch': epoch,
            'network': network,
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, ckpt_path)

    def _save_images(self, epoch, inputs, outputs, prefix='train'):
        input_images = torchvision.utils.make_grid(inputs, nrow=2, normalize=True, scale_each=True)
        output_images = torchvision.utils.make_grid(outputs, nrow=2, normalize=True, scale_each=True)

        self.writer.add_image(f'{prefix}/input_image', input_images, epoch)
        self.writer.add_image(f'{prefix}/output_image', output_images, epoch)
