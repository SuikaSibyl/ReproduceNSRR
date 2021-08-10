import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from cg_manipulation.cg_utils import optical_flow_to_motion
from torchvision.utils import save_image


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (img_view_0, img_view_1, img_view_2, img_view_3, img_view_4,
                        img_depth_0, img_depth_1, img_depth_2, img_depth_3, img_depth_4,
                        img_motion_0, img_motion_1, img_motion_2, img_motion_3, img_motion_4, img_view_truth) in enumerate(self.data_loader):
            # data, target = data.to(self.device), target.to(self.device)
            truth = img_view_truth.to(self.device)
            view_0, depth_0, flow_0 = img_view_0.to(self.device), img_depth_0.to(self.device), img_motion_0.to(self.device)
            view_1, depth_1, flow_1 = img_view_1.to(self.device), img_depth_1.to(self.device), img_motion_1.to(self.device)
            view_2, depth_2, flow_2 = img_view_2.to(self.device), img_depth_2.to(self.device), img_motion_2.to(self.device)
            view_3, depth_3, flow_3 = img_view_3.to(self.device), img_depth_3.to(self.device), img_motion_3.to(self.device)
            view_4, depth_4, flow_4 = img_view_4.to(self.device), img_depth_4.to(self.device), img_motion_4.to(self.device)

            with torch.no_grad():
                flow_0 = optical_flow_to_motion(flow_0, 5.37)
                flow_1 = optical_flow_to_motion(flow_1, 5.37)
                flow_2 = optical_flow_to_motion(flow_2, 5.37)
                flow_3 = optical_flow_to_motion(flow_3, 5.37)
                flow_4 = optical_flow_to_motion(flow_4, 5.37)

            self.optimizer.zero_grad()
            output = self.model(view_0, depth_0, flow_0, view_1, depth_1, flow_1,
                                view_2, depth_2, flow_2, view_3, depth_3, flow_3,
                                view_4, depth_4, flow_4)

            loss = self.criterion(output, truth, 0.1)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                for i in range(0,1):
                    save_image(output[i], "./data/Test/" + str(batch_idx * 1 + i) + "res.png")
                    save_image(img_view_0[i], "./data/Test/" + str(batch_idx * 1 + i) + "origin.png")
                    save_image(img_view_truth[i], "./data/Test/" + str(batch_idx * 1 + i) + "gtruth.png")


            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, loss)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self.writer.add_image('input', make_grid(view_0.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (img_view_0, img_view_1, img_view_2, img_view_3, img_view_4, \
                            img_depth_0, img_depth_1, img_depth_2, img_depth_3, img_depth_4, \
                            img_motion_0, img_motion_1, img_motion_2, img_motion_3, img_motion_4, img_view_truth) \
                    in enumerate(self.valid_data_loader):
                # data, target = data.to(self.device), target.to(self.device)
                truth = img_view_truth.to(self.device)
                view_0, depth_0, flow_0 = img_view_0.to(self.device), img_depth_0.to(self.device), img_motion_0.to(self.device)
                view_1, depth_1, flow_1 = img_view_1.to(self.device), img_depth_1.to(self.device), img_motion_1.to(self.device)
                view_2, depth_2, flow_2 = img_view_2.to(self.device), img_depth_2.to(self.device), img_motion_2.to(self.device)
                view_3, depth_3, flow_3 = img_view_3.to(self.device), img_depth_3.to(self.device), img_motion_3.to(self.device)
                view_4, depth_4, flow_4 = img_view_4.to(self.device), img_depth_4.to(self.device), img_motion_4.to(self.device)

                flow_0 = optical_flow_to_motion(flow_0, 5.37)
                flow_1 = optical_flow_to_motion(flow_1, 5.37)
                flow_2 = optical_flow_to_motion(flow_2, 5.37)
                flow_3 = optical_flow_to_motion(flow_3, 5.37)
                flow_4 = optical_flow_to_motion(flow_4, 5.37)

                output = self.model(view_0, depth_0, flow_0, view_1, depth_1, flow_1,
                                    view_2, depth_2, flow_2, view_3, depth_3, flow_3,
                                    view_4, depth_4, flow_4)

                loss = self.criterion(output, truth, 0.1)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, loss)
                self.writer.add_image('input', make_grid(view_0.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
