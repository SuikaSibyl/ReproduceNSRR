import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from cg_manipulation.cg_utils import optical_flow_to_motion

from torchvision.utils import save_image

import matplotlib.pyplot as plt

def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        # config['data_loader']['args']['root_dir'],
        batch_size=4,
        shuffle=False,
        validation_split=0.0,
        root_dir='data/',
        view_dirname='View/',
        depth_dirname='Depth/',
        flow_dirname='Motion/',
        # training=True,
        num_workers=2
    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    print("main  1")
    checkpoint = torch.load(config.resume)
    print("main  2")
    state_dict = checkpoint['state_dict']
    print("main  3")
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    print("main  4")
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for batch_idx, (img_view_0, img_view_1, img_view_2, img_view_3, img_view_4, \
                        img_depth_0, img_depth_1, img_depth_2, img_depth_3, img_depth_4, \
                        img_motion_0, img_motion_1, img_motion_2, img_motion_3, img_motion_4, img_view_truth) in enumerate(tqdm(data_loader)):
            # data, target = data.to(self.device), target.to(self.device)
            truth = img_view_truth.to(device)
            view_0, depth_0, flow_0 = img_view_0.to(device), img_depth_0.to(device), img_motion_0.to(device)
            view_1, depth_1, flow_1 = img_view_1.to(device), img_depth_1.to(device), img_motion_1.to(device)
            view_2, depth_2, flow_2 = img_view_2.to(device), img_depth_2.to(device), img_motion_2.to(device)
            view_3, depth_3, flow_3 = img_view_3.to(device), img_depth_3.to(device), img_motion_3.to(device)
            view_4, depth_4, flow_4 = img_view_4.to(device), img_depth_4.to(device), img_motion_4.to(device)

            flow_0 = optical_flow_to_motion(flow_0, 5.37)
            flow_1 = optical_flow_to_motion(flow_1, 5.37)
            flow_2 = optical_flow_to_motion(flow_2, 5.37)
            flow_3 = optical_flow_to_motion(flow_3, 5.37)
            flow_4 = optical_flow_to_motion(flow_4, 5.37)

            output = model(view_0, depth_0, flow_0, view_1, depth_1, flow_1,
                                view_2, depth_2, flow_2, view_3, depth_3, flow_3,
                                view_4, depth_4, flow_4)

            #
            # save sample images, or do something with output here
            #
            img = output.cpu()
            for i in range(0,2):
                save_image(img[i], "./data/Test/" + str(batch_idx * 2 + i) + "res.png")
                save_image(img_view_0[i], "./data/Test/" + str(batch_idx * 2 + i) + "origin.png")
                save_image(img_view_truth[i], "./data/Test/" + str(batch_idx * 2 + i) + "gtruth.png")

            # computing loss, metrics on test set
            loss = loss_fn(output, truth, 0.1)
            batch_size = view_0.shape[0]
            total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
