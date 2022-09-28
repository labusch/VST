# import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from .dataset import get_loader
from .transforms import *
from torchvision import transforms
import time
from .Models.ImageDepthNet import ImageDepthNet
from torch.utils import data
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

from qurator.sbb_images.parallel import run_unordered as prun
import multiprocessing as mp


class ImgSaver:
    save_test_path_root = None

    @staticmethod
    def initialize(save_test_path_root):
        ImgSaver.save_test_path_root = save_test_path_root

    def __init__(self, output_s, image_w, image_h, image_path):

        self.output_s = output_s
        self.image_w = image_w
        self.image_h = image_h
        self.image_path = image_path

    def __call__(self, *args, **kwargs):

        transform = Compose([
            transforms.ToPILImage(),
            Scale((self.image_w, self.image_h))
        ])

        output_s = transform(self.output_s)

        # save saliency maps
        img_path = os.path.dirname(self.image_path)

        save_test_path = ImgSaver.save_test_path_root + '/RGB_VST/' + img_path
        Path(save_test_path).mkdir(parents=True, exist_ok=True)

        output_s.save(save_test_path + "/" + Path(self.image_path).stem + ".png")

        return


def test_net(args):
    mp.set_start_method('spawn', force=True)

    cudnn.benchmark = True

    net = ImageDepthNet(args)
    net.cuda()
    net.eval()

    # load model (multi-gpu)
    model_path = args.save_model_dir  # + 'RGB_VST.pth'
    state_dict = torch.load(model_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    net.load_state_dict(new_state_dict)
    print('Model loaded from {}'.format(model_path))

    test_paths = args.test_paths.split('+')

    def saliency():

        for test_dir_img in test_paths:
    
            test_dataset = get_loader(test_dir_img, args.data_root, args.img_size, mode='test')
    
            test_loader = data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                          shuffle=False, num_workers=16)
            print('''
                       Starting testing:
                           dataset: {}
                           Testing size: {}
                       '''.format(test_dir_img.split('/')[0], len(test_loader.dataset)))
    
            time_list = []
            for i, data_batch in enumerate(tqdm(test_loader)):
                images, image_w_batch, image_h_batch, image_paths = data_batch
                images = Variable(images.cuda())
    
                starts = time.time()
                outputs_saliency, outputs_contour = net(images)
                ends = time.time()
                time_use = ends - starts
                time_list.append(time_use)
    
                mask_1_16, mask_1_8, mask_1_4, mask_1_1 = outputs_saliency
    
                output_s_batch = F.sigmoid(mask_1_1)

                for output_s, image_w, image_h, image_path in \
                    zip(output_s_batch, image_w_batch, image_h_batch, image_paths):
    
                    image_w, image_h = int(image_w), int(image_h)

                    output_s = output_s.data.cpu().squeeze(0).detach()
    
                    yield ImgSaver(output_s, image_w, image_h, image_path)

    for _ in prun(saliency(), initializer=ImgSaver.initialize, initargs=(args.save_test_path_root,),
                  processes=16):
        pass






