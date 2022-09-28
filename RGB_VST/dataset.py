from PIL import Image
from torch.utils import data
from .transforms import *
from torchvision import transforms
import random
import os
from fnmatch import fnmatch
from tqdm import tqdm


def load_list(dataset_name, data_root):

    images = []
    labels = []
    contours = []

    img_root = data_root + dataset_name + '/DUTS-TR-Image/'
    img_files = os.listdir(img_root)

    for img in img_files:

        images.append(img_root + img[:-4]+'.jpg')
        labels.append(img_root.replace('/DUTS-TR-Image/', '/DUTS-TR-Mask/') + img[:-4]+'.png')
        contours.append(img_root.replace('/DUTS-TR-Image/', '/DUTS-TR-Contour/') + img[:-4] + '.png')

    return images, labels, contours


def load_test_list(test_path, pattern=None, follow_symlinks=False):

    if pattern is None:
        pattern = ["*.jpg", "*.png"]

    images = []

    # if 'DUTS' in test_path:
    #     img_root = data_root + test_path + '/DUTS-TE-Image/'
    # else:
    #     img_root = data_root + test_path + '/images/'
    #
    # img_files = os.listdir(img_root)
    # if '/HKU-IS/' in img_root:
    #     ext = '.png'
    # else:
    #     ext = '.jpg'
    # for img in img_files:
    #     images.append(img_root + img[:-4] + ext)

    def file_it(to_scan):

        for f in os.scandir(to_scan):

            try:
                if f.is_dir(follow_symlinks=follow_symlinks):
                    for g in file_it(f):
                        yield g
                else:
                    for pat in pattern:
                        if fnmatch(f.path, pat):
                            yield f.path
                            continue
            except NotADirectoryError:
                continue

    _file_it = tqdm(file_it(test_path))

    print("Scanning for {} files ...".format(pattern))
    images = []
    for p in _file_it:
        images.append(p)
        _file_it.set_description("Scanning for {} files ... [{}]".format(pattern, len(images)))

    return images


class ImageData(data.Dataset):
    def __init__(self, dataset_list, data_root, transform, mode, img_size=None, scale_size=None, t_transform=None, label_14_transform=None, label_28_transform=None, label_56_transform=None, label_112_transform=None):

        if mode == 'train':
            self.image_path, self.label_path, self.contour_path = load_list(dataset_list, data_root)
        else:
            self.image_path = load_test_list(dataset_list)

        self.transform = transform
        self.t_transform = t_transform
        self.label_14_transform = label_14_transform
        self.label_28_transform = label_28_transform
        self.label_56_transform = label_56_transform
        self.label_112_transform = label_112_transform
        self.mode = mode
        self.img_size = img_size
        self.scale_size = scale_size

    def __getitem__(self, item):
        fn = self.image_path[item].split('/')

        filename = fn[-1]

        image = Image.open(self.image_path[item]).convert('RGB')
        image_w, image_h = int(image.size[0]), int(image.size[1])

        if self.mode == 'train':

            label = Image.open(self.label_path[item]).convert('L')
            contour = Image.open(self.contour_path[item]).convert('L')
            random_size = self.scale_size

            new_img = Scale((random_size, random_size))(image)
            new_label = Scale((random_size, random_size), interpolation=Image.NEAREST)(label)
            new_contour = Scale((random_size, random_size), interpolation=Image.NEAREST)(contour)

            # random crop
            w, h = new_img.size
            if w != self.img_size and h != self.img_size:
                x1 = random.randint(0, w - self.img_size)
                y1 = random.randint(0, h - self.img_size)
                new_img = new_img.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))
                new_label = new_label.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))
                new_contour = new_contour.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))

            # random flip
            if random.random() < 0.5:
                new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                new_label = new_label.transpose(Image.FLIP_LEFT_RIGHT)
                new_contour = new_contour.transpose(Image.FLIP_LEFT_RIGHT)

            new_img = self.transform(new_img)

            label_14 = self.label_14_transform(new_label)
            label_28 = self.label_28_transform(new_label)
            label_56 = self.label_56_transform(new_label)
            label_112 = self.label_112_transform(new_label)
            label_224 = self.t_transform(new_label)

            contour_14 = self.label_14_transform(new_contour)
            contour_28 = self.label_28_transform(new_contour)
            contour_56 = self.label_56_transform(new_contour)
            contour_112 = self.label_112_transform(new_contour)
            contour_224 = self.t_transform(new_contour)

            return new_img, label_224, label_14, label_28, label_56, label_112, \
                   contour_224, contour_14, contour_28, contour_56, contour_112
        else:

            image = self.transform(image)

            return image, image_w, image_h, self.image_path[item]

    def __len__(self):
        return len(self.image_path)


def get_loader(dataset_list, data_root, img_size, mode='train'):

    if mode == 'train':

        transform = Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        t_transform = Compose([
            transforms.ToTensor(),
        ])
        label_14_transform = Compose([
            Scale((img_size // 16, img_size // 16), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_28_transform = Compose([
            Scale((img_size//8, img_size//8), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_56_transform = Compose([
            Scale((img_size//4, img_size//4), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_112_transform = Compose([
            Scale((img_size//2, img_size//2), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        scale_size = 256
    else:
        transform = Compose([
            Scale((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
        ])

    if mode == 'train':
        dataset = ImageData(dataset_list, data_root, transform, mode, img_size, scale_size, t_transform, label_14_transform, label_28_transform, label_56_transform, label_112_transform)
    else:
        dataset = ImageData(dataset_list, data_root, transform, mode)

    # data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return dataset