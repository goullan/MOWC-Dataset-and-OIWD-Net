
from torch.utils.data import Dataset
import PIL.Image as Image
import os
import scipy.io as scio
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import cv2 as cv
import numpy as np


def make_PSFdataset1(root1, root2, root3, k):  # 远场图片的路径和泽尼克系数的路径
    images = []
    n = len(os.listdir(root2))
    # labelsfile = scio.loadmat(zernike_path)
    for i in range(k):
        x1_path = os.path.join(root1, "%05d.mat" % (i + 1))
        x2_path = os.path.join(root2, "%05d.mat" % (i + 1))
        label_path = os.path.join(root3, "%05d.mat" % (i + 1))
        x1_file = scio.loadmat(x1_path)
        x2_file = scio.loadmat(x2_path)
        label_file = scio.loadmat(label_path)
        label = torch.from_numpy(label_file['a_low'])
        x1_file['Z_defocus'] = x1_file['Z_defocus']/1.0
        x2_file['F_defocus'] = x2_file['F_defocus'] / 1.0
        image1 = torch.from_numpy(x1_file['Z_defocus'])
        image2 = torch.from_numpy(x2_file['F_defocus'])
        images.append((image1, image2, label))
        print(i)
    return images

def make_PSFdataset2(root1, root2, root3, k):  # 远场图片的路径和泽尼克系数的路径
    images = []
    n = len(os.listdir(root2))
    # labelsfile = scio.loadmat(zernike_path)
    for i in range(k, n):
        x1_path = os.path.join(root1, "%05d.mat" % (i + 1))
        x2_path = os.path.join(root2, "%05d.mat" % (i + 1))
        label_path = os.path.join(root3, "%05d.mat" % (i + 1))
        x1_file = scio.loadmat(x1_path)
        x2_file = scio.loadmat(x2_path)
        label_file = scio.loadmat(label_path)
        label = torch.from_numpy(label_file['a_low'])
        x1_file['Z_defocus'] = x1_file['Z_defocus']/1.0
        x2_file['F_defocus'] = x2_file['F_defocus'] / 1.0
        image1 = torch.from_numpy(x1_file['Z_defocus'])
        image2 = torch.from_numpy(x2_file['F_defocus'])
        images.append((image1, image2, label))
        print(i)
    return images


class PSFDataset(Dataset):
    def __init__(self, root1, root2, root3, resize, mode, k):  #初始化工作：把所有图片信息加载进来
        super(PSFDataset, self).__init__()

        self.root1 = root1
        self.root2 = root2
        self.resize = resize


        if mode =='train':
            images = make_PSFdataset1(root1, root2, root3, k)
            self.images = images
            # self.images = self.images[int(0.03*len(self.images)):]
            # self.labels = self.labels[:int(0.6*len(self.labels))]
        elif mode =='val':
            images = make_PSFdataset2(root1, root2, root3, k)
            self.images = images
            # self.images = self.images[:int(0.03 * len(self.images))]
            # self.labels = self.labels[int(0.6 * len(self.labels)):int(0.8*len(self.labels))]



    def __getitem__(self, index):  # 返回当前索引的图片的数据，比如label
        image_x1, image_x2, label_x = self.images[index]
        # img_x = Image.open(x_path)
        # img_x = img_x.convert('L')
        # # img_x.show()
        # img_x = transforms.ToTensor()(img_x)
        # # xx = np.array(img_x)
        return image_x1, image_x2, label_x




    def __len__(self):
        return len(self.images)





