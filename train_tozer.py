
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torch import nn, optim
from torchvision.transforms import transforms
# from EffNet import EffNet
# from CNN7_feature import CNN7
# from DeepcossNet import efficientnetv2_s
# from lamb import Lamb
from dataset2 import PSFDataset
from tqdm import tqdm
# from transformer import ViT
from transformer3 import ResMLP
# from DenseNet import densenet
# 是否使用cuda
torch.cuda.set_device(0)
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#获取学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net, epoch, epoch_size, epoch_val_size, dataload, dataload_val, Epoch, criterion):
    total_loss = 0
    total_val_loss = 0
    net.train()
    print('Start Train')
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(dataload):
            x1, x2, y = batch[0], batch[1], batch[2]

            with torch.no_grad():
                # y = y.squeeze(1)
                inputs_1 = x1.to(device)
                inputs_2 = x2.to(device)
                labels = y.to(device)
                inputs_1 = inputs_1.unsqueeze(1)
                inputs_1 = inputs_1.to(torch.float32)
                inputs_2 = inputs_2.unsqueeze(1)
                inputs_2 = inputs_2.to(torch.float32)
                labels = labels.squeeze(1)
                labels = labels.to(torch.float32)
                
                inputs_12 = torch.cat((inputs_1, inputs_2), dim=1)
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs = net(inputs_12)
            losses = []
            batch_size = len(x1)
            #----------------------#
            #   计算损失
            #----------------------#
            for i in range(batch_size):
                loss_item = 100 * criterion(outputs[i], labels[i])
                losses.append(loss_item)
            loss = sum(losses) / batch_size

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix(**{'total_loss': total_loss/(iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            # Loss = total_loss/(iteration + 1)
            # Loss0 = torch.tensor(Loss)
            Loss0 = torch.tensor(total_loss)
            Loss0 = Loss0.cpu().numpy()
            np.save(r'E:\最新实验\DATA2\Loss2\Loss0/epoch_{}'.format(epoch+1), Loss0)
    print('Finish Train')

    # net.eval()
    # print('Start Validation')
    # with tqdm(total=epoch_val_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
    #     for iteration, batch in enumerate(dataload_val):
    #         x1, x2, y = batch[0], batch[1], batch[2]
    #
    #         with torch.no_grad():
    #             # y = y.squeeze(1)
    #             inputs_1 = x1.to(device)
    #             inputs_2 = x2.to(device)
    #             labels = y.to(device)
    #             inputs_1 = inputs_1.unsqueeze(1)
    #             inputs_1 = inputs_1.to(torch.float32)
    #             inputs_2 = inputs_2.unsqueeze(1)
    #             inputs_2 = inputs_2.to(torch.float32)
    #             labels = labels.squeeze(1)
    #
    #             inputs_12 = torch.cat((inputs_1, inputs_2), dim=1)
    #         # ----------------------#
    #         #   清零梯度
    #         # ----------------------#
    #         optimizer.zero_grad()
    #         #----------------------#
    #         #   前向传播
    #         #----------------------#
    #         outputs = net(inputs_12)
    #         losses = []
    #         batch_size = len(x1)
    #         #----------------------#
    #         #   计算损失
    #         #----------------------#
    #         for i in range(batch_size):
    #             loss_item = 100 * criterion(outputs[i], labels[i])
    #             losses.append(loss_item)
    #         loss = sum(losses) / batch_size
    #         total_val_loss += loss.item()
    #         pbar.set_postfix(**{'total_val_loss': total_val_loss/(iteration + 1),
    #                             'lr'        : get_lr(optimizer)})
    #         pbar.update(1)
    # print('Finish Validation')
    # print('Total Loss: %.4f || Val Loss: %.3f ' %(total_loss / (epoch_size + 1), total_val_loss / (epoch_val_size + 1)))
    if epoch % 10 == 0:
        torch.save(model.state_dict(), r'E:\最新实验\DATA2\Loss2\Epoch%d-Total_Loss%.4f.pth' % ((epoch + 1), total_loss / (epoch_size + 1)))

#模型初始化
def weights_init(net, init_type='kaiming', init_gain = 0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)



# 训练模型


if __name__ == '__main__':
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True
    # data_path1 = r'F:\最新实验\TEST\Z/'
    # data_path2 = r'F:\最新实验\TEST\F/'
    # data_path3 = r'F:\最新实验\TEST\ZER/'
    data_path1 = r'E:\最新实验\DATA2\imageZ/'
    data_path2 = r'E:\最新实验\DATA2\imageF/'
    data_path3 = r'E:\最新实验\DATA2\zernike/'
    # data_path1 = r'F:\最新实验\imageZ/'
    # data_path2 = r'F:\最新实验\imageF/'
    # data_path3 = r'F:\最新实验\zernike/'
    # model_path = r'E:\ExtenedTarget\loss\Epoch11-Total_Loss0.4163.pth'
    # print('Load weights {}.'.format(model_path))
    deep_supervision = True
    # model = densenet().to(device)
    # # model = EffNet().to(device)
    #
    model = ResMLP(in_channels=2, image_size=350, patch_size=14, num_classes=224*224,
                   dim=384, depth=12, mlp_dim=384 * 4).to(device)
    #模型初始化
    # weights_init(model)
    # model_dict = model.state_dict()
    # pretrained_dict = torch.load(model_path, map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    batch_size = 8
    criterion = nn.MSELoss()
    # optimizer = Lamb(net.parameters(), lr=5e-4, weight_decay = 0.02)
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
    #调整学习率
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                             step_size=1,
                                             gamma=0.95)
    PSF_dataset = PSFDataset(data_path1, data_path2, data_path3, 224, 'train', 15100)
    PSF_dataset_val = PSFDataset(data_path1, data_path2, data_path3, 224, 'val', 15550)
    dataloaders = DataLoader(PSF_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=False)
    dataloaders_val = DataLoader(PSF_dataset_val, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=False)
    # viz = Visdom()
    # viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    # train_model(model, criterion, optimizer, dataloaders)
    Epoch = 200
    num_img = len(dataloaders)
    num_val_img = len(dataloaders_val)
    epoch_size = num_img // batch_size
    epoch_val_size = num_val_img // batch_size

    for epoch in range(Epoch):
        fit_one_epoch(net=net, epoch=epoch, epoch_size=epoch_size, epoch_val_size=epoch_val_size,
                      dataload=dataloaders, dataload_val=dataloaders_val,
                      Epoch=Epoch, criterion=criterion)
        #调整学习率
        lr_scheduler.step()

# import matplotlib.pyplot as plt
# import torch
# import numpy as np

# def plot_loss(n):
#     y = []
#     for i in range(0,n):
#         enc = np.load('D:\MobileNet_v1\plan1-AddsingleLayer\loss\epoch_{}.npy'.format(i))
#         # enc = torch.load('D:\MobileNet_v1\plan1-AddsingleLayer\loss\epoch_{}'.format(i))
#         tempy = list(enc)
#         y += tempy
#     x = range(0,len(y))
#     plt.plot(x, y, '.-')
#     plt_title = 'BATCH_SIZE = 32; LEARNING_RATE:0.001'
#     plt.title(plt_title)
#     plt.xlabel('per 200 times')
#     plt.ylabel('LOSS')
#     # plt.savefig(file_name)
#     plt.show()

# if __name__ == "__main__":
#     plot_loss(20)
