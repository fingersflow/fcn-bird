from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from MyData import test_dataloader, train_dataloader
import time
full_to_train = {0: 0, 50: 1, 100: 2, 150: 3, 200: 4, 250: 5}
train_to_full = {0: 0, 1: 50, 2: 100, 3: 150, 4: 200, 5: 250}
color_to_label = {'Sedan': 40, 'Truck': 80, 'Bus': 120,
                'Microbus': 160, 'Minivan': 200, 'SUV': 240}

fmap1_block = list()
input1_block = list()
fmap2_block = list()
input2_block = list()
fmap3_block = list()
input3_block = list()
fmap4_block = list()
input4_block = list()


def forward_hook1(module, data_input, data_output):
    fmap1_block.append(data_output)
    input1_block.append(data_input)

def forward_hook2(module, data_input, data_output):
    fmap2_block.append(data_output)
    input2_block.append(data_input)

def forward_hook3(module, data_input, data_output):
    fmap3_block.append(data_output)
    input3_block.append(data_input)

def forward_hook4(module, data_input, data_output):
    fmap4_block.append(data_output)
    input4_block.append(data_input)

def train(epo_num=50):

    vis = visdom.Visdom()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # fcn_model = torch.hub.load('pytorch/vision', 'fcn_resnet101', pretrained=True)
    # fcn_model.classifier[4] = nn.Conv2d(512, 6, kernel_size=(1, 1), stride=(1, 1))
    # fcn_model.aux_classifier[4] = nn.Conv2d(256, 6, kernel_size=(1, 1), stride=(1, 1))

    model_path = "checkpoints/fcn_model_95.pth"
    fcn_model = torch.load(model_path)

    fcn_model = fcn_model.to(device)

    fcn_model.backbone.conv1.register_forward_hook(forward_hook1)
    fcn_model.backbone.layer1[0].conv1.register_forward_hook(forward_hook2)
    fcn_model.backbone.layer1[0].conv2.register_forward_hook(forward_hook3)
    fcn_model.backbone.layer2[0].conv1.register_forward_hook(forward_hook4)

    criterion = nn.BCELoss().to(device)
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    all_train_iter_loss = []
    all_test_iter_loss = []

    train_epo_count = []
    test_epo_count = []
    piex_total = 50176

    # start timing
    prev_time = datetime.now()
    for epo in range(epo_num):
        epo_IoU = 0
        eop_PA = 0
        train_loss = 0
        fcn_model.train()
        for index, (imgA, imgB, one_hot_mask, mask, img_name) in enumerate(train_dataloader):
            batch_size = len(img_name)

            imgA = imgA.to(device)
            one_hot_mask = one_hot_mask.to(device)

            optimizer.zero_grad()
            result = fcn_model(imgA)

            # output = result
            output= result['out']
            output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])

            loss = criterion(output, one_hot_mask)

            loss.backward()

            output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)
            output_np = np.argmax(output_np, axis=1)
            output_cpoy = output_np.copy()
            for k, v in train_to_full.items():
                output_cpoy[output_np == k] = v
            output_np = output_cpoy

            train_epo_count.append(epo+index/len(train_dataloader))
            # IOU
            mask = mask.detach().numpy()
            count = mask.copy()
            for i in range(batch_size):
                count[i][output_np[i] != mask[i]] = 1
            arr_gb = count.flatten()  # 数组转为1维
            arr_gb = pd.Series(arr_gb)  # 转换数据类型
            arr_gb = arr_gb.value_counts()  # 计数
            arr_gb = dict(arr_gb)
            FN_FP = arr_gb[1]

            mask_copy = mask.copy()
            for i in range(batch_size):
                mask_copy[i][mask[i] == 0] = 2
            count = mask.copy()
            for i in range(batch_size):
                count[i][output_np[i] == mask_copy[i]] = 1
            arr_gb = count.flatten()  # 数组转为1维
            arr_gb = pd.Series(arr_gb)  # 转换数据类型
            arr_gb = arr_gb.value_counts()  # 计数
            arr_gb = dict(arr_gb)
            TP = arr_gb[1]
            IOU = TP/(TP+FN_FP)
            epo_IoU += IOU
            print('IOU:{}'.format(IOU))

            # # 像素准确率PA
            count = output_np-mask
            arr_gb = count.flatten()  # 数组转为1维
            arr_gb = pd.Series(arr_gb)  # 转换数据类型
            arr_gb = arr_gb.value_counts()  # 计数
            arr_gb = dict(arr_gb)
            TPN = arr_gb[0]
            PA = TPN/(piex_total*batch_size)
            print('PA:{}'.format(PA))
            eop_PA += PA
            iter_loss = loss.item()
            all_train_iter_loss.append(iter_loss)
            train_loss += iter_loss
            optimizer.step()
            if np.mod(index, 6) == 0:
                print('epoch {}, {}/{},train loss is {}'.format(epo, index, len(train_dataloader), iter_loss))
                # vis.close()
                vis.images(fmap1_block[0][0][0][None, None, :, :], win='feature map11',
                           opts=dict(title='train feature11'))
                vis.images(fmap1_block[0][0][1][None, None, :, :], win='feature map12',
                           opts=dict(title='train feature12'))
                vis.images(fmap2_block[0][0][0][None, None, :, :], win='feature map21',
                           opts=dict(title='train feature21'))
                vis.images(fmap2_block[0][0][1][None, None, :, :], win='feature map22',
                           opts=dict(title='train feature22'))
                vis.images(fmap3_block[0][0][0][None, None, :, :], win='feature map31',
                           opts=dict(title='train feature31'))
                vis.images(fmap3_block[0][0][1][None, None, :, :], win='feature map32',
                           opts=dict(title='train feature32'))
                vis.images(fmap4_block[0][0][0][None, None, :, :], win='feature map41',
                           opts=dict(title='train feature41'))
                vis.images(fmap4_block[0][0][1][None, None, :, :], win='feature map42',
                           opts=dict(title='train feature42'))
                vis.images(imgB[0, :, :, :], win='train_img', opts=dict(title='train image'))
                # vis.images(output_np[:, None, :, :], win='train_pred', opts=dict(title='train prediction'))
                # vis.images(mask[:, None, :, :], win='train_label', opts=dict(title='train mask'))
                # vis.line(all_train_iter_loss, train_epo_count,
                #          win='train_iter_loss', opts=dict(title='train iter loss'))
        # scheduler.step()
        test_loss = 0
        fcn_model.eval()
        with torch.no_grad():
            for index, (imgA, imgB, one_hot_mask, mask, img_name) in enumerate(test_dataloader):

                imgA = imgA.to(device)
                one_hot_mask = one_hot_mask.to(device)
                optimizer.zero_grad()

                result = fcn_model(imgA)
                output = result['out']
                output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])

                loss = criterion(output, one_hot_mask)
                iter_loss = loss.item()
                all_test_iter_loss.append(iter_loss)
                test_loss += iter_loss

                output_np = output.cpu().detach().numpy().copy() # output_np.shape = (4, 2, 160, 160)  
                output_np = np.argmax(output_np, axis=1)
                output_cpoy = output_np.copy()
                for k, v in train_to_full.items():
                    output_cpoy[output_np == k] = v
                output_np = output_cpoy

                test_epo_count.append(epo + index / len(test_dataloader))

                if np.mod(index, 6) == 0:
                    print(r'Testing... Open http://localhost:8097/ to see test result.')
                    # vis.close()
                    # vis.images(imgB[:, :, :, :], win='test_img', opts=dict(title='test image'))
                    # vis.images(output_np[:, None, :, :], win='test_pred', opts=dict(title='test prediction'))
                    # vis.images(mask[:, None, :, :], win='test_label', opts=dict(title='test mask'))
                    # vis.line(all_test_iter_loss, test_epo_count,  win='test_iter_loss', opts=dict(title='test iter loss'))

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time

        print('epoch: train loss = %f IoU = %f PA = %f,  test loss = %f, %s'
                %(train_loss/len(train_dataloader),epo_IoU/len(train_dataloader) ,
                  eop_PA/len(train_dataloader), test_loss/len(test_dataloader), time_str))
        if np.mod(epo, 5) == 0:
            torch.save(fcn_model, 'checkpoints/fcn_model_{}.pth'.format(epo))
            print('saveing checkpoints/fcn_model_{}.pth'.format(epo))


if __name__ == "__main__":

    train(epo_num=1)
