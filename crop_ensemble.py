import json
import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from timm.utils import accuracy, AverageMeter
from sklearn.metrics import classification_report
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torchvision import datasets
from IPython.display import display

torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ['CUDA_VISIBLE_DEVICES']='1'

cls_label = ['asparagus', 'bambooshoots', 'betel', 'broccoli', 'cauliflower', 'chinesecabbage', 'chinesechives', 'custardapple', 'grape', 
            'greenhouse', 'greenonion', 'kale', 'lemon', 'lettuce', 'litchi', 'longan', 'loofah', 'mango', 'onion', 'others', 'papaya', 
            'passionfruit', 'pear', 'pennisetum', 'redbeans', 'roseapple', 'sesbania', 'soybeans', 'sunhemp', 'sweetpotato', 'taro', 'tea',
            'waterbamboo']

if __name__ == '__main__':
           
# set global parameters
    BATCH_SIZE = 10
    DATA_SIZE = 384
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classes = 33
                            
transform_test = transforms.Compose([
    transforms.Resize((DATA_SIZE, DATA_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std= [0.229, 0.224, 0.225])
])

dataset_test = datasets.ImageFolder("data_sp91/val", transform=transform_test)

# load train/test data
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# set the loss function
criterion = torch.nn.CrossEntropyLoss()
# set the model
model1 = torch.load("checkpoints/SwinV2_1208/model2_11_90.996.pth")
model2 = torch.load("checkpoints/SwinV2_1208/model2_14_90.952.pth")
model3 = torch.load("checkpoints/Beit_0117/model_9_89.298.pth")
model4 = torch.load("checkpoints/Beit_0117/model_12_89.689.pth")

# If cuda is available, move to gpu
print(DEVICE, torch.cuda.get_device_name(0))
model1 = model1.to(DEVICE)
model2 = model2.to(DEVICE)
model3 = model3.to(DEVICE)
model4 = model4.to(DEVICE)

# WP
def WP(all_pred, all_label):
    df = pd.DataFrame(columns = ('class', 'recall', 'precision', 'f1_score'))
    WP = 0
    for cls in range(0, classes):
        recall = all_pred[all_label==cls].tolist().count(cls) / all_label.tolist().count(cls)
        precision = all_label[all_pred==cls].tolist().count(cls) / all_pred.tolist().count(cls)
        f1_score = ( 2*precision*recall ) / ( precision+recall )
        WP += precision*all_label.tolist().count(cls)
        df.loc[cls] = [cls_label[cls], round(recall,3), round(precision,3), round(f1_score,3)]
    WP /= len(all_label)
    display(df)
    print(f'WP= {WP}')
    return WP

# Define validation
@torch.no_grad() # The data does not need to compute gradients, nor does backpropagation
def val(model1, model2, model3, model4, device, test_loader):
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
                                                                                                                                
    ep_loss = []
    all_label = []
    all_pred = []
                                                                                                                                                
    for data, target in test_loader:
        for t in target:
           all_label.append(t.data.item())                  
        data, target = data.to(device,non_blocking=True), target.to(device,non_blocking=True)
        output1 = model1(data)
        output2 = model2(data)
        output3 = model3(data)
        output4 = model4(data)
        
        output = F.softmax(output1, dim=1) + F.softmax(output2, dim=1) + F.softmax(output3, dim=1) + F.softmax(output4, dim=1)

        loss = (criterion(output1, target) + criterion(output2, target) + criterion(output3, target) + criterion(output4, target)) / 4
        ep_loss.append(loss.item())
        _, pred = torch.max(output.data, 1)
        for p in pred:
            all_pred.append(p.data.item())
    wp = WP(np.array(all_pred), np.array(all_label))
    
    return np.mean(ep_loss), wp


loss, wp = val(model1, model2, model3, model4, DEVICE, test_loader)

print('\nVal set: Average loss: {:.4f}\tWP_score:{:.3f}\n'.format(loss,  wp))
