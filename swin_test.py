import torch.utils.data.distributed
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import timm
import torch.nn as nn

classes = ('asparagus', 'bambooshoots', 'betel', 'broccoli', 'cauliflower', 'chinesecabbage', 'chinesechives', 'custardapple', 'grape', 
            'greenhouse', 'greenonion', 'kale', 'lemon', 'lettuce', 'litchi', 'longan', 'loofah', 'mango', 'onion', 'others', 'papaya', 
            'passionfruit', 'pear', 'pennisetum', 'redbeans', 'roseapple', 'sesbania', 'soybeans', 'sunhemp', 'sweetpotato', 'taro', 'tea', 'waterbamboo')

num_classes = 33
size = 384
path = './model_to_predict.pth'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45923316, 0.48780942, 0.42035612], std= [0.25167787, 0.24934903, 0.2854071])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=torch.load(path)
model.eval()
model.to(DEVICE)

test_name, pred_name = [], []
public_path = "../public_test/public_test/"
private_path = "../private_data/private_data/"
publicList = os.listdir(public_path)
privateList = os.listdir(private_path)

print("--START--")
for file in publicList:
    img = Image.open(public_path + file)
    img = data_transform(img)
    img.unsqueeze_(0)
    img = Variable(img).to(device)
    out = model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    test_name.append(file)
    pred_name.append(classes[pred.data.item()])
print("Number of test(public) images : ", len(test_name))

for file in privateList:
    img = Image.open(private_path + file)
    img = data_transform(img)
    img.unsqueeze_(0)
    img = Variable(img).to(device)
    out = model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    test_name.append(file)
    pred_name.append(classes[pred.data.item()])
print("Number of test(total) images : ", len(test_name))


results = pd.DataFrame({"filename":test_name,
                        "label":pred_name})

results.head()

results.to_csv("submission.csv",index=False)
    
