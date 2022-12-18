import json
import os
import shutil
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from timm.utils import accuracy, AverageMeter
from sklearn.metrics import classification_report
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torchvision import datasets
torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING']='1'
os.environ['CUDA_VISIBLE_DEVICES']='0'

#Define EMA
class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

if __name__ == '__main__':
    # create the file to save the model
    file_dir = 'checkpoints/SwinV2'
    if os.path.exists(file_dir):
       os.rmdir(file_dir)
       shutil.rmtree(file_dir)  # remove the old one, then create a new one
       os.makedirs(file_dir)
    else:
       os.makedirs(file_dir)
       
    # set global parameters
    model_lr = 2e-5
    BATCH_SIZE = 8
    DATA_SIZE = 384
    EPOCHS = 20
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    classes = 33
    model_path = 'best.pth'
    Best_ACC = 0 # record the best acc
    use_ema=True
    ema_epoch=32

transform = transforms.Compose([
    transforms.Resize((510, 510)),
    transforms.GaussianBlur(kernel_size=(5,5),sigma=(0.1, 3.0)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    transforms.RandomRotation(20),
    transforms.RandomCrop((DATA_SIZE, DATA_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45923316, 0.48780942, 0.42035612], std= [0.25167787, 0.24934903, 0.2854071])
])
transform_test = transforms.Compose([
    transforms.Resize((510, 510)),
    transforms.CenterCrop((DATA_SIZE, DATA_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45923316, 0.48780942, 0.42035612], std= [0.25167787, 0.24934903, 0.2854071])
])
# Define data augmentation mixup
mixup_fn = Mixup(
    mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
    prob=0.1, switch_prob=0.5, mode='batch',
    label_smoothing=0.1, num_classes=classes)

dataset_train = datasets.ImageFolder('data_sp91/train', transform=transform)
dataset_test = datasets.ImageFolder("data_sp91/val", transform=transform_test)
with open('class.txt', 'w') as file:
    file.write(str(dataset_train.class_to_idx))
with open('class.json', 'w', encoding='utf-8') as file:
    file.write(json.dumps(dataset_train.class_to_idx))
# load train/test data
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)

# set the loss function
criterion_train = SoftTargetCrossEntropy()
criterion_val = torch.nn.CrossEntropyLoss()
# set the model-swinv2
model = timm.create_model('swinv2_base_window12to24_192to384_22kft1k', pretrained = True)
num_ftrs = model.head.in_features
model.head = nn.Linear(num_ftrs, classes)
print(model)

# If cuda is available, move to gpu
print(DEVICE, torch.cuda.get_device_name(0))
model = model.to(DEVICE)

# choose the optimizer AdamW(Adam + L2)
optimizer = optim.AdamW(model.parameters(), lr=model_lr)
# the learning rate adjustment strategy is chosen as cosine
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-7)

if use_ema:
    ema = EMA(model, 0.999)
    ema.register()
    

# Define train
def train(model, device, train_loader, optimizer, epoch):
    model.train()

    # AverageMeter(): Computes and stores the average and current value
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    # mixup_fn only accepts even
    for batch_idx, (data, target) in enumerate(train_loader):
        if len(data) % 2 != 0:
            if len(data) < 2:
                continue
            data = data[0:len(data) - 1]
            target = target[0:len(target) - 1]
            print(len(data))
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        samples, targets = mixup_fn(data, target)
        output = model(samples)
        optimizer.zero_grad()
        
        
        loss = criterion_train(output, targets)
        loss.backward()
        optimizer.step()
        if use_ema and epoch%ema_epoch==0:
            ema.update()
            
        torch.cuda.synchronize() # Wait for all the above operations to complete
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loss_meter.update(loss.item(), target.size(0))
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), lr))
    ave_loss =loss_meter.avg
    acc = acc1_meter.avg
    print('epoch:{}\tloss:{:.2f}\tacc:{:.2f}'.format(epoch, ave_loss, acc))
    return ave_loss, acc


# Define validation
@torch.no_grad() # The data does not need to compute gradients, nor does backpropagation
def val(model, device, test_loader):
    global Best_ACC
    model.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    val_list = []
    pred_list = []
    if use_ema and epoch%ema_epoch==0:
        ema.apply_shadow()
    for data, target in test_loader:
        for t in target:
            val_list.append(t.data.item())
        data, target = data.to(device,non_blocking=True), target.to(device,non_blocking=True)
        output = model(data)
        loss = criterion_val(output, target)
        _, pred = torch.max(output.data, 1)
        for p in pred:
            pred_list.append(p.data.item())
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))
    if use_ema and epoch%ema_epoch==0:
        ema.restore()
    acc = acc1_meter.avg
    print('\nVal set: Average loss: {:.4f}\tAcc1:{:.3f}%\tAcc5:{:.3f}%\n'.format(
        loss_meter.avg,  acc,  acc5_meter.avg))
    if acc > Best_ACC: 
        Best_ACC = acc
    torch.save(model, file_dir + "/" + 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
    torch.save(model, file_dir + '/' + 'best.pth')
        
    return val_list, pred_list, loss_meter.avg, acc


# train/validation
log_dir = {} # record log
train_loss_list, val_loss_list, train_acc_list, val_acc_list, epoch_list = [], [], [], [], []
for epoch in range(1, EPOCHS + 1):
    epoch_list.append(epoch)
    train_loss, train_acc = train(model, DEVICE, train_loader, optimizer, epoch)
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    log_dir['train_acc'] = train_acc_list
    log_dir['train_loss'] = train_loss_list
    val_list, pred_list, val_loss, val_acc = val(model, DEVICE, test_loader)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    log_dir['val_acc'] = val_acc_list
    log_dir['val_loss'] = val_loss_list
    log_dir['best_acc'] = Best_ACC
    with open(file_dir + '/result.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(log_dir))
    print(classification_report(val_list, pred_list, target_names=dataset_train.class_to_idx))
    scheduler.step()
    
    
    fig = plt.figure(1)
    plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train Loss')
    # picture
    plt.plot(epoch_list, val_loss_list, 'b-', label=u'Val Loss')
    plt.legend(["Train Loss", "Val Loss"], loc="upper right")
    plt.xlabel(u'epoch')
    plt.ylabel(u'loss')
    plt.title('Model Loss ')
    plt.savefig(file_dir + "/loss.png")
    plt.close(1)
    fig2 = plt.figure(2)
    plt.plot(epoch_list, train_acc_list, 'r-', label=u'Train Acc')
    plt.plot(epoch_list, val_acc_list, 'b-', label=u'Val Acc')
    plt.legend(["Train Acc", "Val Acc"], loc="lower right")
    plt.title("Model Acc")
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.savefig(file_dir + "/acc.png")
    plt.close(2)
