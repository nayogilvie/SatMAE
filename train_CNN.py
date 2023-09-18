import torch
from torch import nn
import numpy as np
import transforms_seg as T_seg
from folder import SegmentationDataset
from ignite.engine import Engine
from ignite.metrics import IoU, Precision, Recall, ConfusionMatrix
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import os
import focal_loss
import csv
import segmentation_models_pytorch as smp



def train_one_epoch(epoch, dataloader, model, criterion, optimizer, device, writer):
    print('train epoch {}'.format(epoch))
    model.train()
    for idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.float()
        targets = targets.float()
        
        # infrared = inputs[:, 3, :, :]
        # RGB = inputs[:, :3, :, :]
        # infrared = infrared.unsqueeze(1)
        
        #print("infrared shape: ", infrared.shape)
        #print("RGB shape: ", RGB.shape)
        
        # MAE extract raw feature from RGB
        # with torch.no_grad():
        #     mae_output = model_mae(RGB)

        
        outputs = model(inputs)
        #get loss
        targets = torch.argmax(targets, dim=1)
        loss = criterion(outputs, targets)
        #propogate results
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #record results
        print('train-epoch:{} [{}/{}], loss: {:5.3}'.format(epoch, idx+1, len(dataloader), loss.item()))
        writer.add_scalar('train/loss', loss.item(), len(dataloader)*epoch+idx)

def evalidation(epoch, dataloader, model, criterion, device, writer, csv_file):
    print('\neval epoch {}'.format(epoch))
    model.eval()
    recall = Recall(lambda x: (x[0], x[1]))
    precision = Precision(lambda x: (x[0], x[1]))
    #default_evaluator = Engine(lambda x: (x[0], x[1]))
    #print(lambda x: (x[0], x[1]))
    mean_recall = []
    mean_precision = []
    mean_loss = []
    mean_iou = []
    confusion_matrix = ConfusionMatrix(num_classes=5)
    metric = IoU(confusion_matrix)
    def eval_step(engine, batch):
        return batch
    iou = Engine(eval_step)
    metric.attach(iou, 'iou')
    #metric.attach(default_evaluator, 'cm')
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()
            #targets = targets.type(torch.LongTensor) For CPU
            targets = targets.type(torch.cuda.LongTensor)
            targets = targets.squeeze(1)
            
            # infrared = inputs[:, 3, :, :]
            # RGB = inputs[:, :3, :, :]
            # infrared = infrared.unsqueeze(1)
            
            # MAE extract raw feature from RGB
            with torch.no_grad():
                outputs = model(inputs)
        
            
            precision.update((outputs, targets))
            recall.update((outputs, targets))
            #iou.update((preds, targets))
            confusion_matrix.update((outputs, targets))
            # mean_loss.append(loss.item())
            #print(recall.compute().numpy())
            mean_recall.append(recall.compute().numpy())
            mean_precision.append(precision.compute().numpy())
            #mean_iou.append(iou.compute().numpy())

            # print('val-epoch:{} [{}/{}], loss: {:5.3}'.format(epoch, idx + 1, len(dataloader), loss.item()))
            # writer.add_scalar('test/loss', loss.item(), len(dataloader) * epoch + idx)

    #mean_precision, mean_recall = np.array(mean_precision).mean(), np.array(mean_recall).mean()
    #mean_iou = (precision.compute().numpy()).mean()
    mean_precision, mean_recall = (precision.compute().numpy()).mean(), (recall.compute().numpy()).mean()
    f1 = mean_precision * mean_recall * 2 / (mean_precision + mean_recall + 1e-20)

    #print('epoch-loss: {:07.5}, miou: {:07.5} \n'.format(np.array(mean_loss).mean(), mean_iou))
    print('precision: {:07.5}, recall: {:07.5}, f1: {:07.5}\n'.format(mean_precision, mean_recall, f1))
    print('Confusion: ')
    print(confusion_matrix.compute().numpy())
    with open(csv_file, "ab") as f:
        f.write(b"\n")
        np.savetxt(f, confusion_matrix.compute().numpy(), delimiter=',')
    # writer.add_scalar('test/epoch-loss', np.array(mean_loss).mean(), epoch)
    writer.add_scalar('test/f1', f1, epoch)
    writer.add_scalar('test/precision', mean_precision, epoch)
    writer.add_scalar('test/recall', mean_recall, epoch)
    writer.add_scalar('test/recall2', (recall.compute().numpy()).mean(), epoch)


csv_file = "./output/test_4_cross_drop_0.1_weighted_inverse_epoch_100_step_5_img_256_emd_1024_head_8_lrte_001.csv"

total_epochs = 100
Batch_size = 24
#Test with 512 at later time
img_size = 256
n_cls = 5
lrate = 0.0001


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Training on {device}")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=4,
    classes=5,
)

model = model.to(device)


# dataset is here
"""
transform_train = T_seg.Compose([
     T_seg.RandomHorizontalFlip(),
     T_seg.RandomVerticalFlip(),
     T_seg.ToTensor(),
])
transform_val = T_seg.Compose([
     T_seg.ToTensor(),
])
train_dataset = SegmentationDataset(root = "/users/n/o/nogilvie/scratch/pytorch_2/cdata_overlap/train", mode="train", extentions = ("tif"), transforms=transform_train, size=img_size)
val_dataset = SegmentationDataset(root = "/users/n/o/nogilvie/scratch/pytorch_2/cdata_overlap/val", mode="val", extentions = ("tif"), transforms=transform_val, size=img_size)
"""
transform_train = T_seg.Compose([
    T_seg.RandomRotation(degrees=20),
    T_seg.RandomHorizontalFlip(),
    T_seg.RandomVerticalFlip(),
    T_seg.RandomResizedCrop(crop_size=int(img_size*0.6), target_size=img_size),
    T_seg.ToTensor()
])

transform_test = T_seg.Compose([
    T_seg.Resize(size=img_size),
    T_seg.ToTensor()
])

train_dataset = SegmentationDataset(root = "/users/n/o/nogilvie/scratch/pytorch_2/cdata_overlap/train", mode="train", extentions = ("tif"), transforms=transform_train, size=img_size)
val_dataset = SegmentationDataset(root = "/users/n/o/nogilvie/scratch/pytorch_2/cdata_overlap/val", mode="val", extentions = ("tif"), transforms=transform_test, size=img_size)

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=Batch_size, shuffle=True)


# criterion = focal_loss.FocalLoss(n_cls, alpha=None, gamma=2, ignore_index=None, reduction='sum').to(device)
criterion = smp.losses.DiceLoss(mode = 'multiclass')

optimizer = optim.AdamW(model.parameters(), lr=lrate)
# optimizer = optim.AdamW(megSeg.parameters(), lr=lrate)
# lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-8)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)


#obtain one hot encoding

writer = SummaryWriter("./output/")
#get loss function from that
for epoch in range(total_epochs):
        writer.add_scalar('train/lr', lr_scheduler.get_lr()[0], epoch)
        train_one_epoch(epoch, data_loader_train, model, criterion, optimizer, device, writer)
        evalidation(epoch, val_loader, model, criterion, device, writer, csv_file)
        lr_scheduler.step()
        torch.save(model.state_dict(), os.path.join("./output/", 'cls_drop_0.3_weighted_inverse_test4_img_256_emd_1024_head_8_epoch_{}.pth'.format(epoch)))
