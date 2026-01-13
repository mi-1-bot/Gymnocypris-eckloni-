import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from torchvision.ops import generalized_box_iou_loss

# --------------------------
# 数据增强
# --------------------------
def get_train_transform():
    return A.Compose([
        A.Resize(640, 640),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def get_val_transform():
    return A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# --------------------------
# 数据集类
# --------------------------
class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        bboxes, class_labels = [], []
        with open(self.label_paths[idx], 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                bboxes.append(list(map(float, parts[1:])))
                class_labels.append(class_id)

        if self.transform:
            transformed = self.transform(image=img, bboxes=bboxes, class_labels=class_labels)
            img = transformed['image']
            bboxes = transformed['bboxes']
            class_labels = transformed['class_labels']

        return img, (torch.tensor(bboxes), torch.tensor(class_labels))

# --------------------------
# 注意力机制模块
# --------------------------
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.conv(x)

# --------------------------
# FocalLoss + IoU Loss
# --------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return loss.mean()

class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = FocalLoss()
        self.box_loss = lambda pred, tgt: generalized_box_iou_loss(pred, tgt, reduction="mean")

    def forward(self, preds, targets):
        pred_boxes, pred_cls = preds[:, :4], preds[:, 4:]
        target_boxes, target_cls = targets[:, :4], targets[:, 4:]
        return self.box_loss(pred_boxes, target_boxes) + self.cls_loss(pred_cls, target_cls)

# --------------------------
# 训练主函数
# --------------------------
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 模型
    model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')
    model.load('yolo11n.pt')

    # 替换骨干网络
    backbone = EfficientNet.from_pretrained('efficientnet-b3')
    model.model.backbone = backbone

    # 插入注意力模块（仅后几层）
    for i in range(len(model.model.model) - 3, len(model.model.model)):
        if isinstance(model.model.model[i], nn.Conv2d):
            model.model.model[i] = nn.Sequential(model.model.model[i], AttentionBlock(model.model.model[i].out_channels))

    model.loss_fn = CustomLoss()

    # 数据路径（替换为实际路径）
    train_image_paths = [os.path.join("D:/Deep Learning/YOLOv11/MACHAO/VOCdevkit/train/images", f) for f in os.listdir("D:/Deep Learning/YOLOv11/MACHAO/VOCdevkit/train/images")]
    train_label_paths = [os.path.join("D:/Deep Learning/YOLOv11/MACHAO/VOCdevkit/train/labels", f) for f in os.listdir("D:/Deep Learning/YOLOv11/MACHAO/VOCdevkit/train/labels")]
    val_image_paths = [os.path.join("D:/Deep Learning/YOLOv11/MACHAO/VOCdevkit/val/images", f) for f in os.listdir("D:/Deep Learning/YOLOv11/MACHAO/VOCdevkit/val/images")]
    val_label_paths = [os.path.join("D:/Deep Learning/YOLOv11/MACHAO/VOCdevkit/val/labels", f) for f in os.listdir("D:/Deep Learning/YOLOv11/MACHAO/VOCdevkit/val/labels")]

    train_dataset = YOLODataset(train_image_paths, train_label_paths, transform=get_train_transform())
    val_dataset = YOLODataset(val_image_paths, val_label_paths, transform=get_val_transform())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 训练
    model.train(
        data='data.yaml',
        epochs=200,
        imgsz=640,
        device=device,
        optimizer='AdamW',
        lr0=0.0005,
        lrf=0.00005,
        momentum=0.937,
        weight_decay=0.001,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=0.05,
        cls=0.6,
        dfl=0.1,
        label_smoothing=0.05,
        patience=0 # 早停
    )
