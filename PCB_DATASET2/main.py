import os
import sys
import warnings

import pandas as pd
import numpy as np
import random

from tqdm import tqdm
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN

import argparse
import glob
import cv2
import ast

# подключение наших модулей
# функции обработки данных
from funcs import parseXMLToDS
from funcs import readXMLAnnotation
from funcs import openImgFile
from funcs import drawPlotImage

# функции вывода информации
from printInfo import printAnnotationPath

# !!!написать документацию по функциям !!!
# !!!разобраться, что делают эти строки, прокомментировать!!!
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
warnings.filterwarnings("ignore")


# I Generate CSV


print("\nЧасть 1: Обработка XML аннотаций\n\n")
# пути аннотаций и изображений
path_an = "PCB_DATASET/Annotations"
path_img = "PCB_DATASET/images"
print("Путь аннотаций в датасете: ", path_an, "", sep="\n")

dataset = {
    "xmin": [],
    "ymin": [],
    "xmax": [],
    "ymax": [],
    "class": [],
    "file": [],
    "width": [],
    "height": [],
}

# !!!разобраться, как работает парсинг, прокомментировать!!!
all_files = readXMLAnnotation(path_an)

# красивый вывод файлов путей аннотаций
# printAnnotationPath(all_files, path_an)

# пока пустой ДС
print("Датасет до заполнения: ")
# print(type(dataset))
print(dataset)

# собранные данные из файлов аннотаций
dataset = parseXMLToDS(dataset, all_files)
# print("\n\nРазобранным xml файлы в словаре dataset: ")
# print(dataset, "\n\n")
# датафрейм пандас по полученным данным
data = pd.DataFrame(dataset)
# print("Пандавское представление полученных данных: ")
# print(data, "\n\n")


# II Reading the CSV file


# разделим данные на обучающие и тестовые, используя 80% данных для обучения,
# а оставшиеся 20% - для тестирования
train, test = train_test_split(data, shuffle=True, test_size=0.2, random_state=34)

print("\n\nЧасть 2: Чтение и преобразование полученного дата фрейма\n\n")
print("Размеры обучающего и тестового ДС")
print(train.shape, test.shape)
print("\n\nШапка обучающего ДС: ")
print(train.head())
print("\n\nШапка тестового ДС: ")
print(test.head())

# меняем название классов на цифровые обозначения
classes_la = {"missing_hole": 0, "mouse_bite": 1, "open_circuit": 2, "short": 3, 'spur': 4, 'spurious_copper': 5}

train["class"] = train["class"].apply(lambda x: classes_la[x])
test["class"] = test["class"].apply(lambda x: classes_la[x])

print("\n\nШапка обучающего ДС после преобразования: ")
print(train.head())
print("\n\nШапка тестового ДС после преобразования: ")
print(test.head())


# III Visualisation


print("\n\nЧасть 3: Визуализация полученных данных\n\n")
# PJC (deep copy)
df = train.copy()

df_grp = df.groupby(['file'])
# print(*df_grp)
# print(df_grp.size())

# !!!сделать проверку на правильность имени файла!!!
'''
файл должен быть из папки
PCB_DATASET\images\Missing_hole или Mouse_bite или Open_circuit или Short или Spur или Spurious_copper\
и называться 
??_missing_hole_??.jpg
??_mouse_bite_??.jpg
и т.д.
'''
# Открытие файла изображения для анализа
image_name = openImgFile()
print("Вы выбрали изображение для анализа: ")
print(image_name, "\n")

# !!!разобраться, что делают эти строки, прокомментировать!!!
image_group = df_grp.get_group(image_name)
print(image_group)

# получаем информацию по ограничивающим рамкам для данного изображения
bbox = image_group.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']]
print([bbox, type(bbox)])

# создаем график matplotlib с изображением, где отмечены дефекты
# !!!разобраться, как работает функция, прокомментировать!!!
name = drawPlotImage(image_name, path_img, image_name, df_grp)

'''
name = train.file[500]
name = drawPlotImage(image_name, path_img, image_name, df_grp)

name = train.file[100]
name = drawPlotImage(image_name, path_img, image_name, df_grp)

name = train.file[105]
name = drawPlotImage(image_name, path_img, image_name, df_grp)
'''

'''
# IV. Creating Custom database


class fcbData(object):
    def __init__(self, df, IMG_DIR, transforms):
        self.df = df
        self.img_dir = IMG_DIR
        self.image_ids = self.df['file'].unique().tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        a = ''
        if "missing" in image_id.split('_'):
            a = 'Missing_hole/'
        elif "mouse" in image_id.split('_'):
            a = 'Mouse_bite/'
        elif "open" in image_id.split('_'):
            a = 'Open_circuit/'
        elif "short" in image_id.split('_'):
            a = 'Short/'
        elif "spur" in image_id.split('_'):
            a = 'Spur/'
        elif "spurious" in image_id.split('_'):
            a = 'Spurious_copper/'
        image_values = self.df[self.df['file'] == image_id]
        image = cv2.imread(self.img_dir + a + image_id + ".jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        boxes = image_values[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        labels = image_values["class"].values
        labels = torch.tensor(labels)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([idx])
        target['area'] = torch.as_tensor(area, dtype=torch.float32)
        target['iscrowd'] = torch.zeros(len(classes_la), dtype=torch.int64)

        if self.transforms:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }

            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return torch.tensor(image), target, image_id

def get_train_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

path = "PCB_DATASET/images/"
fcb_dataset = fcbData(df, path, get_train_transform())

print(type(fcb_dataset[0]), len(fcb_dataset[0]), type(fcb_dataset[0][0]), type(fcb_dataset[0][1]), type(fcb_dataset[0][2]))

print([fcb_dataset[0][0], fcb_dataset[0][1], fcb_dataset[0][2]])

img, tar, _ = fcb_dataset[random.randint(0,50)]
bbox = tar['boxes']
fig,ax = plt.subplots(figsize=(18,10))
ax.imshow(img.permute(1,2,0).cpu().numpy())
for j in tar["labels"].tolist():
    classes_la = {0:"missing_hole", 1: "mouse_bite", 2:"open_circuit",3: "short", 4:'spur',5:'spurious_copper'}
    l = classes_la[j]
    for i in range(len(bbox)):
        box = bbox[i]
        x,y,w,h = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = matplotlib.patches.Rectangle((x,y),w,h,linewidth=2,edgecolor='r',facecolor='none',)
        ax.text(*box[:2], l, verticalalignment='top', color='red', fontsize=13, weight='bold')
        ax.add_patch(rect)
    plt.show()

print(len(df))

image_ids = df['file'].unique()
valid_ids = image_ids[-665:]
train_ids = image_ids[:-665]
valid_df = df[df['file'].isin(valid_ids)]
train_df = df[df['file'].isin(train_ids)]

print(train_df.shape, valid_df.shape)

'''

# Dataloader

'''

print(path)

def collate_fn(batch):
    return tuple(zip(*batch))

train_dataset = fcbData(df, path, get_train_transform())
valid_dataset = fcbData(df, path, get_valid_transform())

# split the dataset in train and test set
indices = torch.randperm(len(train_dataset)).tolist()

train_data_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=6,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=6,
    collate_fn=collate_fn
)

print(next(iter(train_data_loader)))

## num_classes = 6 # + background
num_classes = 6

# load a model; pre-trained on COCO
# .. fpn = 'feature pyramid network'
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.0005,)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 1



# VI. Training and evaluation



print(train_data_loader)

best_epoch = 0
min_loss = sys.maxsize

for epoch in range(num_epochs):
    tk = tqdm(train_data_loader)
    model.train();
    for images, targets, image_ids in tk:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        tk.set_postfix(train_loss=loss_value)
    tk.close()

    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_value}")

    # validation
    model.eval();
    with torch.no_grad():
        tk = tqdm(valid_data_loader)
        for images, targets, image_ids in tk:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            val_output = model(images)
            val_output = [{k: v.to(device) for k, v in t.items()} for t in val_output]
            IOU = []
            for j in range(len(val_output)):
                a, b = val_output[j]['boxes'].cpu().detach(), targets[j]['boxes'].cpu().detach()
                chk = torchvision.ops.box_iou(a, b)
                res = np.nanmean(chk.sum(axis=1) / (chk > 0).sum(axis=1))
                IOU.append(res)
            tk.set_postfix(IoU=np.mean(IOU))
        tk.close()

torch.save(model.state_dict(), 'pcbdetection.pt')



# VII. Evaluation



y_true =[]
y_pred = []
for i in range(50):
    img,target,_ = valid_dataset[i]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])[0]
        y_true.append(target['labels'][0])
        y_pred.append(prediction['labels'][0])

print(y_pred)

yy_pred = []
for v in y_pred:
    yy_pred.append(v.cpu())

print(yy_pred)
print(y_true)

confusion_matrix(y_true, yy_pred)

# classification_report로 평가 지표 확인하기
# .. https://blog.naver.com/PostView.naver?blogId=hannaurora&logNo=222498671200&parentCategoryNo=&categoryNo=41&viewDate=&isShowPopularPosts=true&from=search
print(classification_report(y_true, yy_pred))

# Define the device for inference (CPU or GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

defect_names = {
1: "Open Circuit",
2: "Short Circuit",
3: "Mouse Bite",
4: "Spur",
5: "Copper Trace Cut"
}
# Load the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=6)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=6)
model.load_state_dict(torch.load('/kaggle/working/pcbdetection.pt'))
model.eval()
model.to(device)

# Load the image
image_path = '/kaggle/input/pcb-defects/PCB_DATASET/images/Mouse_bite/01_mouse_bite_01.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image / 255.0

# Define the transformation to apply to the image
transform = T.Compose([T.ToTensor()])

# Apply the transformation to the image
image = transform(image).to(device)

# Predict the bounding boxes and labels for the image
image = image.float()
outputs = model([image])
boxes = outputs[0]['boxes'].detach().cpu().numpy()
labels = outputs[0]['labels'].detach().cpu().numpy()

# Visualize the image and the predicted bounding boxes
fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(image.permute(1, 2, 0).cpu().numpy())
for box, label in zip(boxes, labels):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    rect = matplotlib.patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x1-20, y2 + 50, defect_names[label], fontsize=12, color='g', backgroundcolor='w')
plt.show()
'''
