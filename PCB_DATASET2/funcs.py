'''
Функции для обработки ДС, файлов и др.
'''

import xml.etree.ElementTree as ET
import os

import random

from tkinter import filedialog

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as immg

import albumentations as A
from albumentations.pytorch import ToTensorV2

# чтение файлов аннотаций
def readXMLAnnotation(path_an):
    # получение всех файлов аннотаций
    all_files = []  # файлы аннотаций
    for path, subdirs, files in os.walk(path_an):
        #print([path, subdirs, files])
        for name in files:
            all_files.append(os.path.join(path, name))

    # исправление путей файлов аннотаций
    for i in range(len(all_files)):
        all_files[i] = all_files[i].replace("\\", "/")

    all_files.sort()

    return all_files


# заполнение ДС информацией из файлов аннотаций
# !!!разобраться, как работает парсинг, прокомментировать!!!
def parseXMLToDS(dataset, all_files):
    for anno in all_files:
        tree = ET.parse(anno)

        for elem in tree.iter():
            #print(elem)

            if 'size' in elem.tag:
                #print('[size] in elem.tag ==> list(elem)\n'), print(list(elem))
                for attr in list(elem):
                    if 'width' in attr.tag:
                        width = int(round(float(attr.text)))
                    if 'height' in attr.tag:
                        height = int(round(float(attr.text)))

            if 'object' in elem.tag:
                #print('[object] in elem.tag ==> list(elem)\n'), print(list(elem))
                for attr in list(elem):

                    #print('attr = %s\n' % attr)
                    if 'name' in attr.tag:
                        name = attr.text
                        dataset['class'] += [name]
                        dataset['width'] += [width]
                        dataset['height'] += [height]
                        dataset['file'] += [anno.split('/')[-1][0:-4]]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                xmin = int(round(float(dim.text)))
                                dataset['xmin'] += [xmin]
                            if 'ymin' in dim.tag:
                                ymin = int(round(float(dim.text)))
                                dataset['ymin'] += [ymin]
                            if 'xmax' in dim.tag:
                                xmax = int(round(float(dim.text)))
                                dataset['xmax'] += [xmax]
                            if 'ymax' in dim.tag:
                                ymax = int(round(float(dim.text)))
                                dataset['ymax'] += [ymax]
    return dataset

# открытие через проводник изображения для вывода ограничивающей рамки вокруг дефектов
def openImgFile():
    # root = Tk()
    # root.title("Выберете изображение для анализа")
    # root.geometry("250x200")

    # root.grid_rowconfigure(index=0, weight=1)
    # root.grid_columnconfigure(index=0, weight=1)
    # root.grid_columnconfigure(index=1, weight=1)

    # открываем файл в текстовое поле
    filepath = filedialog.askopenfilename()

    # !!!проверка на правильность директории!!!
    #while (flag):
        #filepath = filedialog.askopenfilename()

    filepath = filepath.split("/")
    filepath = filepath[-1].split(".")
    return filepath[0]

# вывод ограничивающей рамки
def drawPlotImage(image_name, path_img, name, df_grp):
    image_group = df_grp.get_group(image_name)
    bbox = image_group.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']]
    if ("missing" in name.split('_')):
        path_img += '/Missing_hole/'
    if ("mouse" in name.split('_')):
        path_img += '/Mouse_bite/'
    if ("open" in name.split('_')):
        path_img += '/Open_circuit/'
    if ("short" in name.split('_')):
        path_img += '/Short/'
    if ("spur" in name.split('_')):
        path_img += '/Spur/'
    if ("spurious" in name.split('_')):
        path_img += '/Spurious_copper/'

    img = immg.imread(path_img + "" + name + '.jpg')
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(img, cmap='binary')
    for i in range(len(bbox)):
        box = bbox.iloc[i].values
        # print(box)
        x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = matplotlib.patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none', )
        # ax.text(*box[:2], image_group["class"].values, verticalalignment='top', color='white', fontsize=13, weight='bold')
        ax.add_patch(rect)
    plt.show()

    return name

# !!!разобраться, что делают эти строки, прокомментировать!!!
def getTrainTransform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# !!!разобраться, что делают эти строки, прокомментировать!!!
def getValidTransform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# !!!разобраться, что делает функция, прокомментировать!!!
# обозначение дефектов
def titleDefects(fcb_dataset, image_name):
    # img, tar, _ = fcb_dataset[random.randint(0, 50)]
    img, tar, _ = fcb_dataset[image_name]
    # print(img, tar)
    bbox = tar['boxes']
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.imshow(img.permute(1, 2, 0).cpu().numpy())
    for j in tar["labels"].tolist():
        classes_la = {0: "missing_hole", 1: "mouse_bite", 2: "open_circuit", 3: "short", 4: 'spur', 5: 'spurious_copper'}
        l = classes_la[j]
        for i in range(len(bbox)):
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2] - box[0], box[3] - box[1]
            rect = matplotlib.patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none',)
            ax.text(x, y - 35, l, verticalalignment='top', color='red', fontsize=13, weight='bold')
            ax.add_patch(rect)
        plt.show()


def getTupleFromZip(batch):
    return tuple(zip(*batch))

