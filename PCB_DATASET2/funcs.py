'''
Функции для обработки ДС, файлов и др.
'''

import xml.etree.ElementTree as ET
import os

import numpy as np

from tqdm import tqdm
from tqdm.notebook import tqdm

import torch
import torchvision

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
def parseXMLToDS(dataset, all_files):
    # Проходим по каждому файлу в списке all_files
    for anno in all_files:
        # Парсим (делаем удобный стиль для нас) XML файл и создаем объект ElementTree
        tree = ET.parse(anno)

        # Итерация по каждому элементу в дереве XML
        for elem in tree.iter():
            #print(elem)

            # Если элемент содержит информацию о размере изображения
            if 'size' in elem.tag:
                #print('[size] in elem.tag ==> list(elem)\n'), print(list(elem))
                for attr in list(elem):
                    if 'width' in attr.tag:
                        width = int(round(float(attr.text)))
                    if 'height' in attr.tag:
                        height = int(round(float(attr.text)))

            # Если элемент содержит информацию об объекте
            if 'object' in elem.tag:
                #print('[object] in elem.tag ==> list(elem)\n'), print(list(elem))
                for attr in list(elem):

                    # Извлекаем имя объекта и добавляем его в словарь dataset
                    #print('attr = %s\n' % attr)
                    if 'name' in attr.tag:
                        name = attr.text
                        dataset['class'] += [name]
                        dataset['width'] += [width]
                        dataset['height'] += [height]
                        dataset['file'] += [anno.split('/')[-1][0:-4]]

                    # Извлекаем координаты ограничивающего прямоугольника и добавляем их в словарь dataset
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

    # Возвращаем обновленный словарь dataset
    return dataset

# открытие через проводник изображения для вывода ограничивающей рамки вокруг дефектов
def openImgFile():
    # Возможные типы дефектов
    defects = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']
    # Возможные типы открываемых файлов
    file_types = ['jpg', 'png']
    # Логическая переменная для выхода из цикла проверки
    bool_exit = False
    # Результатное имя файла
    result = ''
    while (not(bool_exit)):
        # Открываем файл
        filepath = filedialog.askopenfilename().lower()
        # разбираем путь
        file_list = filepath.split('/')
        if ('pcb_dataset' in file_list) and ('images' in file_list):
            # Проверяем, что в пути к файлу есть один из типов дефектов
            filename = file_list[-1].split('.')[0]
            if (filename[3:-3] in defects):
                if (file_list[-1].split('.')[1] in file_types):
                    # Проверяем на сопадение типа файла
                    bool_exit = True
                    result = filename
                else:
                    print('Неверный тип выбранного файла')
            else:
                print('Неверное наименование выбранного файла')
        else:
            print('Неверная директория выбранного файла')
    return result

    # filepath = filepath.split("/")
    # filepath = filepath[-1].split(".")
    # return filepath[0]

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

# Определяем функцию getTrainTransform, которая возвращает объект Compose из библиотеки albumentations.
# Этот объект Compose содержит одно преобразование ToTensorV2, которое преобразует изображение в тензор PyTorch.
def getTrainTransform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# Определяем функцию getValidTransform, которая не принимает никаких аргументов.
# Зачем это нужно: Для создания и возврата набора трансформаций, которые будут применены к данным при их валидации (проверке).
def getValidTransform():
    # Что делает: Возвращает объект A.Compose, который является композицией (сочетанием) трансформаций, используемых для обработки изображений и их метаданных.
    # Зачем это нужно: Для последовательного применения ряда трансформаций к изображениям и соответствующим меткам в процессе валидации.
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


# разорхиватор
def getTupleFromZip(batch):
    return tuple(zip(*batch))

# обучение модели
def trainingModel(model, train_data_loader, num_epochs, valid_data_loader, device, optimizer, lr_scheduler):
    for epoch in range(num_epochs):
        # для красивого отображения загрузки
        tk = tqdm(train_data_loader)
        model.train()
        loss_value = 0
        # обучение на данных для обучения
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

        # цикл обучения и процент ошибок
        print(f"Epoch #{epoch} loss: {loss_value}")

        # validation
        model.eval()
        with torch.no_grad():
            tk = tqdm(valid_data_loader)
            # тестирование полученной модели
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

# визуализируем изображение и предсказанные ограничивающие рамки
def visualiseTrainingResults(image, boxes, labels, defect_names):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        rect = matplotlib.patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1-20, y2 + 50, defect_names[label], fontsize=12, color='g', backgroundcolor='w')
    plt.show()