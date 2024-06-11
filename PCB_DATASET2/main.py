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
from funcs import (parseXMLToDS, readXMLAnnotation, openImgFile, drawPlotImage, getTrainTransform)
from funcs import (getValidTransform, titleDefects, getTupleFromZip, trainingModel, visualiseTrainingResults)

# функции вывода информации
from printInfo import printAnnotationPath

# классы
from models import CustomDataBase

def main():
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

    # Открытие файла изображения для анализа
    image_name = openImgFile()

    print("Вы выбрали изображение для анализа: ")
    print(image_name, "\n")

    # !!!разобраться, что делают эти строки, прокомментировать!!!
    image_group = df_grp.get_group(image_name)
    print(image_group, "\n")

    # получаем информацию по ограничивающим рамкам для данного изображения
    bbox = image_group.loc[:, ['xmin', 'ymin', 'xmax', 'ymax']]
    print(bbox, "\n", type(bbox), "\n")

    # создаем график matplotlib с изображением, где отмечены дефекты
    # !!!разобраться, как работает функция, прокомментировать!!!
    # name = drawPlotImage(image_name, path_img, image_name, df_grp)

    # вывод дополнительного изображения
    # name = train.file[500]
    # name = drawPlotImage(image_name, path_img, image_name, df_grp)


    # IV. Creating Custom database


    print("\nЧасть 4: Создание пользовательской БД и обозначение дефектов\n\n")
    # !!!разобраться, как работает класс, прокомментировать!!!
    custom_dataset = CustomDataBase(df, path_img + "/", getTrainTransform())

    # датасет от тензора
    # print(custom_dataset)
    # print(type(custom_dataset[0]), len(custom_dataset[0]), type(custom_dataset[0][0]), type(custom_dataset[0][1]), type(custom_dataset[0][2]))
    # print([custom_dataset[0][0], custom_dataset[0][1], custom_dataset[0][2]])

    # подписываем дефекты
    titleDefects(custom_dataset, image_name)

    print("Длина датафрейма: ")
    print(len(df))

    # на всякий случай избавляемся от дубляжей
    image_ids = df['file'].unique()

    # разделяем изображения и датафрейм на две части: для обучения и для проверки
    valid_ids = image_ids[-665:]
    train_ids = image_ids[:-665]
    valid_df = df[df['file'].isin(valid_ids)]
    train_df = df[df['file'].isin(train_ids)]

    print("Размеры датафремов для обучения и для проверки: ")
    print(train_df.shape, valid_df.shape)


    # Dataloader


    print("\n\nЧасть 5: Загрузка и настройка предобученной модели COCO\n\n")

    # сгенерированные датасеты для обучения и проверки
    train_dataset = CustomDataBase(df, path_img + "/", getTrainTransform())
    valid_dataset = CustomDataBase(df, path_img + "/", getValidTransform())

    # перемешиваем данные в нашем датасете для увеличения независимости результатов
    indices = torch.randperm(len(train_dataset)).tolist()

    # загружаем в память датасеты методами параллельной загрузки
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        collate_fn=getTupleFromZip
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        collate_fn=getTupleFromZip
    )

    # должен печатать следующий блок данных, но перезапускает прогу:))
    # print(next(iter(train_data_loader)))

    # ЗАГРУЗКА И НАСТРОЙКА МОДЕЛИ ОБНАРУЖЕНИЯ, ПРЕДОБУЧЕННОЙ НА COCO

    # Большинство предварительно обученных моделей обучаются с помощью фонового класса,
    # мы включим его в нашу модель, так что в этом случае количество наших классов составит 6
    num_classes = 6

    # загрузите модель, предварительно обученную на COCO
    # fpn = 'feature pyramid network'
    # !!!узнать подробности, разобраться, как работает, написать вывод в файлике ворда!!!
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # получение количества входных признаков для классификатора
    # !!!узнать подробности, разобраться, как работает, написать вывод в файлике ворда!!!
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # заменим предварительно подготовленный заголовок на новый
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # !!!узнать подробности, разобраться, как работает, написать вывод в файлике ворда!!!

    # !!!узнать подробности, разобраться, как работает, комментарии!!!
    device = torch.device('cuda') if (torch.cuda.is_available()) else torch.device('cpu')

    # !!!узнать подробности, разобраться, как работает, комментарии!!!
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=0.0005,)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # количество итераций обучения
    num_epochs = 1


    # VI. Training and evaluation


    print("\n\nЧасть 6: Обучение модели для работы с нашими данными\n\n")
    # print("Загрузчик tourch: ")
    # print(train_data_loader)

    # лучшая итерация
    best_epoch = 0
    #для поиска минимума потерь
    min_loss = sys.maxsize

    # обучение модели
    trainingModel(model, train_data_loader, num_epochs, valid_data_loader, device, optimizer, lr_scheduler)

    print("\nОбучение модели завершено!\n")
    print(torch.save(model.state_dict(), 'pcbdetection.pt'))


    # VII. Evaluation


    print("\n\nЧасть 7: Оценка качества обучения модели\n\n")
    y_true = []
    y_pred = []

    # проверяем работу модели на той части датасета, которую мы не использовали для обучения
    for i in range(50):
        img, target, _ = valid_dataset[i]
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])[0]

    # получаем из модели столбцы с именами файлов
    y_true.append(target['labels'][0])
    y_pred.append(prediction['labels'][0])

    print("Массив полученных после обучения модели изображений: ")
    print(y_pred, "\n")

    # добавляем время обработки процессором
    yy_pred = []
    for v in y_pred:
        yy_pred.append(v.cpu())

    print("Затраченное на обработку изображений процессорное время: ")
    print(yy_pred, "\n")
    print(y_true, "\n")

    # оценка точности классификации
    print("Матрица спутанности: ")
    print(confusion_matrix(y_true, yy_pred))

    # print(classification_report(y_true, yy_pred))
    
    # Определите устройство для вывода (CPU или GPU-процессор)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # наименования дефектов
    defect_names = {
        1: "Open Circuit",
        2: "Short Circuit",
        3: "Mouse Bite",
        4: "Spur",
        5: "Copper Trace Cut"
    }

    # загружаем модель
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=6)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=6)
    model.load_state_dict(torch.load('pcbdetection.pt'))
    model.eval()
    model.to(device)
    
    # загружаем изображения
    image_name = openImgFile()

    print("Вы выбрали изображение для анализа: ")
    print(image_name, "\n")
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    
    # преобразования, которые будут применены к изображению
    transform = T.Compose([T.ToTensor()])
    
    # применим преобразования к изображению
    image = transform(image).to(device)
    
    # предсказываем ограничивающие рамки и надписи для изображения
    image = image.float()
    outputs = model([image])
    boxes = outputs[0]['boxes'].detach().cpu().numpy()
    labels = outputs[0]['labels'].detach().cpu().numpy()
    
    # визуализируем изображение и предсказанные ограничивающие рамки
    visualiseTrainingResults(image, boxes, labels, defect_names)


if __name__ == "__main__":
    # freeze_support()
    main()