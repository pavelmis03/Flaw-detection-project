'''
Файл для описания классов
'''

import numpy as np

import cv2

import torch

# !!!разобраться с классом, прокомментировать!!!
class CustomDataBase(object):
    def __init__(self, df, IMG_DIR, transforms):
        # Инициализация класса. Этот конструктор инициализирует объект CustomDataBase с датафреймом df,
        # директорией изображений IMG_DIR и преобразованиями transforms.

        # Сохраняем переданный датафрейм
        self.df = df

        # Сохраняем директорию с изображениями
        self.img_dir = IMG_DIR

        # image_ids - список уникальных идентификаторов файлов из датафрейма
        # Получаем список уникальных идентификаторов изображений из столбца 'file' датафрейма
        self.image_ids = self.df['file'].unique().tolist()

        # Сохраняем переданные трансформации для изображений
        self.transforms = transforms

        # Словарь с классами дефектов и их цифровыми обозначениями
        # classes_la - словарь сопоставления меток классов и числовых обозначений
        self.classes_la = {"missing_hole": 0, "mouse_bite": 1, "open_circuit": 2, "short": 3, 'spur': 4,
                      'spurious_copper': 5}

    def __len__(self):
        # Возвращает количество уникальных изображений в датасете
        return len(self.image_ids)

    def __getitem__(self, img):
        # Метод для получения изображения и его метаданных по номеру или имени
        # Получает изображение и соответствующие аннотации по индексу или имени файла
        # Выбирает подкаталог в зависимости от наличия ключевых слов в имени изображения
        # Загружает изображение, конвертирует его в RGB и нормализует
        # Получает координаты и площади bounding box'ов, а также метки классов
        # Формирует словарь target с ключевыми данными для обучения модели
        # Применяет аугментации, если они указаны в transforms.

        # Если img - число, оно будет строкой, проверяем и обрабатываем как индекс
        if (img.isdigit()):
            # Убедимся, что индекс в допустимых пределах
            img = max(min(0, img), len(self.image_ids))
        else:
            # Если img - строка, ищем его в списке идентификаторов изображений
            try:
                img = self.image_ids.index(img)
            except ValueError:
                # Если такого идентификатора нет, выбрасываем предупреждение и выбираем первое изображение
                print("\nТакого изображения нет в списке доступных для анализа!\nВыбрано случайное изображение\n")
                img = 0
        # Получаем идентификатор изображения по индексу
        image_id = self.image_ids[img]

        # Определяем поддиректорию на основе идентификатора изображения
        a = ''
        if ("missing" in image_id.split('_')):
            a = 'Missing_hole/'
        elif ("mouse" in image_id.split('_')):
            a = 'Mouse_bite/'
        elif ("open" in image_id.split('_')):
            a = 'Open_circuit/'
        elif ("short" in image_id.split('_')):
            a = 'Short/'
        elif ("spur" in image_id.split('_')):
            a = 'Spur/'
        elif ("spurious" in image_id.split('_')):
            a = 'Spurious_copper/'
        # Фильтруем строки датафрейма, соответствующие текущему изображению
        image_values = self.df[self.df['file'] == image_id]
        # Загружаем изображение из файла
        image = cv2.imread(self.img_dir + a + image_id + ".jpg", cv2.IMREAD_COLOR)
        # Преобразуем изображение из BGR в RGB и нормализуем
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        # Получаем координаты bounding box'ов из датафрейма
        boxes = image_values[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
        # Вычисляем площади bounding box'ов
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Получаем метки классов дефектов и преобразуем их в тензоры
        labels = image_values["class"].values
        labels = torch.tensor(labels)

        # Создаем словарь с аннотациями для текущего изображения
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([img])
        target['area'] = torch.as_tensor(area, dtype=torch.float32)
        target['iscrowd'] = torch.zeros(len(self.classes_la), dtype=torch.int64)

        # Применяем переданные трансформации, если они есть
        if (self.transforms):
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }

            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        # Возвращаем тензор изображения, аннотации и идентификатор изображения
        return torch.tensor(image), target, image_id

    def __str__(self):
        # Метод для строкового представления объекта класса
        # Возвращает строковое представление объекта, включая датафрейм, список идентификаторов изображений и преобразования
        return f"\n\nДатафрейм: {self.df}\n\n{self.image_ids}\n\n{self.transforms}"