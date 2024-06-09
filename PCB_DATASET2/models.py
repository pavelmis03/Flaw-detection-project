'''
Файл для описания классов
'''

import numpy as np

import cv2

import torch

# !!!разобраться с классом, прокомментировать!!!
class CustomDataBase(object):
    def __init__(self, df, IMG_DIR, transforms):
        # dataframe
        self.df = df
        # директория изображений
        self.img_dir = IMG_DIR
        self.image_ids = self.df['file'].unique().tolist()
        self.transforms = transforms
        # цифровые обозначения дефектов
        self.classes_la = {"missing_hole": 0, "mouse_bite": 1, "open_circuit": 2, "short": 3, 'spur': 4,
                      'spurious_copper': 5}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, img):
        # получить изображение можно по номеру или по имени
        if (img.isdigit()):
            # если по номеру, выбираем из массива в пределах индексов
            img = max(min(0, img), len(self.image_ids))
        else:
            # если по имени, проверяем, что такое имя есть или выбираем первое изображение
            try:
                img = self.image_ids.index(img)
            except ValueError:
                print("\nТакого изображения нет в списке доступных для анализа!\nВыбрано случайное изображение\n")
                img = 0
        image_id = self.image_ids[img]

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
        target['image_id'] = torch.tensor([img])
        target['area'] = torch.as_tensor(area, dtype=torch.float32)
        target['iscrowd'] = torch.zeros(len(self.classes_la), dtype=torch.int64)

        if (self.transforms):
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': labels
            }

            sample = self.transforms(**sample)
            image = sample['image']

            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)

        return torch.tensor(image), target, image_id

    def __str__(self):
        # вывод класса
        return f"\n\nДатафрейм: {self.df}\n\n{self.image_ids}\n\n{self.transforms}"