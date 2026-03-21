import torch.utils.data as data
import cv2
import numpy as np
import torch

class WiderFaceDataset(data.Dataset):
    def __init__(self,txt_path: str,preproc) -> None:
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear() #TODO
                path = line[2:]
                path = txt_path.replace('label.txt','images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)
        f.close()

    def __len__(self):
        return len(self.imgs_path)
    
    def __getitem__(self, index):
        while True:
            img = cv2.imread(self.imgs_path[index])
            labels = self.words[index]

            if img is not None and len(labels) > 0:
                break

            index = np.random.randint(0, self.__len__())

        labels = self.words[index]
        annotations = np.zeros((len(labels),15),dtype=np.float32)
        if len(labels) == 0:
            return self.__getitem__(np.random.randint(0, self.__len__()))
        for idx, label in enumerate(labels):
            annotations[idx, :4] = label[:4]
            annotations[idx, [2, 3]] += annotations[idx, [0, 1]]

            for i in range(5):
                annotations[idx, 4 + 2 * i] = label[4 + 3 * i]
                annotations[idx, 5 + 2 * i] = label[5 + 3 * i]
            
            if annotations[idx, 4] < 0:
                annotations[idx, 14] = -1
            else:
                annotations[idx, 14] = 1
        
        if self.preproc is not None:
            img, annotations = self.preproc(img, annotations)

        return torch.from_numpy(img), torch.from_numpy(annotations)

def detection_collate(batch):
    imgs = []
    targets = []
    
    for sample in batch:
        imgs.append(sample[0])
        targets.append(sample[1].float()) #TODO
        
    imgs = torch.stack(imgs, 0)
    
    return imgs, targets
