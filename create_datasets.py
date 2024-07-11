"""
THIS PROGRAM CONTAINS ALL THE CLASSES THAT CREATE THE REQUIRED IMAGE SETS TO DO A TRAINING, VALIDATION OR TEST
These classes will be called by training and test programmes
"""


from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dset
from PIL import Image
import pandas as pd
import os


csvDir = "CSV"

trainDir = os.path.join("DATASETS", "FRIBURGO", "Train")
folderDataset = dset.ImageFolder(root=trainDir)
rooms = folderDataset.classes


def get_coords(imageDir):
    idxX, idxY, idxA = imageDir.index('_x'), imageDir.index('_y'), imageDir.index('_a')
    x, y = float(imageDir[idxX + 2:idxY]), float(imageDir[idxY + 2:idxA])
    return [x, y]


def process_image(image, tf):
    image = Image.open(image).convert("RGB")
    if tf is not None:
        image = tf(image)
    return image


class Train(Dataset):

    def __init__(self, imageFolderDataset, rPos=0.5, rNeg=0.5, transform=transforms.ToTensor(), should_invert=False):

        self.rPos, self.rNeg = rPos, rNeg
        rPos, rNeg = str(self.rPos), str(self.rNeg)

        trainCSV = pd.read_csv(csvDir + '/TrainrPos' + rPos + 'rNeg' + rNeg + '.csv')

        print(f"CSV utilizado: '/TrainrPos' + {rPos} + 'rNeg' + {rNeg} + '.csv'")

        self.imgsAnc, self.imgsPos, self.imgsNeg = trainCSV['ImgAnc'], trainCSV['ImgPos'], trainCSV['ImgNeg']
        self.imageFolderDataset, self.transform, self.should_invert = imageFolderDataset, transform, should_invert

    def __getitem__(self, index):

        imgAnc, imgPos, imgNeg = self.imgsAnc[index], self.imgsPos[index], self.imgsNeg[index]

        anchor = process_image(imgAnc, self.transform)
        positive = process_image(imgPos, self.transform)
        negative = process_image(imgNeg, self.transform)

        return anchor, positive, negative

    def __len__(self):
        return len(self.imgsAnc)


class Validation(Dataset):

    def __init__(self, imageFolderDataset, transform=transforms.ToTensor()):

        valCSV = pd.read_csv(csvDir + '/Validation.csv')

        self.imgList, self.coords = valCSV['Img'], valCSV['Coords']
        self.imageFolderDataset, self.transform = imageFolderDataset, transform

    def __getitem__(self, index):

        imgVal, coordsVal = self.imgList[index], self.coords[index]

        imgVal = process_image(imgVal, self.transform)

        return imgVal, coordsVal

    def __len__(self):
        return len(self.imgList)


class Test(Dataset):

    def __init__(self, illumination, imageFolderDataset, transform=transforms.ToTensor()):

        testCSV = pd.read_csv(csvDir + '/Test' + illumination + '.csv')

        self.imageFolderDataset, self.transform = imageFolderDataset, transform
        self.imgTestList, self.coords = testCSV['Img'], testCSV['Coords']

    def __getitem__(self, index):

        imgTest, coordsTest = self.imgTestList[index], self.coords[index]

        imgTest = process_image(imgTest, self.transform)

        return imgTest, coordsTest

    def __len__(self):
        return len(self.imgTestList)


class VisualModel(Dataset):

    def __init__(self, imageFolderDataset, transform=transforms.ToTensor()):

        vmCSV = pd.read_csv(csvDir + '/VisualModel.csv')

        self.imgsDir, self.coords = vmCSV['Img'], vmCSV['Coords']

        self.imageFolderDataset, self.transform = imageFolderDataset, transform

    def __getitem__(self, index):

        imgVM, coordsVM = self.imgsDir[index], self.coords[index]
        imgVM = process_image(imgVM, self.transform)

        return imgVM, coordsVM

    def __len__(self):
        return len(self.imgsDir)
