"""
THIS PROGRAM CREATES ALL THE CSV NEEDED TO LAUNCH EVERY TRAINING, VALIDATION AND TEST PROGRAMMES
Therefore, it must be executed before any other training or test program
"""


import os
import csv
import random
import numpy as np
from sklearn.neighbors import KDTree

import torchvision.datasets as dset

csvDir = "CSV"
datasetDir = os.path.join("DATASETS", "FRIBURGO")

condIlum = ['Cloudy', 'Night', 'Sunny']

trainDataset = dset.ImageFolder(root=datasetDir + "/Train")
rooms = trainDataset.classes


def get_coords(imageDir):
    idxX, idxY, idxA = imageDir.index('_x'), imageDir.index('_y'), imageDir.index('_a')
    x, y = float(imageDir[idxX + 2:idxY]), float(imageDir[idxY + 2:idxA])
    return np.array([x, y])


def train(epochLength, tree, rPos, rNeg):

    with open(csvDir + '/TrainrPos' + str(rPos) + 'rNeg' + str(rNeg) + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ImgAnc", "ImgPos", "ImgNeg"])

        for i in range(epochLength):
            idxAnc = random.randrange(len(imgsList))
            imgAnc = imgsList[idxAnc]
            coordsAnc = coordsList[idxAnc]

            indices = tree.query_radius(coordsAnc.reshape(1, -1), r=rPos)[0]
            idxPos = random.choice(indices)
            while idxAnc == idxPos:
                idxPos = random.choice(indices)
            imgPos = imgsList[idxPos]

            indices = tree.query_radius(coordsAnc.reshape(1, -1), r=rNeg)[0]
            idxNeg = random.randrange(len(imgsList))
            while idxNeg in indices or idxAnc == idxNeg:
                idxNeg = random.randrange(len(imgsList))
            imgNeg = imgsList[idxNeg]

            writer.writerow([imgAnc, imgPos, imgNeg])
    return


def validation():
    valDir = os.path.join(datasetDir, "Validation")

    with open(csvDir + '/Validation.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Room", "Coords"])

        for room in rooms:
            roomDir = os.path.join(valDir, room)
            imgsVal = os.listdir(roomDir)
            for image in imgsVal:
                imgValDir = os.path.join(roomDir, image)
                coordsVal = get_coords(imgValDir)
                writer.writerow([imgValDir, coordsVal])
    return


def test(il):
    valDir = os.path.join(datasetDir, "Test" + il)
    with open(csvDir + '/Test' + il + '.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Idx Room", "Coords"])
        for room in rooms:
            roomDir = os.path.join(valDir, room)
            imgsTest = os.listdir(roomDir)
            for image in imgsTest:
                imgTestDir = os.path.join(roomDir, image)
                coordsTest = get_coords(imgTestDir)
                writer.writerow([imgTestDir, coordsTest])
    return


def visual_model():
    vmDir = datasetDir + "/Train/"

    with open(csvDir + '/VisualModel.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Img", "Coords"])

        imgsVM, coordsVM = [], []
        for room in rooms:
            roomDir = os.path.join(vmDir, room)
            imgsDir = os.listdir(roomDir)
            for image in imgsDir:
                imgDir = os.path.join(roomDir, image)
                coords = get_coords(imgDir)
                imgsVM.append(imgDir)
                coordsVM.append(coords)
                writer.writerow([imgDir, coords])

    return imgsVM, coordsVM


imgsList, coordsList = visual_model()

treeVM = KDTree(coordsList, leaf_size=2)


rPosList = [0.5, 1, 2, 5]
rNegList = [0.5, 1, 2, 5]
for rPos in rPosList:
    for rNeg in rNegList:
        if rNeg < rPos:
            continue
        train(epochLength=25000, tree=treeVM, rPos=rPos, rNeg=rNeg)

validation()
for ilum in condIlum:
    test(ilum)
