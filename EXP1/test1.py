"""
% EXPERIMENT 1: EVALUATION OF DIFFERENT TRIPLET LOSSES
TEST CODE: PLACE RECOGNITION TASK

Test dataset:
Cloudy: seq2cloudy2 (2595 images)
Night: seq2night2 (2707 images)
Sunny: seq2sunny2 (2114 images)

Visual model dataset: the training set is employed as visual model (seq2cloudy3)

The test is performed in one step:
    -each test image is compared with the images of the visual model of the entire map
    -the nearest neighbour indicates the retrieved coordinates

Loss functions: Triplet Margin Loss, Lazy Triplet Loss, Circle Loss, Angular Loss
"""

import torch
import os
import csv
import numpy as np
from torch.utils.data import DataLoader
from sklearn.neighbors import KDTree
import create_figures
import create_datasets
from triplet_network import TripletNetwork


def create_path(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


csvDir = create_path(os.path.join("CSV", "RESULTS"))
figuresDir = create_path(os.path.join("FIGURES", "EXP1"))
datasetDir = os.path.join("DATASETS", "FRIBURGO")

kMax = 20

condIlum = ['Cloudy', 'Night', 'Sunny']

vmDataset = create_datasets.VisualModel(imageFolderDataset=datasetDir + '/Train/')
vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)


with open(csvDir + "/Exp1Results.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    rowCSV = ["Loss", "Margin", "Net", "Ilum", "Geom error", "Min error"]
    for k in range(kMax):
        rowCSV.append("Recall@k" + str(k+1))
    writer.writerow(rowCSV)

    netDir = os.path.join("SAVED_MODELS", "EXP1")
    losses = os.listdir(netDir)
    bestError, bestLoss = 100, ""
    for loss in losses:
        lossDir = os.path.join(netDir, loss)
        margins = os.listdir(lossDir)
        bestLossError, bestMargin = 100, ""
        for margin in margins:
            marginDir = os.path.join(lossDir, margin)
            testNets = os.listdir(marginDir)
            bestMarginError, bestMarginNet = 100, ""
            for testNet in testNets:
                testDir = os.path.join(marginDir, testNet)
                net = torch.load(testDir).to(device)
                print(f"TEST NETWORK {testDir}")

                """VISUAL MODEL"""

                descriptorsVM, coordsVM = [], []

                for i, vmData in enumerate(vmDataloader, 0):
                    imgVM, coords = vmData
                    imgVM = imgVM.to(device)

                    output = net(imgVM).cpu().detach().numpy()[0]
                    descriptorsVM.append(output)
                    coordsVM.append(coords.detach().numpy()[0])
                treeCoordsVM = KDTree(coordsVM, leaf_size=2)
                treeDesc = KDTree(descriptorsVM, leaf_size=2)

                """
                
                
                
                
                
                """

                recallLG = np.zeros((len(condIlum), kMax))
                geomError = np.zeros(len(condIlum))
                minErrorPossible = np.zeros(len(condIlum))

                for ilum in condIlum:
                    idxIlum = condIlum.index(ilum)

                    print(f"Test {ilum}\n")

                    testDataset = create_datasets.Test(
                        illumination=ilum, imageFolderDataset=datasetDir + "/Test" + ilum + "/")
                    testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                    coordsMapTest = []

                    for i, data in enumerate(testDataloader, 0):
                        imgTest, coordsImgTest = data
                        imgTest = imgTest.to(device)

                        output = net(imgTest).cpu().detach().numpy()[0]
                        coordsImgTest = coordsImgTest.detach().numpy()[0]

                        if loss == 'CL' or loss == 'AL':
                            cosSimilarities = np.dot(descriptorsVM, output)
                            idxMinPred = np.argmax(cosSimilarities)
                        else:
                            _, idxDesc = treeDesc.query(output.reshape(1, -1), k=1)
                            idxMinPred = idxDesc[0][0]

                        geomDistances, idxGeom = treeCoordsVM.query(coordsImgTest.reshape(1, -1), k=kMax)
                        idxMinReal = idxGeom[0][0]

                        coordsPredictedImg, coordsClosestImg = coordsVM[idxMinPred], coordsVM[idxMinReal]

                        if idxMinPred in idxGeom[0]:
                            label = str(idxGeom[0].tolist().index(idxMinPred)+1)
                            recallLG[idxIlum][idxGeom[0].tolist().index(idxMinPred):] += 1
                        else:
                            label = "F"

                        coordsMapTest.append([coordsPredictedImg[0], coordsPredictedImg[1],
                                              coordsImgTest[0], coordsImgTest[1], label])

                        geomError[idxIlum] += np.linalg.norm(coordsImgTest - coordsPredictedImg)
                        minErrorPossible[idxIlum] += np.linalg.norm(coordsImgTest - coordsClosestImg)

                    recallLG[idxIlum] *= 100 / len(testDataloader)
                    geomError[idxIlum] /= len(testDataloader)
                    minErrorPossible[idxIlum] /= len(testDataloader)

                    create_figures.display_coord_map(figuresDir, coordsVM, coordsMapTest, kMax, ilum, loss)

                    print(f"Geometric error: {geomError[idxIlum]} m")
                    print(f"Minimum reachable error: {minErrorPossible[idxIlum]} m\n")

                    rowCSV = [loss, margin, testNet, ilum, geomError[idxIlum], minErrorPossible[idxIlum]]
                    for k in range(kMax):
                        rowCSV.append(recallLG[idxIlum][k])
                    writer.writerow(rowCSV)

                avgGeomError = np.average(geomError)
                avgRecallLG = np.average(recallLG, axis=0)
                avgMinError = np.average(minErrorPossible)

                rowCSV = [loss, margin, testNet, "Average", avgGeomError, avgMinError]
                for k in range(kMax):
                    rowCSV.append(avgRecallLG[k])
                writer.writerow(rowCSV)

                if avgGeomError < bestMarginError:
                    bestMarginNet, bestMarginError = testNet, avgGeomError
                    if avgGeomError < bestLossError:
                        bestMargin, bestLossError = margin, avgGeomError
                        if avgGeomError < bestError:
                            bestLoss, bestError = loss, avgGeomError

            if bestMarginNet != "":
                print(f"Best net loss {loss}, margin {margin}: {bestMarginNet}, Geometric Error: {bestMarginError} m")
        if bestMargin != "":
            print(f"Best margin loss {loss}: {bestMargin}, Geometric Error: {bestLossError} m")
    if bestLoss != "":
        print(f"Best loss: {bestLoss}, Geometric error: {bestError} m")
