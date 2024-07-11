"""
% EXPERIMENT 3: EVALUATION OF THE INFLUENCE OF THE BATCH SIZE
TEST CODE: PLACE RECOGNITION TASK

Test dataset:
Cloudy: seq2cloudy2 (2595 images)
Night: seq2night2 (2707 images)
Sunny: seq2sunny2 (2114 images)

Visual model dataset: the training set is employed as visual model (seq2cloudy3)

The test is performed in one step:
    -each test image is compared with the images of the visual model of the entire map
    -the nearest neighbour indicates the retrieved coordinates

Batch size values: 1, 2, 4, 8, 16
"""


import os
from torch.utils.data import DataLoader
import numpy as np
import torch
import csv
from sklearn.neighbors import KDTree
import create_figures
import create_datasets
from triplet_network import TripletNetwork


def create_path(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


csvDir = create_path(os.path.join("CSV", "RESULTS"))
figuresDir = create_path(os.path.join("FIGURES", "EXP3"))
datasetDir = os.path.join("DATASETS", "FRIBURGO")

kMax = 20

condIlum = ['Cloudy', 'Night', 'Sunny']

vmDataset = create_datasets.VisualModel(imageFolderDataset=datasetDir + '/Train/')
vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)


with open(csvDir + "/Exp3Results.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    rowCSV = ["N", "Loss", "Ilum", "Geom error", "Min error"]
    for k in range(kMax):
        rowCSV.append("Recall@k" + str(k+1))
    writer.writerow(rowCSV)

    netDir = os.path.join("SAVED_MODELS", "EXP3")
    losses = os.listdir(netDir)

    Nlist = [1, 2, 4, 8, 16]

    for loss in losses:
        lossDir = os.path.join(netDir, loss)
        bestLossError, bestN = 100, ""

        print(f"Loss: {loss}")

        for N in Nlist:
            print(f"N: {N}")
            nDir = os.path.join(lossDir, str(N))
            testNets = os.listdir(nDir)
            bestNError, bestNNet = 100, ""

            for testNet in testNets:
                testDir = os.path.join(nDir, testNet)
                net = torch.load(testDir).to(device)
                print(f"TEST NETWORK {testNet}")

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

                    testDataset = create_datasets.Test(illumination=ilum,
                                                       imageFolderDataset=datasetDir + "/Test" + ilum + "/")
                    testDataloader = DataLoader(testDataset, num_workers=0, batch_size=1, shuffle=False)

                    coordsMapTest = []

                    for i, data in enumerate(testDataloader, 0):
                        imgTest, coordsImgTest = data
                        imgTest = imgTest.to(device)

                        output = net(imgTest).cpu().detach().numpy()[0]
                        coordsImgTest = coordsImgTest.detach().numpy()[0]

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

                    create_figures.display_coord_map(figuresDir, 'Global', coordsVM, coordsMapTest, kMax, ilum, N)

                    print(f"Geometric error: {geomError[idxIlum]} m")
                    print(f"Minimum reachable error: {minErrorPossible[idxIlum]} m\n")

                    rowCSV = [N, loss, ilum, geomError[idxIlum], minErrorPossible[idxIlum]]
                    for k in range(kMax):
                        rowCSV.append(recallLG[idxIlum][k])
                    writer.writerow(rowCSV)

                avgGeomError = np.average(geomError)
                avgRecallLG = np.average(recallLG, axis=0)
                avgMinError = np.average(minErrorPossible)

                rowCSV = [N, loss, "Average", avgGeomError, avgMinError]
                for k in range(kMax):
                    rowCSV.append(avgRecallLG[k])
                writer.writerow(rowCSV)

                if avgGeomError < bestNError:
                    bestNNet, bestNError = testNet, avgGeomError
                    if avgGeomError < bestLossError:
                        bestN, bestLossError = N, avgGeomError

            if bestNNet != "":
                print(f"Best net N={N}: {bestNNet}, Geometric Error: {bestNError} m")
        if bestN != "":
            print(f"Best N loss {loss}: {bestN}, Geometric Error: {bestLossError} m")
