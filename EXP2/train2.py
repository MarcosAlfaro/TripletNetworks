"""
% EXPERIMENT 2: EVALUATION OF THE INFLUENCE OF THRESHOLDS r+ AND r-
TRAIN CODE: PLACE RECOGNITION TASK

Train dataset: seq2cloudy3 sampled (588 images)
Validation dataset: seq2cloudy3 sampled (586 images)
Visual model dataset: the training set is employed as visual model (seq2cloudy3)

Epoch length: 25000 triplet samples/epochs
Num. epochs: 10

Training samples are chosen randomly, but following these restrictions:
- dist(Img Anchor, Img Positive) < rPos
- dist(Img Anchor, Img Negative) > rNeg

r+ values: 0.5m, 1m, 2m, 5m
r- values: 0.5m, 1m, 2m, 5m (r+ <= r-)
"""


import torch
import csv
import os
import numpy as np
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from sklearn.neighbors import KDTree
import losses
import create_datasets
import triplet_network


def create_path(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

csvDir = os.path.join("CSV", "TRAIN_DATA")
datasetDir = os.path.join("DATASETS", "FRIBURGO")

vmDataset = create_datasets.VisualModel(imageFolderDataset=datasetDir + "/Train/")
vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

valDataset = create_datasets.Validation(imageFolderDataset=datasetDir + "/Validation/")
valDataloader = DataLoader(valDataset, shuffle=False, num_workers=0, batch_size=1)


"""NETWORK TRAINING"""

with open(csvDir + "/Exp2TrainData.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["r+", "r-", "Epoch", "Iteration", "Recall@k1", "Geometric Error"])

    criterion = losses.TripletLoss()
    lossFunction = "triplet loss"
    sl = "TL"
    margin = 1
    netDir = create_path(os.path.join("SAVED_MODELS", "EXP2"))

    """NETWORK TRAINING"""
    rPosList = [0.5, 1, 2, 5]
    rNegList = [0.5, 1, 2, 5]

    for rPos in rPosList:
        for rNeg in rNegList:
            if rNeg < rPos:
                continue

            print("\nNEW TRAINING: ")
            print(f"Loss: {lossFunction}, margin/alpha: {margin}")
            print(f"rPos: {rPos}, rNeg: {rNeg}\n")

            net = triplet_network.TripletNetwork().to(device)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            coordsVM = []
            for i, vmData in enumerate(vmDataloader, 0):
                _, coords = vmData
                coordsVM.append(coords.detach().numpy()[0])
            treeCoordsVM = KDTree(coordsVM, leaf_size=2)

            trainDataset = create_datasets.Train(imageFolderDataset=dset.ImageFolder(datasetDir + "/Train/"),
                                                 rPos=rPos, rNeg=rNeg)
            trainDataloader = DataLoader(trainDataset, shuffle=False, num_workers=0, batch_size=16)

            rDir = create_path(os.path.join(netDir, "rPos" + str(rPos) + "rNeg" + str(rNeg)))

            bestError = 1000
            for epoch in range(10):
                print(f"Epoch {epoch}\n")

                for i, data in enumerate(trainDataloader, 0):

                    anc, pos, neg = data
                    anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

                    optimizer.zero_grad()

                    output1, output2, output3 = net(anc), net(pos), net(neg)
                    loss = criterion(output1, output2, output3, margin)
                    loss.backward()

                    optimizer.step()

                    if i % 100 == 0:
                        print(f"Epoch {epoch}, It {i}, Current loss: {loss}")

                    """VALIDATION"""

                    if i % 100 == 0:

                        recall, geomError, minErrorPossible = 0, 0, 0

                        descriptorsVM = []
                        for j, vmData in enumerate(vmDataloader, 0):
                            imgVM = vmData[0].to(device)
                            output = net(imgVM).cpu().detach().numpy()[0]
                            descriptorsVM.append(output)
                        treeDesc = KDTree(descriptorsVM, leaf_size=2)

                        for j, valData in enumerate(valDataloader, 0):

                            imgVal, coordsImgVal = valData
                            imgVal = imgVal.to(device)

                            output = net(imgVal).cpu().detach().numpy()[0]
                            coordsImgVal = coordsImgVal.detach().numpy()[0]

                            _, idxDesc = treeDesc.query(output.reshape(1, -1), k=1)
                            idxMinPred = idxDesc[0][0]

                            geomDistances, idxGeom = treeCoordsVM.query(coordsImgVal.reshape(1, -1), k=1)
                            idxMinReal = idxGeom[0][0]

                            coordsPredictedImg, coordsClosestImg = coordsVM[idxMinPred], coordsVM[idxMinReal]

                            if idxMinPred in idxGeom[0]:
                                recall += 1

                            geomError += np.linalg.norm(coordsImgVal - coordsPredictedImg)
                            minErrorPossible += np.linalg.norm(coordsImgVal - coordsClosestImg)

                        recall *= 100 / len(valDataloader)
                        geomError /= len(valDataloader)
                        minErrorPossible /= len(valDataloader)

                        print(f"Average recall (k=1)= {recall} %")
                        print(f"Average geometric error: {geomError} m, Current error: {bestError} m")
                        print(f"Minimum reachable error: {minErrorPossible} m")
                        print(f"Relative error: {geomError - minErrorPossible} m\n")

                        if geomError < bestError:
                            bestError = geomError

                            if geomError <= 0.20:

                                netName = os.path.join(rDir, "net_ep" + str(epoch) + "it" + str(i))
                                torch.save(net, netName)

                                print("SAVED MODEL")
                                print(f"Epoch: {epoch}, It: {i}")
                                print(f"Validation recall: {recall} %, Geometric error: {geomError} m\n")

                                writer.writerow([rPos, rNeg, epoch, i + 1, recall, geomError])

                    if recall >= 100:
                        print("Training finished")
                        break

                netName = os.path.join(rDir, "net_ep" + str(epoch) + "_end")
                torch.save(net, netName)
