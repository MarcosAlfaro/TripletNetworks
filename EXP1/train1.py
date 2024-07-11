"""
% EXPERIMENT 1: EVALUATION OF DIFFERENT TRIPLET LOSSES
TRAIN CODE: PLACE RECOGNITION TASK

Train dataset: seq2cloudy3 sampled (588 images)
Validation dataset: seq2cloudy3 sampled (586 images)
Visual model dataset: the training set is employed as visual model (seq2cloudy3)

Epoch length: 25000 triplet samples/epochs
Num. epochs: 10

Training samples are chosen randomly, but following these restrictions:
- dist(Img Anchor, Img Positive) < 0.5m
- dist(Img Anchor, Img Negative) > 0.5m

Loss functions: Triplet Margin Loss, Lazy Triplet Loss, Circle Loss, Angular Loss
"""

import torch
import os
import csv
import numpy as np
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from sklearn.neighbors import KDTree
import losses
import create_datasets1
import triplet_network


def create_path(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

csvDir = create_path(os.path.join("CSV", "TRAIN_DATA"))
datasetDir = os.path.join("DATASETS", "FRIBURGO")


vmDataset = create_datasets1.VisualModel(imageFolderDataset=datasetDir + "/Train/")
vmDataloader = DataLoader(vmDataset, shuffle=False, num_workers=0, batch_size=1)

coordsVM = []
for i, vmData in enumerate(vmDataloader, 0):
    _, coords = vmData
    coordsVM.append(coords.detach().numpy()[0])
treeCoordsVM = KDTree(coordsVM, leaf_size=2)

trainDataset = create_datasets1.Train(imageFolderDataset=dset.ImageFolder(datasetDir + "/Train/"))
trainDataloader = DataLoader(trainDataset, shuffle=False, num_workers=0, batch_size=16)

valDataset = create_datasets1.Validation(imageFolderDataset=datasetDir + "/Validation/")
valDataloader = DataLoader(valDataset, shuffle=False, num_workers=0, batch_size=1)


"""NETWORK TRAINING"""

with open(csvDir + "/Exp1TrainData.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Loss", "Margin", "Epoch", "Iteration", "Recall@k1", "Geometric Error"])

    selectedLosses = ["triplet loss", "lazy triplet", "angular loss", "circle loss"]
    lossAbreviations = ["TL", "LT", "AL", "CL"]
    marginsGlobalLoc = [[1], [1.25], [30], [1]]

    for lossFunction in selectedLosses:

        criterion = losses.get_loss(lossFunction)
        if criterion == -1:
            continue

        idxLoss = selectedLosses.index(lossFunction)
        sl = lossAbreviations[idxLoss]
        margins = marginsGlobalLoc[idxLoss]
        lossDir = create_path(os.path.join("SAVED_MODELS", "EXP1", sl))

        for margin in margins:
            marginDir = create_path(os.path.join(lossDir, str(margin)))

            net = triplet_network.TripletNetwork().to(device)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            """NETWORK TRAINING"""

            print("\nNEW TRAINING: ")
            print(f"Loss: {lossFunction}, margin/alpha: {margin}\n")

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

                            if lossFunction == 'circle loss' or lossFunction == 'angular loss':
                                cosSimilarities = np.dot(descriptorsVM, output)
                                idxMinPred = np.argmax(cosSimilarities)
                            else:
                                _, idxDesc = treeDesc.query(output.reshape(1, -1), k=1)
                                idxMinPred = idxDesc[0][0]

                            _, idxGeom = treeCoordsVM.query(coordsImgVal.reshape(1, -1), k=1)
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
                        print(f"Relative error: {geomError - minErrorPossible}\n")

                        if geomError <= bestError:
                            bestError = geomError

                            if geomError <= 0.20:

                                netDir = os.path.join(marginDir, "net_" + sl + "_ep" + str(epoch) + "it" + str(i))
                                torch.save(net, netDir)

                                print("SAVED MODEL")
                                print(f"Epoch: {epoch}, It: {i}")
                                print(f"Validation recall: {recall}%, Geometric error: {geomError} m\n")

                                writer.writerow([lossFunction, margin, epoch, i + 1, recall, geomError])

                    if recall >= 100:
                        print("Training finished")
                        break
                netDir = os.path.join(marginDir, "net_" + sl + "_ep" + str(epoch) + "_end")
                torch.save(net, netDir)
