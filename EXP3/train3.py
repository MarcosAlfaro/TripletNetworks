"""
% EXPERIMENT 3: EVALUATION OF THE INFLUENCE OF THE BATCH SIZE
TRAIN CODE: PLACE RECOGNITION TASK

Train dataset: seq2cloudy3 sampled (588 images)
Validation dataset: seq2cloudy3 sampled (586 images)
Visual model dataset: the training set is employed as visual model (seq2cloudy3)

Epoch length: 25000 triplet samples/epochs
Num. epochs: 10

Training samples are chosen randomly, but following these restrictions:
- dist(Img Anchor, Img Positive) < 0.5m
- dist(Img Anchor, Img Negative) > 0.5m

Batch size values: 1, 2, 4, 8, 16
"""

import torch
import os
import csv
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

coordsVM = []
for i, vmData in enumerate(vmDataloader, 0):
    _, coords = vmData
    coordsVM.append(coords.detach().numpy()[0])
treeCoordsVM = KDTree(coordsVM, leaf_size=2)


"""NETWORK TRAINING"""

with open(csvDir + "/Exp3TrainData.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["N", "Loss", "Epoch", "Iteration", "Recall@k1", "Geometric Error"])

    lossList = ["triplet loss", "lazy triplet"]
    slList = ["TL", "LT"]
    marginList = [1, 1.25]
    netDir = create_path(os.path.join("SAVED_MODELS", "EXP3"))

    """NETWORK TRAINING"""
    N_list = [16, 8, 4, 2, 1]
    for lossFunction in lossList:
        idxLoss = lossList.index(lossFunction)
        sl = slList[idxLoss]
        margin = marginList[idxLoss]
        criterion = losses.get_loss(lossFunction)
        lossDir = create_path(os.path.join(netDir, sl))

        for N in N_list:
            print("\nNEW TRAINING: ")
            print(f"Loss function: {lossFunction}, Batch size = {N}")

            net = triplet_network.TripletNetwork().to(device)
            optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

            trainDataset = create_datasets.Train(imageFolderDataset=dset.ImageFolder(datasetDir + "/Train/"))
            trainDataloader = DataLoader(trainDataset, shuffle=False, num_workers=0, batch_size=N)

            Ndir = create_path(os.path.join(lossDir, str(N)))

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

                    if i % (1600/N) == 0:
                        print(f"Epoch {epoch}, It {i}, Current loss: {loss}")

                    """VALIDATION"""

                    if i % (1600/N) == 0:

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

                            if geomError - minErrorPossible <= 0.20:

                                netName = os.path.join(Ndir, "net_ep" + str(epoch) + "it" + str(i))
                                torch.save(net, netName)

                                print("SAVED MODEL")
                                print(f"Epoch: {epoch}, It: {i}")
                                print(f"Validation recall: {recall} %, Geometric error: {geomError} m\n")

                                writer.writerow([N, lossFunction, epoch, i + 1, recall, geomError])

                    if recall >= 100:
                        print("Training finished")
                        break

                netName = os.path.join(Ndir, "net_ep" + str(epoch) + "_end")
                torch.save(net, netName)
