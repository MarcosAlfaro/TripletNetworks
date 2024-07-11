# TripletNetworks
In this work, we have studied the main factors that influence the learning process of a triplet network.
Three different experiments have been carried out:
  - Experiment 1. Study of the triplet loss function
  - Experiment 2. Study of the selection process of the training samples
  - Experiment 3. Study of the batch size

To conduct the experiments, we have employed COLD-Freiburg database, which can be downloaded from here: https://www.cas.kth.se/COLD/.

VGG16 model was adapted and retrained to perform visual localization.
Omnidirectional images were converted into a panoramic format.

Image sets:
Freiburg, Part A, Path 2
- Training/Visual Model: seq2_cloudy3 sampled (588 images)
- Validation: seq2_cloudy3 sampled (586 images)
- Test Cloudy: seq2_cloudy2 (2595 images)
- Test Night: seq2_night2 (2707 images)
- Test Sunny: seq2_sunny2 (2114 images)

This repository contains:
  - The scripts to train and test the model for each experiment (trainX.py, testX.py)
  - The scripts to generate the training samples and load the images into the CPU for each experiment (create_csv.py, create_datasets.py)
  - The script where the model is defined (triplet_network.py)
  - The script where the triplet losses are defined (losses.py)
  - The script to generate additional figures (create_figures.py) 

