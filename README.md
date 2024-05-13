# BirdClassifier
!!!This project is still under construction!!!
The goal of this project is to gain familiarity with the conformer architecture (based on the paper https://arxiv.org/pdf/2005.08100) and obtain hands-on experience with audio classification tasks. The project is based on the BirdCLEF 2023 dataset provided by kaggle.
The original dataset contains audio files of 264 different bird species. The dataset we use is obtained by preprocessing the original audio files and contains black-and-white spectrograms obtained from 5-second snippets of bird sounds. The goal is to classify the bird species based on the spectrograms.

## Motivation for classifying bird sounds
The following quote from the competition description gives a good overview of the theoretical motivation for this task.

"Birds are excellent indicators of biodiversity change since they are highly mobile and have diverse habitat requirements. Changes in species assemblage and the number of birds can thus indicate the success or failure of a restoration project. However, frequently conducting traditional observer-based bird biodiversity surveys over large areas is expensive and logistically challenging. In comparison, passive acoustic monitoring (PAM) combined with new analytical tools based on machine learning allows conservationists to sample much greater spatial scales with higher temporal resolution and explore the relationship between restoration interventions and biodiversity in depth."

## The dataset
For the input data, black-and-white spectrograms obtained from 5-second snippets of the audio data are used (see https://www.kaggle.com/datasets/forbo7/spectrograms-birdclef-2023 for details on the preprocessing).
The data contains audio from 264 different bird species. However, the dataset is highly imbalanced with some species having only a few samples and others having several hundred samples. This issue is not addressed in the current version but should be investigated later.

## The model
We use the conformer architecture as described in https://arxiv.org/abs/2005.08100. 
The model is organized in the following blocks:
- A sub-sampling block consisting of two convolutional layers with a stride of 2 reducing the input size by a factor of 4.
- A fully connected layer
- A conformer block consisting of a multi-head self-attention layer, a feed-forward layer and a convolutional layer.
- A dense layer with output size equal to the number of classes.
For regularization, dropout layers and normalization layers are used.
A avoid vanishing gradients, skip connections are used in the conformer block.
We use Adam as the optimizer and categorical crossentropy as the loss function.

## Results
In its current state, the model is strongly overfitting on the training data and only archives about 0.5 accuracy on the test data.
Attempts to reduce overfitting by regularization such as dropout layers, normalization layers, reducing model size and weight decay have not been successful so far.
Further attempts to reduce overfitting will need to be made as the current result is worse than other models. Possible approaches include:
- Data augmentation 
- Further reducing model complexity 
- Try to reduce imbalance in the dataset
