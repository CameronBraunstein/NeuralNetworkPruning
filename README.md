# NeuralNetworkPruning
Project by Abhishek Ramdas Nair, Vishal Shaw and Cameron Braunstein


Here is an overview of the code:

Network.py: The class representing a full neural network

Layer.py: The class representing a single layer of the Network

DataLoader.py: A helper class to load the NMIST data from the /samples folder

FileLoader.py: A class to handle storing and retrieving our neural network

Specs of our Network: a 748-300-100-10 feed forward neural network. Input data is scaled so all entries lie between 0 and 1. Sigmoids are used for activation functions. The network is trained with l2 regularization. 
