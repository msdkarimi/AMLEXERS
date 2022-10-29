from torchvision.datasets import VisionDataset
# import Caltech101.loadImage
import math
import copy
from PIL import Image

import os
import os.path
import sys


# def pil_loader(path):
#     # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
#     with open(path, 'rb') as f:
#         img = Image.open(f)
#         return img.convert('RGB')

class LoadImage:


    def __init__(self, root,label, imagePath):
        self.label = label
        self.imagePath = root+"/"+imagePath
        self.imageTypePIL = self.pil_loader(self.imagePath )

    def __repr__(self):
        return f'label= {self.label}'

    def pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root = root
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self.setFulladdress = set()
        self.labelPlusecounter = dict()
        self.trainData = list()
        self.hashOfLabels = dict()
        self.grandListOfAllImages = list()
        self.setOfTrainIndices = list()
        self.setOfValidationIndices = list()
        self.readTrainFile()




        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''


        # for key, value in self.labelPluseCounter.items():
        #     self.labelCounterForTrain[key] = math.ceil(value / 2)
        #     self.labelCounterForValidation[key] = self.labelPluseCounter[key] - self.labelCounterForTrain[key]

    def getTrainChunk(self):
        for key, value in self.labelPlusecounter.items():
            trainRange = math.ceil(value / 2)
            counter = 0
            for aline in self.setFulladdress:
                label = str(aline.split("/")[0])
                if label == key and counter != trainRange:
                    self.trainData.append(aline.strip())
                    counter += 1

        # return trainData


    def readTrainFile(self):

        listOfLabes = list()
        backGround = "BACKGROUND_Google"
        setOfLabels = set()

        with open("Caltech101/"+self.split + ".txt") as testFile:
            for aline in testFile:

                label = str(aline.split("/")[0])
                listOfLabes.append(label)
                if label != backGround:
                    setOfLabels.add(label)
                    self.setFulladdress.add(aline.strip())

        setOfSortedLabels = sorted(setOfLabels)
        labelCounter = 0
        for aLabel in setOfSortedLabels:
            self.hashOfLabels[aLabel] = labelCounter
            labelCounter += 1

        self.labelPlusecounter = copy.deepcopy(self.hashOfLabels)
        for key, value in self.hashOfLabels.items():
            self.labelPlusecounter[key] = listOfLabes.count(key)

        for key, _ in self.labelPlusecounter.items():
            # with open(fileName) as testFile:
            for aline in self.setFulladdress:
                label = str(aline.split("/")[0])
                if label == key:
                    imagDir = aline.strip()
                    self.grandListOfAllImages.append(LoadImage(self.root, self.hashOfLabels[label], imagDir))
        if self.split == "train":
            trainImageAddress = set(self.getTrainChunk())
            validationImageAddress = self.setFulladdress - trainImageAddress

            for i in range(len(self.grandListOfAllImages)):
                if self.grandListOfAllImages[i].imagePath in trainImageAddress:
                    self.setOfTrainIndices.append(i)
                else:
                    self.setOfValidationIndices.append(i)



    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = self.grandListOfAllImages[index].imageTypePIL, self.grandListOfAllImages[index].label

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = self.grandListOfAllImages
        return length
