from torchvision.datasets import VisionDataset
import loadImage
import math
import copy
from PIL import Image

import os
import os.path
import sys


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        self.backGround = "BACKGROUND_Google"
        self.root = root
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self.hashOfLabels = dict()
        self.labelPluseCounter = dict()
        self.labelCounterForTrain = dict()
        self.labelCounterForValidation = dict()
        self.grandListOfAllImages = list()


        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''

    def getTrainChunk(self):
        listOfLabes = list()
        setOfLabels = set()

        with open(self.split+".txt") as testFile:
            for aline in testFile:
                label = str(aline.split("/")[0])
                listOfLabes.append(label)
                if label != self.backGround:
                    setOfLabels.add(label)
        setOfSortedLabels = sorted(setOfLabels)
        labelCounter = 0

        for aLabel in setOfSortedLabels:
            self.hashOfLabels[aLabel] = labelCounter
            labelCounter += 1

        self.labelPluseCounter = copy.deepcopy(self.hashOfLabels)

        for key, value in self.hashOfLabels.items():
            self.labelPluseCounter[key] = listOfLabes.count(key)

        for key, value in self.labelPluseCounter.items():
            self.labelCounterForTrain[key] = math.ceil(value / 2)
            self.labelCounterForValidation[key] = self.labelPluseCounter[key] - self.labelCounterForTrain[key]

        for key, value in self.labelCounterForTrain.items():
            with open(self.split+".txt") as testFile:
                for aline in testFile:
                    label = str(aline.split("/")[0])
                    if label != key:
                        continue
                    else:
                        for image in range(value):
                            trimmedPath = aline.strip()
                            newObjOfpathAndLabel = loadImage.LoadImage( self.root, self.hashOfLabels[key], trimmedPath)
                            self.grandListOfAllImages.append(newObjOfpathAndLabel)

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image, label = ... # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = ... # Provide a way to get the length (number of elements) of the dataset
        return length
