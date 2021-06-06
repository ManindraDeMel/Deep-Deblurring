from functools import lru_cache
import numpy as np
import cv2

# Arithmetic (Here we use the cache to store repeated calculations decreasing the time used.)
multiply = lru_cache(maxsize=1000)(lambda a, b: a * b) 
sigmoid = lru_cache(maxsize=1000)(lambda x : 1 / (1 + np.exp(-x))) # Sigmoid function used to limit the range of the neurons' activations to 0 - 1
sigmoidPrime = lru_cache(maxsize=1000)(lambda activation : sigmoid(activation) * (1 - sigmoid(activation))) # Derivative of sigmoid 

def matrixMultiplication(list1, list2): # Multiplying two matrices 
    if (len(list1) != len(list2)):
        raise ValueError("Lists not same length")
    return [multiply(list1[i], list2[i]) for i in range(len(list1))] 

# Image handling 
readImage = lambda imgPath: cv2.imread(imgPath, 0)

def createImage(name, imgArr):
    imgArr = np.array(imgArr)  
    cv2.imwrite(name, imgArr) # Save image