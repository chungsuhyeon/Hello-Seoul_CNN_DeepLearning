import cv2
import numpy as np
from PIL import Image
import os, glob, random
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import RMSprop

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import HashingVectorizer

import pickle


print("WOW")
