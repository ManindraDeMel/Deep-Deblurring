from PIL import Image, ImageFilter
import random

# This library only words with the assumption that the dataset has been formatted as 0.jpg, 1.jpg ... or 0.png, 1.png ... accordingly 

def blurImage(path, blur_num): 
    if (random.randint(0, 2) > 1): # Using different blurs randomly, if you only have a dataset with sharpened images.
        OriImage = Image.open(path)
        boxImage = OriImage.filter(ImageFilter.BoxBlur(blur_num)) 
        boxImage.save(path)
    else: 
        OriImage = Image.open(path)
        gaussImage  = OriImage.filter(ImageFilter.GaussianBlur(blur_num))
        gaussImage.save(path)

def blurImages(path, dataset_size, imgFormat = "jpg"):
    for img in range(dataset_size):
        blur = random.randint(5, 20) # change this range to get different blurs
        blurImage(f"{path}/{img}.{imgFormat}", blur)

def resizeImage(x, y, path):
    image = Image.open(path)
    image = image.resize((x,y),Image.ANTIALIAS)
    image.save(path)

def resizeImages(x, y, dataset_size, path, imgFormat = "jpg"):
    for img in range(dataset_size): # This resizes the image of a given dataset. Make sure that each image is named in ascending order i.e (0.jpg, 1.jpg, 2.jpg, 3.jpg ...)
        resizeImage(x, y, f"{path}/{img}.{imgFormat}")