### Required modules, 
This program is written in python 3.8.0 and tested in 3.9.0.
Modules required for this program to work are:
```
- functools
- Open CV
- Numpy
- Pillow
- Pickle
- random
```
## About this program
This an on-going deep learning implementation de-blurs images once trained properly. (The output will be greyscaled! Sorry! I hope to implement colour support soon)

# Training the network
If you have wish to use the provided dataset to train the network, simply execute the SGD.py file and it will train the network. If you also wish to have a visual representation of the network training, make sure that data_visual/data.text is empty, then run data_visual/plot.py and finally run SGD to get a live graph of the neural network training. It is also recommended to refer to the screenshot below to understand all the modifiable parameters for the training process.

![Training_paramenters](Screenshots/How_to_train.jpg?raw=true "Training Parameters")

#### Train Network with a different dataset
If you want to use a different dataset, you have to format the data to fit the networks input nodes. For instance, all images in both the blurred and sharpened datasets all need to be the same dimensions i.e (500 x 667). Futhermore, both datasets must have the same image format (.jpg, .png, .jpeg) must have their names in ascending order starting from 0 - the number of images in the set. If setting up the dataset is confusing or causing errors, referring to the provided data set in blur_dataset_scaled could prove useful. 


###### There is also an included formatting file that may be helpful found in: data formatting/format.py that resizes images and blurs images. For bulk renaming I recommend the Bulk renaming Utility to quickly rename all the images in ascending order.

# Running the network
Once the network has been trained, type the path of your blurred image in run.py (Ensure it is the same dimensions as the training images).
Then once your execute the file you should see a file by the name of "Unblurred.jpg" in the root directory, this is the network's output of the un-blurred image. The output will also be in greyscale since the network does not support colour as of yet

### This dataset used to train the network: [Deblurred dataset](https://www.kaggle.com/kwentar/blur-dataset)






