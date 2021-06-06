from network import *

#####
blurred_img_path = r"blur_dataset_scaled/blurred/134.jpg"
#####

image = readImage(blurred_img_path)

network = Network(trained = True) # Read the weights and biases stored in a file and run the network. Assuming the network has been trained
network.run(image)

