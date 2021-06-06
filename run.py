from network import *

#####
blurred_img_path = r"blur.jpg" # Path to the image that will be de-blurred
#####

image = readImage(blurred_img_path)

network = Network(trained = True) # Read the weights and biases stored in a file and run the network. Assuming the network has been trained
network.run(image)

