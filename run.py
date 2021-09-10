from network import *

#####
blurred_img_path = r"4_blur.png" # Path to the image that will be de-blurred
#####

image = readImage(blurred_img_path)
batch_sizes = [28, 28]
network = Network(trained = True) # Read the weights and biases stored in a file and run the network. Assuming the network has been trained
network.run(image, batch_sizes)

