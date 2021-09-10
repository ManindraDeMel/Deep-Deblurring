from network import *
training_data_directory = r"blur_dataset_scaled/blurred" # The dataset for these images were sourced from https://www.kaggle.com/kwentar/blur-dataset
training_expected_directory = r"blur_dataset_scaled/sharp"

dataset_size = 184 # number of images in the dataset
epochs = 1 # number of iterations through the dataset
img_format = "jpg" # Image format of the dataset
image_epoch = 500 # The amount of times the network trains on each image (amount of mini-batches)
img_width = 28
img_height = 28

# Instantiate and set up the network with the given parameters 
mini_batch_sizes = [28, 28]
layer_size = mini_batch_sizes[0] * mini_batch_sizes[1]
layers = [layer_size, layer_size, layer_size] # The first and last elements are input and output and stay constant, Hidden layers can be added in the middle
learning_rate = 0.07


NeuralNetwork = Network(layers, learning_rate) # Parameters we can change: Number of Hidden Layers, Learning rate. 

# Batch gradient descent
for t_epochs in range(epochs): # A number of iterations or epochs through the training data,  # Take a certain amount of images for each epoch
    random_img_index = random.randint(0, dataset_size - 1) # Randomly choose a row of pixels from a random image
    print(f"Picked image: {random_img_index}")
    random_blurred_img = readImage(f"{training_data_directory}/{random_img_index}.{img_format}")
    sharp_image = readImage(f"{training_expected_directory}/{random_img_index}.{img_format}")
NeuralNetwork.writeParameters(CONST_PARAMETERS_PATH) # After the network is done training, it's assumed that it has converged to some local minimum and so write the weights and biases to a file.


