from network import *
training_data_directory = r"blur_dataset_scaled/blurred" # The dataset for these images were sourced from https://www.kaggle.com/kwentar/blur-dataset
training_expected_directory = r"blur_dataset_scaled/sharp"

dataset_size = 184 # number of images in the dataset
epochs = 10 # number of iterations through the dataset
img_format = "jpg" # Image format of the dataset
image_epoch = 3 # The amount of times the network trains on each image
# Instantiate and set up the network with the given parameters 
tmp_img = readImage(f"{training_data_directory}/0.{img_format}")
width = tmp_img.shape[1] # Retrieve the width of the first image since all the images should be the same dimensions
del tmp_img # since we got the width of each image we can delete this variable


layers = [width, int(width/2), width] # The first and last elements are input and output and stay constant, Hidden layers can be added in the middle
learning_rate = 3

NeuralNetwork = Network(layers, learning_rate) # Parameters we can change: Number of Hidden Layers, Learning rate. 

# Train the network
for t_epochs in range(epochs): # A number of iterations or epochs through the training data,  # Take a certain amount of images for each epoch
    random_img_index = random.randint(0, dataset_size - 1) # Randomly choose a row of pixels from a random image
    print(f"Picked image: {random_img_index}")
    random_blurred_img = readImage(f"{training_data_directory}/{random_img_index}.{img_format}")
    sharp_image = readImage(f"{training_expected_directory}/{random_img_index}.{img_format}")
    NeuralNetwork.learn(random_blurred_img, sharp_image, image_epoch, [t_epochs, epochs]) # 3 epochs on each image (training on each row)
NeuralNetwork.writeParameters(CONST_PARAMETERS_PATH) # After the network is done training, it's assumed that it has converged to some local minimum and so write the weights and biases to a file.















# So below we're passing several different inputs to the network. We first dataSet_inputs which is an array of images. We then train the 
# network on each image by splitting the images in rows equivalent to its length (due to how they're read by the readImage() function). For example,
# if we have an image with the specified dimensions of 1920 x 1080 the network would be given 1920 inputs and iterate over different 1920 inputs 1080 times.
# This allows more training data and less need to generate an extremely large amount of weights and biases 
# (1920 x 1080 = 2 073 600 x 512 hidden layer nodes = 1 061 683 200 or roughly 1 billion weights for just the input -> first hidden layer weights)
# Instead we only have (1920 x 512 = 983 040 weights for the inputs -> first hidden layer weights which is a significant decrease in storage)









# dataSet_inputs = [
#     [[134, 142], [243, 124]], #this is an image
#     [[143, 65], [156, 234]], # this is another image
# ]

# dataSet_outputs = [
#     [[243, 34], [12, 244]],
#     [[89, 60], [93, 34]],
# ]

# NeuralNetwork = Network([2, 3, 3, 2], 0.3)