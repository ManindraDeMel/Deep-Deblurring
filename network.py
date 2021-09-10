import random
import pickle
from miscellaneous import *
import time
CONST_PARAMETERS_PATH = "weightsBiases.pickle" # The file the weights and biases are stored in once trained

# This is the neuron class. A neuron is the object which holds and activation and is attached to other neurons in different layers through weights. It also holds a bias
# Since neurons held several variables I decided to just make it a class so its easier to work with and makes the code more legible.
class Neuron: 
    def __init__(self, weights = None, bias = None, activation = 0): # I gave some parameters default values for the input nodes which don't have weights or biases
        self.weights = weights
        self.bias = bias
        self.delta = 0
        self.activation = activation
        

    def calculate_activation(self, previousLayer): # This calculates a give neuron's activation or value using this formula: sigmoid (0 - 1) of [(sum of all previous nodes* weights) + bias]
        previousNodes = [node.activation for node in previousLayer]
        self.activation = sigmoid(sum(matrixMultiplication(self.weights, previousNodes)) + self.bias)

# This is the main class which houses all the neurons in several layers and all the methods necessary to train and de-blur images. 
class Network:
    def __init__(self, layersizes = [], descentRate = 0.01, trained = False): 
        self.layersizes = layersizes
        self.layers = []
        self.avgError = 10
        self.DESC_RATE = descentRate # The Descent rate determines the size the network steps up and down. If its too big it won't find a local minimum or optimal point to de-blur images 
        # and if it's too small then it will take an unreasonable amount of time to train and might converge at a local minimum prematurely. 
        
        # Initializing the weights and biases
        if (trained):
            self.readParameters(CONST_PARAMETERS_PATH)
        else:
            self.generateWeightsAndBiases()
    # If we've already trained the network and simply just want to de-blur images we can just read the weights written to a file after a training session
            
    # The method below is actually crucial to the network. By generating these random weights and biases it determines what local minimum the network
    # will eventually converge to. Some training sessions might start extremely lucky with a low cost, whilst more realistically the network starts off
    # with an extremely high cost and converge to some local minimum which will most likely be completely underwhelming and unsatisfactory. 
    def generateWeightsAndBiases(self):
        for layer_index in range(1, len(self.layersizes)):
            layer = []
            for _ in range(self.layersizes[layer_index]):
                nodeWeights = [random.randint(-1000, 1000) / 100 for _ in range(self.layersizes[layer_index - 1])] # Generating random weights (from small as 1 / 100 to 10) 
                bias = random.randint(-1000, 1000) / 100 # Generating biases (from small as 1 / 100 to 10)
                layer.append(Neuron(nodeWeights, bias))
            self.layers.append(layer)

    def readParameters(self, path): # This just reads the weights and biases from a file (I thought of writing it in plain text but it was really big in reality)
        try: # Checking if the file that stores the weights and biases actually exists.
            pickle_file = open(path, "rb")
        except:
            raise FileExistsError("The network has not been trained and thus the weights and biases file has not been created")
        self.layers = pickle.load(pickle_file)
        for layer in self.layers: # Read the hidden layers from the network and set all the activations to 0 so it returns only the weights and biases
            for neuron in layer: 
                neuron.activation = 0

    def writeParameters(self, path): # Writes weights and biases to a file 
        pickle_file = open(path, "wb")
        pickle.dump(self.layers[1:], pickle_file)
        pickle_file.close()

    def forwardPropagate(self, inputs): 
        if (self.layers[0][0].weights == None): # checking if there is still inputs from last iteration hanging around
            self.layers.pop(0)
        self.layers.insert(0, [Neuron(activation = input) for input in inputs]) # Convert the inputs into neuron objects and insert it as the first layer in the network
        for layer in range(1, len(self.layers)): # now starting with the first hidden layer we can use the previous layer's activations, attached weights and biases to keep going forward in the network.
            for node in self.layers[layer]:
                node.calculate_activation(self.layers[layer - 1])

    # In backpropgation we're going to calculate the cost (the difference between the network output and the expected outputs). We then take this cost
    # for each neuron in the output layer and then traverse back into the hidden layers. In the hidden layers we find the derivative with 
    # respect to the derivative of the activation (The sum of the weights and previous inputs plus some bias passed through sigmoid)
    # Then we finally calculate what each weight's contribution to the cost of the network.

    def backPropagate(self, expected):
        for i in reversed(range(len(self.layers))): # reversing the layers to start from the output layer since we're spreading the error out from the last layer backwards
            layer = self.layers[i]
            deltas = []
            if i == len(self.layers) - 1: # If this is the first iteration (output layer) then we find the cost of the network between the expected output and the network's output
                for j in range(len(layer)):
                    deltas.append(((expected[j]) - (layer[j].activation * 255))) # This is the cost calculation 

            else: # otherwise for all the other layers find the respective error to the neuron's delta. The delta of which originated from the output layer
                for j in range(len(layer)):
                    delta = 0.0
                    for neuron in self.layers[i + 1]:
                        delta += (neuron.weights[j] * neuron.delta) # Sum each weight's effect on the cost on the proceeding layer's neurons
                    deltas.append(delta)
            
            for j in range(len(layer)):
                neuron = layer[j]
                neuron.delta = deltas[j] * sigmoidPrime(neuron.activation) # calculate the cost for each neuron's activation 

    def updateNetwork(self): 
        # Update Weights and Biases of the network based on each neuron's delta
        for i in range(1, len(self.layers)):
            previous_layer = [neuron.activation for neuron in self.layers[i - 1]]
            for neuron in range(len(self.layers[i])):
                for j in range(len(previous_layer)):
                    self.layers[i][neuron].weights[j] += self.DESC_RATE * self.layers[i][neuron].delta * previous_layer[j]
                self.layers[i][neuron].bias += self.DESC_RATE * self.layers[i][neuron].delta
    # The learn method receives an image and then splits that image into rows of pixels, the length of this row is the width of the image. 
    # It then runs the network on each row of pixels until its reached the height of the image
    def learn(self, training_data, expected_data, batch_size, img_dimensions,t_epochs, n_epoch):#, t_epochs):
        avg_error = []
        for epoch in range(n_epoch): # A local epoch for each individual image
            #pick 2 randoms numbers
            xy = [random.randint(batch_size[0], img_dimensions[0] - batch_size[0]), random.randint(batch_size[1], img_dimensions[1] - batch_size[1])]
            print(f"Coords: {xy}")
            mini_batch = training_data[xy[1]:xy[1]+batch_size[1]]
            mini_batch = list(map(lambda row: row[xy[0]:xy[0]+batch_size[0]], mini_batch))
            mini_batch = [element for row in mini_batch for element in row]
            mini_batch_expected = expected_data[xy[1]:xy[1]+batch_size[1]]
            mini_batch_expected = list(map(lambda row: row[xy[0]:xy[0]+batch_size[0]], mini_batch_expected))
            mini_batch_expected = [element for row in mini_batch_expected for element in row]
            self.forwardPropagate(mini_batch)
            self.run(readImage("4_blur.png"), [28, 28])
            error = sum([0.5 *((mini_batch_expected[i] / 255) - (self.layers[-1][i].activation))**2 for i in range(len(mini_batch_expected))]) # This is the total error between the network's output and the answer 
            avg_error.append(error)
            self.backPropagate(mini_batch_expected)
            self.updateNetwork()
            print(f"Error: {error}, Epoch: {epoch + 1}/{n_epoch}, Image: {t_epochs[0] + 1}/{t_epochs[1]}")
 ######################################################### Un-comment the encapsulated code if you want to visualize the data            
            if (epoch + 1) + (n_epoch * t_epochs[0] < 10000):
                with open("data_visual/data.txt", "a+") as write_file: 
                    write_file.write(f"{(epoch + 1) + (n_epoch * t_epochs[0])}, {error}\n")
 #########################################################
        print(f"average error: {sum(avg_error) / len(avg_error)}")
        self.avgError = sum(avg_error) / len(avg_error)

    def run(self, image, batch_size): # This method is used once the network has been trained so it can hopefully de-blur other images which it hasn't encountered before.
        last_layers = [] # The below code was originally written for mini-batch. So if required, different mini-batch sizes can be tested. However, its still used for batch gradient descent as we can technically call the whole image a mini-batch
        batches = []
        width = len(image[0])
        height = len(image)
        counter = 1
        if (width % batch_size[0] != 0) or (height % batch_size[1] != 0):
            raise ValueError("Can't deblur images evenly")
            
        for batch_y in range(0, height, batch_size[1]):
            row_batch = []
            for batch_x in range(0, width, batch_size[0]):
                batch = list(map(lambda row: row[batch_x: batch_x + batch_size[0]], image[batch_y: batch_y + batch_size[1]])) # Get batches of a certain size starting from the top left
                batch = [element for row in batch for element in row] # flatten array
                self.forwardPropagate(batch) # split it into a bunch of even squares
                output = (list(map(lambda x: x * 255, [neuron.activation for neuron in self.layers[-1]])))
                x = 0
                output_batch = []
                for _ in range(batch_size[1]):
                    output_batch.append(output[x : x + batch_size[0]])
                    x += batch_size[0]
                row_batch.append(output_batch)
                print(f"batch: {counter} added")
                counter += 1
            batches.append(row_batch)

        for batch_row in batches: # Re organise batches by rows
            for row_index in range(batch_size[1]):
                row = []
                for batch_index in range(int(width / batch_size[0])):
                    row.append(batch_row[batch_index][row_index])
                last_layers.append(row)  

        for row_index in range(len(last_layers)): # Flatten list
            last_layers[row_index] = [element for row in last_layers[row_index] for element in row]
        createImage("Unblurred.jpg", last_layers) 

