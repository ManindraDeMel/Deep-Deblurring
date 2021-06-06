import random
import pickle
from miscellaneous import *
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
        self.layers.insert(0, [Neuron(activation = inputs[node_index]) for node_index in range(len(inputs))]) # Convert the inputs into neuron objects and insert it as the first layer in the network
        for layer in range(1, len(self.layers)): # now starting with the first hidden layer we can use the previous layer's activations, attached weights and biases to keep going forward in the network.
            for node in self.layers[layer]:
                node.calculate_activation(self.layers[layer - 1])

    def backPropagate(self, expected):
        for i in reversed(range(len(self.layers))): # reversing the layers to start from the output layer since we're spreading the error out from the last layer backwards
            layer = self.layers[i]
            deltas = []
            if i == len(self.layers) - 1: # If this is the first iteration (output layer) then we find the cost of the network between the expected output and the network's output
                for j in range(len(layer)):
                    deltas.append(2 * ((expected[j] / 255) - layer[j].activation))

            else: # otherwise for all the other layers find the respective error to the neuron's delta. The delta of which originated from the output layer
                for j in range(len(layer)):
                    delta = 0.0
                    for neuron in self.layers[i + 1]:
                        delta += (neuron.weights[j] * neuron.delta) # Sum each weight's effect on the error/delta on the proceeding layer's neurons
                    deltas.append(delta)
            
            for j in range(len(layer)):
                neuron = layer[j]
                neuron.delta = deltas[j] * sigmoidPrime(neuron.activation) # calculate the delta for that neuron (z holder)

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
    def learn(self, training_data, expected_data, n_epoch, t_epochs):
        for epoch in range(n_epoch): # A local epoch for each individual image
            for training_row in range(len(training_data)): # Training row is each pixel row of an image i.e 1920 pixels of a 1920 x 1080 image. Whilst len(training_data) is 1080
                # Here we're just checking if the inputs we have fed to the network correspond with the network layers and sizes so it doesn't raise any errors later on
                if (len(training_data[training_row]) != self.layersizes[0]):
                    raise ValueError("input data and input length mismatch")
                if (len(expected_data[training_row]) != self.layersizes[-1]):
                    raise ValueError("expected data and output layer length mismatch")
                if (len(expected_data) != len(training_data)):
                    raise ValueError("Training data mismatch")

                training_list = list(map(lambda x: x / 1, training_data[training_row])) # Here we are getting each pixel row of the image and then dividing it by 255 to make it between 0 - 1 (the same range as the network's output due to sigmoid)
                self.forwardPropagate(training_list)
                error = sum([((expected_data[training_row][i] / 255) - (self.layers[-1][i].activation))**2 for i in range(len(expected_data[training_row]))]) # This is the total error between the network's output and the answer 
                self.backPropagate(expected_data[training_row])
                self.updateNetwork()
                print(f"Pixel Row: {training_row}, Error: {error}, Epoch: {epoch + 1}/{n_epoch}, Image: {t_epochs[0] + 1}/{t_epochs[1]}")
 ######################################################### Un-comment the encapsulated code if you want to visualize the data            
                with open("data_visual/data.txt", "a+") as write_file: 
                    write_file.write(f"{training_row}, {error}\n")
 #########################################################
    def run(self, image): # This method is used once the network has been trained so it can hopefully de-blur other images which it hasn't encountered before.
        last_layers = []
        for pixel_row in image: # Since we have split the images into individual rows and trained the network on those rows. We need to append them all back together to form an image of original dimensions
            self.forwardPropagate(pixel_row)
            last_layers.append(list(map(lambda x: x * 255, [neuron.activation for neuron in self.layers[-1]])))
        createImage("Unblurred.jpg", last_layers)

