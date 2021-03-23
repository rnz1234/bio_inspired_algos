from config import *
#import math_lib
from math_lib import *
import matplotlib.pyplot as plt




# NeuralNet class - just an NN "circuit"
class NeuralNet(object):
    """
    Constructor
    :param layer_widths - list of number of neurons in each layer (len(layer_widths) == number of layers in network)
    :param activation_functions - a list of string of activation method name - per layer (from 1 to last), valid values: "relu", "sigmoid", "softmax"
    :param loss_function - the loss function is use, valid values: "squared_error", "cross_entropy"
    """
    def __init__(self, layer_widths=[4,5,3], activation_functions=["relu","softmax"], loss_function="squared_error"):
        self.layer_widths = layer_widths
        self.activation_functions = activation_functions
        self.loss_function = loss_function
        self.weights = []
        self.bias = []
        # init network with random numbers
        DEBUG_PRINT("#######################################", this_print_verb=VERBOSITY.PRIO1)
        DEBUG_PRINT("##       INIT                          ", this_print_verb=VERBOSITY.PRIO1)
        DEBUG_PRINT("#######################################", this_print_verb=VERBOSITY.PRIO1)
        for i in range(len(layer_widths)-1):
            # current layer weights (matrix)
            if USED_INIT_METHOD_W == INIT_METHOD.GAUSSIAN:
                print("GAUSSIAN init for w")
                cur_layer_weights_matrix = np.random.randn(layer_widths[i + 1], layer_widths[i])
            if USED_INIT_METHOD_W == INIT_METHOD.UNIFORM:
                print("UNIFORM init for w")
                cur_layer_weights_matrix = np.random.rand(layer_widths[i+1], layer_widths[i])
            DEBUG_PRINT("layer " + str(i) + " weights: ")
            DEBUG_PRINT("shape : " + str(cur_layer_weights_matrix.shape))
            #print(cur_layer_weights_matrix)
            self.weights.append(cur_layer_weights_matrix)
            # current layer bias (vector)
            if USED_INIT_METHOD_B == INIT_METHOD.GAUSSIAN:
                print("GAUSSIAN init for b")
                cur_layer_bias_vec = np.random.randn(layer_widths[i+1], 1)
            if USED_INIT_METHOD_B == INIT_METHOD.UNIFORM:
                print("UNIFORM init for b")
                cur_layer_bias_vec = np.random.rand(layer_widths[i + 1], 1)
            if USED_INIT_METHOD_B == INIT_METHOD.ZEROES:
                print("ZEROES init for b")
                cur_layer_bias_vec = np.zeros((layer_widths[i + 1], 1))
            DEBUG_PRINT("layer " + str(i) + " bias: ")
            DEBUG_PRINT("shape : " + str(cur_layer_bias_vec.shape))
            self.bias.append(cur_layer_bias_vec)

    """
    Calculate feed-forward
    :param input - input vector / array of vector to calculate feed forward    
    :param is_train - is this training or not (dropout applied only during training)
    :return: a list with 2 members:
            member[0] = all neurons' pre non-linearity values
            member[1] = all neurons' post non-linearity values
            member[2] = neurons_dropout_vectors - dropout vectors for all layers
    """
    def feed_forward(self, input, is_train):
        DEBUG_PRINT("")
        DEBUG_PRINT("#######################################")
        DEBUG_PRINT("##       FEED FORWARD                  ")
        DEBUG_PRINT("#######################################")
        DEBUG_PRINT("input for feed forward:")
        DEBUG_PRINT(input)
        cur_layer_neurons_post_nonlnr = np.copy(input).T
        neurons_post_nonlnr = [cur_layer_neurons_post_nonlnr]
        neurons_pre_nonlnr = []

        # only meaningful when using dropout
        neurons_dropout_vectors = []

        # make the forward pass
        for i in range(len(self.layer_widths)-1):
            DEBUG_PRINT("current_layer : " + str(i))
            # calculate layer pre non linearity and add to list
            DEBUG_PRINT("% weights for layer:")
            DEBUG_PRINT("shape : " + str(self.weights[i].shape))
            DEBUG_PRINT(self.weights[i])
            DEBUG_PRINT("% input for layer:")
            DEBUG_PRINT("shape : " + str(cur_layer_neurons_post_nonlnr.shape))
            DEBUG_PRINT(cur_layer_neurons_post_nonlnr)
            cur_layer_neurons_pre_nonlnr = self.weights[i].dot(cur_layer_neurons_post_nonlnr) + self.bias[i]

            if USE_DROPOUT:
                if is_train:
                    if i != len(self.layer_widths)-2: # in order to avoid last layer
                        dv = np.random.binomial(1, DROPOUT_PROB, size=cur_layer_neurons_pre_nonlnr.shape) / DROPOUT_PROB
                        neurons_dropout_vectors.append(dv)
                        cur_layer_neurons_pre_nonlnr = cur_layer_neurons_pre_nonlnr*dv

            DEBUG_PRINT("% pre non linear output for layer:", this_print_verb=VERBOSITY.PRIO3)
            DEBUG_PRINT("shape : " + str(cur_layer_neurons_pre_nonlnr.shape))
            DEBUG_PRINT(cur_layer_neurons_pre_nonlnr, this_print_verb=VERBOSITY.PRIO3)
            neurons_pre_nonlnr.append(cur_layer_neurons_pre_nonlnr)
            # calculate layer post non linearity and add to list
            #if i == len(self.layer_widths)-1-1:
            #    cur_layer_neurons_post_nonlnr = math_lib.sigmoid(neurons_pre_nonlnr[-1])        #sigmoid
            #else:
            #print(self.activation_functions[i])
            cur_layer_neurons_post_nonlnr = eval(self.activation_functions[i])(neurons_pre_nonlnr[-1])#math_lib.relu(neurons_pre_nonlnr[-1])
            neurons_post_nonlnr.append(cur_layer_neurons_post_nonlnr)
            DEBUG_PRINT("% post non linear output for layer:", this_print_verb=VERBOSITY.PRIO3)
            DEBUG_PRINT("shape : " + str(cur_layer_neurons_post_nonlnr.shape))
            DEBUG_PRINT(str_to_print=cur_layer_neurons_post_nonlnr, this_print_verb=VERBOSITY.PRIO3)
            DEBUG_PRINT("")

        #exit()
        return [neurons_pre_nonlnr, neurons_post_nonlnr, neurons_dropout_vectors]




    """
    Back-propagate a result of feed_forward run - create the update needed for weights and biases
    :param neurons_pre_nonlnr - all neurons' pre non-linearity values (from feed_forward)
    :param neurons_post_nonlnr - all neurons' post non-linearity values (from feed_forward)
    :param neurons_dropout_vectors - dropout vectors for all layers
    :param exp_output - expected output  
    :return a list with 2 members:
        member[0] = all biases' final delta (to be multiplied by learning rate)
        member[1] = all weights' final delta (to be multiplied by learning rate)
    """
    def back_propagate(self, neurons_pre_nonlnr, neurons_post_nonlnr, neurons_dropout_vectors, exp_output, epoch=0, iteration=0):
        DEBUG_PRINT("")
        DEBUG_PRINT("#######################################")
        DEBUG_PRINT("##       BACK PROPAGATE                 ")
        DEBUG_PRINT("#######################################")
        DEBUG_PRINT("expected output:")
        DEBUG_PRINT("shape : " + str(exp_output.shape))
        DEBUG_PRINT(exp_output)
        b_f_delta = []
        w_f_delta = []
        # size of batch
        size_of_batch = exp_output.shape[0]
        #DEBUG_PRINT(size_of_batch)
        # this will store all delta values
        all_deltas = [0]*len(self.weights)
        # calculate deltas for last NN layer
        #DEBUG_PRINT(neurons_post_nonlnr[-1].shape)
        #DEBUG_PRINT(exp_output.T.shape)
        #DEBUG_PRINT(math_lib.relu(neurons_pre_nonlnr[-1]).shape)
        #print(neurons_pre_nonlnr[-1])
        #print(math_lib.relu_derivative(neurons_pre_nonlnr[-1]))
        #if epoch == 6 and iteration == 20:
        if epoch == 7 and iteration == 19:
            # print("weights")
            # print(self.weights)
            # print("lin outputs")
            # print(neurons_pre_nonlnr)
            # print("non lin outputs")
            # print(neurons_post_nonlnr)
            # print(self.loss_function+"_derivative")
            # print(eval(self.loss_function+"_derivative")(output= neurons_post_nonlnr[-1], expected=exp_output.T))
            # print(eval(self.activation_functions[-1]+"_derivative")(neurons_pre_nonlnr[-1]))
            # exit()
            pass
        #print(self.loss_function)
        #print(self.activation_functions[-1])
        #if USE_DROPOUT:
        #    all_deltas[-1] = eval(self.loss_function + "_derivative")(output=neurons_post_nonlnr[-1], expected=exp_output.T) * eval(self.activation_functions[-1] + "_derivative")(neurons_pre_nonlnr[-1]) * neurons_dropout_vectors[-1]
        #else:
        all_deltas[-1] = eval(self.loss_function+"_derivative")(output= neurons_post_nonlnr[-1], expected=exp_output.T) * eval(self.activation_functions[-1]+"_derivative")(neurons_pre_nonlnr[-1]) #(exp_output.T-neurons_post_nonlnr[-1]) * math_lib.sigmoid_derivative(neurons_pre_nonlnr[-1])       #sigmoid_derivative
        #print("last delta")
        #print(all_deltas[-1])
        # backpropagate error to former layers
        for i in reversed(range(len(all_deltas)-1)):
            #print(self.activation_functions[i])
            if USE_DROPOUT:
                all_deltas[i] = self.weights[i + 1].T.dot(all_deltas[i + 1]) * eval(self.activation_functions[i] + "_derivative")(neurons_pre_nonlnr[i]) * neurons_dropout_vectors[i] # math_lib.relu_derivative(neurons_pre_nonlnr[i])
            else:
                all_deltas[i] = self.weights[i+1].T.dot(all_deltas[i+1]) * eval(self.activation_functions[i]+"_derivative")(neurons_pre_nonlnr[i]) #math_lib.relu_derivative(neurons_pre_nonlnr[i])

        #exit()
        # average the deltas to be taken later to update the wieghts & biases - over all batch                                                                                                                   !")
        #print(all_deltas[0].shape)
        b_f_delta = [(delta.dot(np.ones((size_of_batch,1)))) / float(size_of_batch) for delta in all_deltas]
        #print(neurons_post_nonlnr[i].T.shape)
        w_f_delta = [(delta.dot(neurons_post_nonlnr[i].T)) / float(size_of_batch) for i,delta in enumerate(all_deltas)]

        #print(b_f_delta[0].shape)
        #print(w_f_delta[0].shape)

        return [b_f_delta, w_f_delta]

    """
    Set/get methods.
    """
    def get_weights(self):
        return self.weights

    def set_weights(self, weights):
        self.weights = weights

    def get_biases(self):
        return self.bias

    def set_biases(self, biases):
        self.bias = biases


# NN wrapper - used for making a NN, training, analyzing & testing it and using it to predict
class NeuralNetWrapper(object):
    """
    Constructor
    :param training_set_features - training set input data
    :param training_set_labels - training set labels
    :param validation_set_features - validation set input data
    :param validation_set_labels - validation set labels
    :param testing_set_features - testing set input data
    :param model_testing_set_features,model_testing_set_labels - used for testing set with known labels in order to test the training after the training
    :param layer_widths - same as for NeuralNet object
    :param activation_functions - same as for NeuralNet object
    :param loss_function - same as for NeuralNet object
    :param max_epoch - number of epochs training will run
    :param learning_rate - learning rate
    :param weight_decay - weight decay factor (used only if enabled in config.py)
    :param size_of_batch - size of batch (update to weights happened after each batch)
    """
    def __init__(self,  training_set_features,
                        training_set_labels,
                        validation_set_features,
                        validation_set_labels,
                        testing_set_features,
                        model_testing_set_features=None,
                        model_testing_set_labels=None,
                        layer_widths=[4, 5, 3],
                        activation_functions=["relu","softmax"],
                        loss_function="squared_error",
                        max_epoch=1,
                        learning_rate=0.1,
                        weight_decay=0.01,
                        size_of_batch=10):
        self.training_set_features = training_set_features
        self.validation_set_features = validation_set_features
        self.testing_set_features = testing_set_features
        self.model_testing_set_features = model_testing_set_features
        self.training_set_labels_int = training_set_labels
        if DEBUG_READ_DUMMY_DATA:
            self.training_set_labels = one_hot_1_to_10(x=training_set_labels, num_of_values=3)
            self.validation_set_labels = validation_set_labels #math_lib.one_hot_1_to_10(x=validation_set_labels, num_of_values=3)
            #self.testing_set_labels = math_lib.one_hot_1_to_10(x=testing_set_labels, num_of_values=3)
            if model_testing_set_labels is not None:
                self.model_testing_set_labels = one_hot_1_to_10(x=model_testing_set_labels, num_of_values=2)#3)
        else:
            #print(training_set_labels)
            self.training_set_labels = one_hot_1_to_10(x=training_set_labels, num_of_values=10)
            #print(self.training_set_labels)
            self.validation_set_labels = validation_set_labels # math_lib.one_hot_1_to_10(x=validation_set_labels, num_of_values=10)
            #self.testing_set_labels = math_lib.one_hot_1_to_10(x=testing_set_labels, num_of_values=10)
            if model_testing_set_labels is not None:
                self.model_testing_set_labels = one_hot_1_to_10(x=model_testing_set_labels, num_of_values=10)
        # init NN
        self.nn = NeuralNet(layer_widths=layer_widths, activation_functions=activation_functions, loss_function=loss_function)
        self.loss_function = loss_function
        # training parameters
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.max_epoch = max_epoch
        self.weight_decay = weight_decay
        self.size_of_batch = size_of_batch
        self.training_acc = []
        self.validation_acc = []
        self.training_loss = []
        self.validation_loss = []
        self.epoch = 0
        self.last_epoch_train_pred = []
        self.last_epoch_train_labels = []
        self.last_epoch_valid_pred = []
        self.last_epoch_valid_labels = []
        self.max_validation_acc = 0
        self.max_validation_acc_epoch = 0

    """
    Pre process network inputs
    """
    def pre_process(self):
        #print("pre process")
        #print(self.training_set_features)
        print("asasasasasasasa1")
        if USED_DATA_AUGMENTATION_IN_PRE_PROCESS_METHOD:
            print("asasasasasasasa2")
            self.training_set_features = data_random_elements_zeroize(data=self.training_set_features, prob_of_zero=DATA_PROPABLISTIC_ZEROIZING_PROB_TO_ZERO)

        if USE_DATA_STD_PRE_PROCESS:
            self.training_set_features = data_standardization(self.training_set_features)
            self.validation_set_features = data_standardization(self.validation_set_features)
            self.testing_set_features = data_standardization(self.testing_set_features)

        #print(self.training_set_features)

    """
    Run training
    """
    def train(self):
        # train for number of epochs as needed
        DEBUG_PRINT("##############################################################################################", this_print_verb=VERBOSITY.PRIO1)
        DEBUG_PRINT("##               T   R   A   I   N   I   N   G                          ")
        DEBUG_PRINT("##############################################################################################", this_print_verb=VERBOSITY.PRIO1)
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            DEBUG_PRINT("$$$$$$$$$$$$$$$$$$$$$$   EPOCH " + str(epoch), this_print_verb=VERBOSITY.PRIO1)
            iteration = 0
            while iteration < len(self.training_set_features):
                DEBUG_PRINT("--------------   ITERATION " + str(iteration), this_print_verb=VERBOSITY.PRIO3)
                # set actual batch size - last one may be smaller than wanted size_of_batch (because length of training set doesn't divide by it)
                if iteration+self.size_of_batch < len(self.training_set_features):
                    cur_batch_size = self.size_of_batch
                else:
                    cur_batch_size = len(self.training_set_features)
                # prepare batch
                if USED_DATA_AUGMENTATION_POST_PRE_PROCESS_METHOD == DATA_AUGMENTATION_METHOD.NO_AUG:
                    batch_features = self.training_set_features[iteration:iteration+cur_batch_size]
                if USED_DATA_AUGMENTATION_POST_PRE_PROCESS_METHOD == DATA_AUGMENTATION_METHOD.PROPABLISTIC_ZEROIZING:
                    batch_features = data_random_elements_zeroize(data=self.training_set_features[iteration:iteration + cur_batch_size], prob_of_zero=DATA_PROPABLISTIC_ZEROIZING_PROB_TO_ZERO)
                batch_labels = self.training_set_labels[iteration:iteration+cur_batch_size]
                # pass batch forward through network
                neurons_pre_nonlnr, neurons_post_nonlnr, neurons_dropout_vectors = self.nn.feed_forward(input=batch_features, is_train=True)
                #print(neurons_post_nonlnr[-1].T)
                #exit()
                # backpropagate
                DEBUG_PRINT("shape of training_set_labels:")
                DEBUG_PRINT(self.training_set_labels.shape)
                b_f_delta, w_f_delta = self.nn.back_propagate(neurons_pre_nonlnr=neurons_pre_nonlnr, neurons_post_nonlnr=neurons_post_nonlnr, neurons_dropout_vectors=neurons_dropout_vectors, exp_output=batch_labels, epoch=epoch, iteration=iteration)
                # do weights & biases updates
                pre_iter_weights = self.nn.get_weights()
                pre_iter_biases = self.nn.get_biases()
                #print("pre_iter_weights")
                #print(pre_iter_weights)
                #print("")
                #print("")
                if USED_WEIGHT_UPDATE_RULE == WEIGHT_UPDATE_RULE.BASIC:
                    post_iter_weights = [weight + self.learning_rate*delta for weight,delta in zip(pre_iter_weights, w_f_delta)]
                if USED_WEIGHT_UPDATE_RULE == WEIGHT_UPDATE_RULE.WEIGHT_DECAY_L2:
                    post_iter_weights = [weight + self.learning_rate * (delta - self.weight_decay*weight) for weight, delta in zip(pre_iter_weights, w_f_delta)]
                if USED_WEIGHT_UPDATE_RULE == WEIGHT_UPDATE_RULE.WEIGHT_DECAY_L1:
                    post_iter_weights = [weight + self.learning_rate * (delta - self.weight_decay*np.sign(weight)) for weight, delta in zip(pre_iter_weights, w_f_delta)]
                post_iter_biases = [bias + self.learning_rate * delta for bias, delta in zip(pre_iter_biases, b_f_delta)]

                self.nn.set_weights(post_iter_weights)
                self.nn.set_biases(post_iter_biases)

               
                
                # move to next batch
                iteration += self.size_of_batch
                #print("")
                #print(neurons_pre_nonlnr[-1].T)
                #print(neurons_post_nonlnr[-1].T)
                if epoch % 20 == 0 and iteration == 1:
                    pass
                    # print("pre iter weights:")
                    # print(pre_iter_weights)
                    # print("deltas")
                    # print([self.learning_rate*delta for weight,delta in zip(pre_iter_weights, w_f_delta)])
                    # print("linear outputs:")
                    # print(neurons_pre_nonlnr)
                    # print("non linear outputs:")
                    # print(neurons_post_nonlnr)
                    # print("weights post update:")
                    # print(post_iter_weights)
                    # print("biases post update:")
                    # print(post_iter_biases)


                # if iteration > 2:
                #     exit()
                #if [0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0] in batch_labels.T:
                #    print(batch_labels.T)
                #    exit()
                #DEBUG_PRINT("loss = " + str(eval(self.loss_function)(output=neurons_post_nonlnr[-1], expected=batch_labels.T)), this_print_verb=VERBOSITY.PRIO3)  #VERBOSITY.PRIO1

            #print(post_iter_weights)
            #DEBUG_PRINT("loss = " + str(eval(self.loss_function)(output=neurons_post_nonlnr[-1], expected=batch_labels.T)),this_print_verb=VERBOSITY.PRIO1)  # VERBOSITY.PRIO1
            if PRINT_WEIGHTS_PERIODIC:
                if epoch % 5 == 0:# and iteration == 0:
                    for x in range(len(self.nn.get_weights())):
                        DEBUG_PRINT("-------------- WEIGHT PRINT ----------------", this_print_verb=VERBOSITY.PRIO1)
                        DEBUG_PRINT("weights: " + str(self.nn.get_weights()[x]), this_print_verb=VERBOSITY.PRIO1)
                        DEBUG_PRINT("biases: " + str(self.nn.get_biases()[x]), this_print_verb=VERBOSITY.PRIO1)
                        DEBUG_PRINT("--------------------------------------------", this_print_verb=VERBOSITY.PRIO1)


            self.training_progress(training_features=self.training_set_features, exp_training_labels_int=self.training_set_labels_int, include_train_acc=True)

    """
    Run network on input data
    :param data - input data
    :return output classification for the data
    """
    def predict(self, data):
        # pass batch forward through network
        neurons_pre_nonlnr, neurons_post_nonlnr, neurons_dropout_vectors = self.nn.feed_forward(data, is_train=False)
        return neurons_post_nonlnr[-1]

    """
    Calculate network accuracy and loss 
    :param features - features (input data) to run for
    :param exp_labels - expected labels
    :param is_of_train - True if for training, False if for validation
    :return [accuracy, loss] - where accuracy is the fraction of data labeled correctly by network, loss is normalized to number of samples
    """
    def check_accuracy_and_loss(self, features, exp_labels, is_of_train):
        #print("check accuracy")
        out_labels = self.predict(features)
        #print(out_labels.T)
        out_labels_int = one_hot_to_int(out_labels.T)
        loss = eval(self.loss_function)(output=out_labels, expected=exp_labels)/len(out_labels_int)
        #print(out_labels_int)
        #print(exp_labels)
        error = out_labels_int - exp_labels

        # this is in order to later output it to file to check accuracy manually - to be sure the accuracy calculation is correct
        if self.epoch == self.max_epoch-1:
            if is_of_train:
                self.last_epoch_train_pred = out_labels_int
                self.last_epoch_train_labels = exp_labels
            else:
                self.last_epoch_valid_pred = out_labels_int
                self.last_epoch_valid_labels = exp_labels

        #print(error)
        return [float(len(error[error==0]))/len(error), loss]

    """
    Track training progress - should be called every epoch
    :param training_features - the training set input data (features)
    :param exp_training_labels_int - expected training set labels
    :param include_train_acc - not in uses
    """
    def training_progress(self, training_features, exp_training_labels_int, include_train_acc=False):
        # accuracy over training batch
        train_acc_and_loss = self.check_accuracy_and_loss(features=training_features, exp_labels=exp_training_labels_int, is_of_train=True)
        # accuracy over validation set
        if DEBUG_TRACK_VALIDATION:
            valid_acc_and_loss = self.check_accuracy_and_loss(features=self.validation_set_features, exp_labels=self.validation_set_labels, is_of_train=False)
        # append
        self.training_acc.append(train_acc_and_loss[0])
        self.training_loss.append(train_acc_and_loss[1])
        if DEBUG_TRACK_VALIDATION:
            self.validation_acc.append(valid_acc_and_loss[0])
            self.validation_loss.append(valid_acc_and_loss[1])
        print("training acc : " + str(train_acc_and_loss[0]))
        print("training loss : " + str(train_acc_and_loss[1]))
        if DEBUG_TRACK_VALIDATION:
            print("validation acc : " + str(valid_acc_and_loss[0]))
            print("validation loss : " + str(valid_acc_and_loss[1]))
            if valid_acc_and_loss[0] > self.max_validation_acc:
                self.max_validation_acc = valid_acc_and_loss[0]
                self.max_validation_acc_epoch = self.epoch
        print("max validation acc : " + str(self.max_validation_acc))
        print("max validation acc epoch : " + str(self.max_validation_acc_epoch))

    """
    Creates a graph of training & validation accuracy and loss
    training accuracy - red
    training loss - black
    validation accuracy - blue
    validation loss - green
    """
    def graph_training_progress(self):
        plt.plot(np.arange(self.max_epoch), np.array(self.training_acc), 'r')
        plt.plot(np.arange(self.max_epoch), np.array(self.training_loss), 'k')
        if DEBUG_TRACK_VALIDATION:
            plt.plot(np.arange(self.max_epoch), np.array(self.validation_acc), 'b')
            plt.plot(np.arange(self.max_epoch), np.array(self.validation_loss), 'g')
        plt.show()

    def model_test(self):
        pass

    """
    Run network on test set and creates prediction
    :return the network prediction for the test set
    """
    def test(self):
        pred = self.predict(data=self.testing_set_features)
        print(pred.T)
        return one_hot_to_int(pred.T)

    """
    Auxiliary debug methods
    """
    def get_last_epoch_train_pred(self):
        return self.last_epoch_train_pred

    def get_last_epoch_train_labels(self):
        return self.last_epoch_train_labels

    def get_last_epoch_valid_pred(self):
        return self.last_epoch_valid_pred

    def get_last_epoch_valid_labels(self):
        return self.last_epoch_valid_labels

    def get_max_validation_acc_and_epoch(self):
        return [self.max_validation_acc, self.max_validation_acc_epoch]
    
