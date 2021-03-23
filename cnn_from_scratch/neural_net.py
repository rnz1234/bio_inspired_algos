import numpy as np
from collections import deque
from config import *
from math_lib import *
from layers import *
import matplotlib.pyplot as plt

from PIL import Image

class NeuralNet(object):
    def __init__(self, layers, learning_rate, loss_func, weight_decay):
        self.layers = layers
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.nn_weights = None
        self.weight_decay = weight_decay

    def feed_forward(self, inputs):
        #print(inputs.shape)
        cur_net_out = inputs
        # go over all layers
        for layer in self.layers:
            #print(cur_net_out.shape)
            cur_net_out = layer.feed_forward(cur_net_out)
        return cur_net_out

    def train_iteration(self, mini_batch, cur_batch_size, epoch):
        #print(mini_batch[0].shape)
        # forward pass
        mini_batch_inputs, mini_batch_outputs = mini_batch
        neurons_pre_nonlnr = deque()
        neurons_pre_nonlnr.appendleft([mini_batch_inputs, 0])
        #print(mini_batch_inputs[0].shape)
        cur_net_out = mini_batch_inputs

        # if epoch == 10:
        #     data = np.zeros((32,32,3), dtype=np.uint8)
        #     plt.imshow(cur_net_out[0][:,:,0], cmap='gray')
        #     plt.show()
        #exit()
        #print(data.shape)
        #img = Image.fromarray(data, 'RGB')
        #img.save('my.png')
        #img.show()

        # go over all layers
        for layer in self.layers:
            cur_layer_neurons_pre_nonlnr, cur_net_out, dv = layer.feed_forward_train(cur_net_out)

            # if epoch == 10:
            #     plt.imshow(cur_net_out[0][:,:,0], cmap='gray')
            #     plt.show()
            #     exit()

            # print(cur_net_out)
            # exit()
            neurons_pre_nonlnr.appendleft([cur_layer_neurons_pre_nonlnr, dv])

        # calculate loss derivative over prediction and expected output
        loss_err = self.loss_func.derivative(X=cur_net_out, Y=mini_batch_outputs)

        # backpropagation to calculate gradients
        cur_layer_neurons_pre_nonlnr_back_pass = neurons_pre_nonlnr.popleft()[0]
        backwarded_err = loss_err
        grads = deque()
        #print(len(self.layers))
        self.nn_weights = []
        for layer in reversed(self.layers):
            #print("layer")
            # if not isinstance(layer, FcLayer):
            #     # --- ORIGINAL ---
            #print("backwarded_err shape " + str(backwarded_err.shape))

            layer_err = layer.error(cur_layer_neurons_pre_nonlnr_back_pass, backwarded_err, dv) #local
            #print("layer_err shape " + str(layer_err.shape))
            #print("data_pre_non_lnr pre fetch shape " + str(cur_layer_neurons_pre_nonlnr_back_pass.shape))
            layer_entry = neurons_pre_nonlnr.popleft()
            #print(layer_entry)
            cur_layer_neurons_pre_nonlnr_back_pass = layer_entry[0]
            dv = layer_entry[1]
            #print("data_pre_non_lnr post fetch shape " + str(cur_layer_neurons_pre_nonlnr_back_pass.shape))

            gd = layer.gradient(cur_layer_neurons_pre_nonlnr_back_pass, layer_err, cur_batch_size)
            #print("-------------")
            #print(gd)
            #exit()
            #print("-------------")
            grads.appendleft(gd)
            # if not isinstance(layer, FlatteningLayer):
            #     print("gradient shape " + str(layer.gradient(cur_layer_neurons_pre_nonlnr_back_pass, layer_err, cur_batch_size)[0].shape))
            backwarded_err = layer.back_propagate(layer_err) # backwarded error

            if isinstance(layer, FcLayer):
                self.nn_weights.append(["fc layer", layer.get_weights(), layer.get_bias()])
            if isinstance(layer, ConvLayer):
                self.nn_weights.append(["conv layer", layer.get_weights()])
        #exit()

        # update weights
        for layer in self.layers:
            layer.update_weights(lr=self.learning_rate, deltas=grads.popleft(), weight_decay=self.weight_decay)

        assert len(grads) == 0

    def get_weights(self):
        return self.nn_weights

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
    :param layers - list of layers objects
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
                        layers=[],
                        loss_function=CrossEntropyLoss,
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
            self.training_set_labels = formatter.one_hot_1_to_10(x=training_set_labels, num_of_values=3)
            self.validation_set_labels = validation_set_labels #math_lib.one_hot_1_to_10(x=validation_set_labels, num_of_values=3)
            #self.testing_set_labels = math_lib.one_hot_1_to_10(x=testing_set_labels, num_of_values=3)
            if model_testing_set_labels is not None:
                self.model_testing_set_labels = formatter.one_hot_1_to_10(x=model_testing_set_labels, num_of_values=2)#3)
        else:
            #print(training_set_labels)
            self.training_set_labels_int = training_set_labels
            self.training_set_labels = formatter.one_hot_1_to_10(x=training_set_labels, num_of_values=10)
            #print(self.training_set_labels)
            self.validation_set_labels = formatter.one_hot_1_to_10(x=validation_set_labels, num_of_values=10)
            self.validation_set_labels_int = validation_set_labels
            #self.testing_set_labels = math_lib.one_hot_1_to_10(x=testing_set_labels, num_of_values=10)
            if model_testing_set_labels is not None:
                self.model_testing_set_labels = formatter.one_hot_1_to_10(x=model_testing_set_labels, num_of_values=10)
        # init NN
        self.nn = NeuralNet(layers=layers, learning_rate=learning_rate, loss_func=loss_function, weight_decay=weight_decay)
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
        self.last_valid_weights = []
        self.test_weights = []

    """
    Pre process network inputs
    """
    def pre_process(self):
        #print("pre process")
        #print(self.training_set_features)
        if USED_DATA_AUGMENTATION_IN_PRE_PROCESS_METHOD == DATA_AUGMENTATION_METHOD.PROPABLISTIC_ZEROIZING:
            self.training_set_features = manipulator.data_random_elements_zeroize(data=self.training_set_features, prob_of_zero=DATA_PROPABLISTIC_ZEROIZING_PROB_TO_ZERO)

        if USE_DATA_STD_PRE_PROCESS:
            self.training_set_features = manipulator.data_standardization(self.training_set_features)
            self.validation_set_features = manipulator.data_standardization(self.validation_set_features)
            self.testing_set_features = manipulator.data_standardization(self.testing_set_features)

        # print(self.training_set_features)
        # print("_________________________")
        #print(np.reshape(self.training_set_features, [-1, 32, 32, 3]))


        if not NO_CONV:
            self.training_set_features = self.training_set_features.reshape(len(self.training_set_features), 3, 32, 32)
            self.validation_set_features = self.validation_set_features.reshape(len(self.validation_set_features), 3, 32, 32)
            self.testing_set_features = self.testing_set_features.reshape(len(self.testing_set_features), 3, 32, 32)
            #print("_________________________")
            #print(self.training_set_features[0])
            # print("_________________________")
            # print(self.training_set_features[0].transpose((1,2,0)))
            # print("_________________________")
            #print(self.training_set_features.transpose((0, 2, 3, 1))) # - this is the correct one




            self.training_set_features = self.training_set_features.transpose((0, 2, 3, 1))
            self.validation_set_features = self.validation_set_features.transpose((0, 2, 3, 1))
            self.testing_set_features = self.testing_set_features.transpose((0, 2, 3, 1))




        #self.testing_set_features = self.testing_set_features.reshape(len(self.testing_set_features), 32, 32, 3)
        # print("_________________________")
        # print(self.training_set_features[0][0].shape)


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
                #print("hi")
                if iteration + self.size_of_batch < len(self.training_set_features):
                    cur_batch_size = self.size_of_batch
                else:
                    cur_batch_size = len(self.training_set_features)-iteration

                batch_features = self.training_set_features[iteration:iteration + cur_batch_size]
                batch_labels = self.training_set_labels[iteration:iteration + cur_batch_size]
                self.nn.train_iteration((batch_features, batch_labels), cur_batch_size, epoch)
                # move to next batch
                iteration += self.size_of_batch

                print(".")
                # if iteration % 256 == 0:
                #     print("iteration=" + str(iteration))

            #if (epoch > 5) and (epoch % 5 == 0):
            self.training_progress(training_features=self.training_set_features, exp_training_labels=self.training_set_labels, exp_training_labels_int=self.training_set_labels_int, include_train_acc=True)   #


            if PRINT_WEIGHTS_PERIODIC:
                if epoch % 1 == 0:
                    print("---------------------------------------")
                    print("network weights:")
                    for lw in [self.nn.get_weights()[-1]]:            #self.nn.get_weights()
                        print("---------------------------------------")
                        print(lw[0])
                        if len(lw) == 2:
                            print("weights:")
                            print(lw[1])
                        elif len(lw) == 3:
                            print("weights:")
                            print(lw[1])
                            print("biases:")
                            print(lw[2])

    """
    Run network on input data
    :param data - input data
    :return output classification for the data
    """
    def predict(self, data):
        # pass batch forward through network
        return self.nn.feed_forward(data)

    """
    Calculate network accuracy and loss 
    :param features - features (input data) to run for
    :param exp_labels - expected labels
    :param is_of_train - True if for training, False if for validation
    :return [accuracy, loss] - where accuracy is the fraction of data labeled correctly by network, loss is normalized to number of samples
    """
    def check_accuracy_and_loss(self, features, exp_labels, exp_labels_int, is_of_train):
        if self.epoch == self.max_epoch-1:
            if not is_of_train:
                self.last_valid_weights = self.nn.get_weights()
        #print("check accuracy")
        out_labels = self.predict(features)
        #print(out_labels.shape)
        out_labels_int = formatter.one_hot_to_int(out_labels)
        loss = self.loss_function.calc_diag(X=out_labels, Y=exp_labels)/len(out_labels_int)
        #print(out_labels)
        #print(exp_labels)
        #print(out_labels_int)
        #print(exp_labels_int)
        error = out_labels_int - exp_labels_int

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
    def training_progress(self, training_features, exp_training_labels, exp_training_labels_int, include_train_acc=False):
        # accuracy over training batch
        train_acc_and_loss = self.check_accuracy_and_loss(features=training_features, exp_labels=exp_training_labels, exp_labels_int=exp_training_labels_int, is_of_train=True)
        # accuracy over validation set
        if DEBUG_TRACK_VALIDATION:
            valid_acc_and_loss = self.check_accuracy_and_loss(features=self.validation_set_features, exp_labels=self.validation_set_labels, exp_labels_int=self.validation_set_labels_int, is_of_train=False)
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
        self.test_weights = self.nn.get_weights()
        pred = self.predict(data=self.testing_set_features)
        print(pred)
        print(formatter.one_hot_to_int(pred))
        return formatter.one_hot_to_int(pred)

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

    def get_last_valid_w(self):
        return self.last_valid_weights

    def get_test_w(self):
        return self.test_weights