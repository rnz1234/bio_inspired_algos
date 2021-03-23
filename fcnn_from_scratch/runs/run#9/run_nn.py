from io_lib import *
from neural_net import *
import math_lib


if DEBUG_READ_DUMMY_DATA:
    print("selected dummy data")
    r = Reader(training_set_path="./data/dummy/train.csv",
               validation_set_path="./data/dummy/validate.csv",
               testing_set_path="./data/dummy/test.csv")
else:
    print("selected real data")
    r = Reader(training_set_path="./data/real/train.csv",
               validation_set_path="./data/real/validate.csv",
               testing_set_path="./data/real/test.csv")



r.read_training_set()
r.read_validation_set()
r.read_testing_set()

#print("training_set : " + str(r.get_training_set()))
#print("validation_set : " + str(r.get_validation_set()))
#print("testing_set : " + str(r.get_testing_set()))

print("training_set shape: " + str(r.get_training_set().shape))
print("validation_set shape: " + str(r.get_validation_set().shape))
print("testing_set shape: " + str(r.get_testing_set().shape))


#print("training labels: " + str(r.get_training_set()[:, 0]))
#print("training labels one hot: " + str(math_lib.one_hot_1_to_10(x=r.get_training_set()[:, 0],num_of_values=3)))
#exit()



print("                                 DONE READING DATA")
print("--------------------------------------------------------------------------------------------")
print("--------------------------------------------------------------------------------------------")


#print("training features: " + str(r.get_training_set()[:, 1:]))

if DEBUG_READ_DUMMY_DATA:
    # nnw = NeuralNetWrapper(training_set_features=r.get_training_set()[:, 1:],
    #                         training_set_labels=r.get_training_set()[:, 0],
    #                         validation_set_features=r.get_validation_set()[:, 1:],
    #                         validation_set_labels=r.get_validation_set()[:, 0],
    #                         testing_set_features=r.get_testing_set()[:, 1:],
    #                         #testing_set_labels=r.get_testing_set()[:, 0],
    #                         layer_widths=[4,5,3],
    #                         activation_functions=["relu", "softmax"],
    #                         loss_function="squared_error",
    #                         max_epoch=1,
    #                         learning_rate=0.1,
    #                         weight_decay=0.01,
    #                         size_of_batch=10)
    nnw = NeuralNetWrapper(training_set_features=r.get_training_set()[:, 1:],
                           training_set_labels=r.get_training_set()[:, 0],
                           validation_set_features=r.get_validation_set()[:, 1:],
                           validation_set_labels=r.get_validation_set()[:, 0],
                           testing_set_features=r.get_testing_set()[:, 1:],
                           # testing_set_labels=r.get_testing_set()[:, 0],
                           layer_widths=[4, 5, 3],
                           activation_functions=["relu", "sigmoid"],
                           loss_function="squared_error",#"cross_entropy",squared_error
                           max_epoch=50,
                           learning_rate=0.05,
                           weight_decay=0.0001,
                           size_of_batch=1)
else:
    nnw = NeuralNetWrapper(training_set_features=r.get_training_set()[:, 1:],
                           training_set_labels=r.get_training_set()[:, 0],
                           validation_set_features=r.get_validation_set()[:, 1:],
                           validation_set_labels=r.get_validation_set()[:, 0],
                           testing_set_features=r.get_testing_set()[:, 1:],
                           #testing_set_labels=r.get_testing_set()[:, 0],
                           layer_widths=[3072, 300, 10], #100
                           activation_functions=["relu", "sigmoid"],
                           loss_function="squared_error", #"cross_entropy"
                           max_epoch=600,#5000,#50,
                           learning_rate=0.03,#0.05,
                           weight_decay=0.015,
                           size_of_batch=50) #50


nnw.pre_process()
#exit()
nnw.train()

w = Writer(output_path="output.txt")
w.write(nnw.test())

if PRINT_LAST_EPOCH_RESULTS:
    w0 = Writer(output_path="last_epoch_train_pred.txt")
    w0.write(nnw.get_last_epoch_train_pred())

    w1 = Writer(output_path="last_epoch_train_labels.txt")
    w1.write(nnw.get_last_epoch_train_labels())

    w2 = Writer(output_path="last_epoch_valid_pred.txt")
    w2.write(nnw.get_last_epoch_valid_pred())

    w3 = Writer(output_path="get_last_epoch_valid_labels.txt")
    w3.write(nnw.get_last_epoch_valid_labels())


nnw.graph_training_progress()

