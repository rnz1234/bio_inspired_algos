from io_lib import *
from neural_net import *
from layers import *

# read datasets
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

# create the network
if DEBUG_READ_DUMMY_DATA:
    pass
else:
    # layers = [
    #     # widthX, widthY, number of input channels, number of output channels
    #     #ConvLayer((3, 3, 3, 96), strides=2, activation_func=ReluActivation, filter_init_func=initializer.init_normal_filters),
    #     ConvLayer((5, 5, 3, 6), strides=1, use_padding=True, activation_func=ReluActivation, filter_init_func=initializer.init_normal_filters),
    #     MaxPoolLayer(pool_size=2, strides=2),
    #     ConvLayer((5, 5, 6, 16), strides=1, use_padding=True, activation_func=ReluActivation, filter_init_func=initializer.init_normal_filters),
    #     MaxPoolLayer(pool_size=2, strides=2),
    #     ConvLayer((5, 5, 16, 120), strides=1, use_padding=True, activation_func=ReluActivation, filter_init_func=initializer.init_normal_filters),
    #     FlatteningLayer((8, 8, 120)),
    #     #FlatteningLayer((32, 32, 3)),
    #     FcLayer((8 * 8 * 120, 84), activation_func=ReluActivation, weight_init_func=initializer.init_normal_weights),
    #     FcLayer((84, 10), activation_func=SigmoidActivation, weight_init_func=initializer.init_normal_weights, last_layer=True)
    # ]

    # layers = [
    #     # widthX, widthY, number of input channels, number of output channels
    #     # ConvLayer((3, 3, 3, 96), strides=2, activation_func=ReluActivation, filter_init_func=initializer.init_normal_filters),
    #     ConvLayer((5, 5, 3, 40), strides=2, use_padding=True, activation_func=relu_act,
    #                filter_init_func=initializer.init_normal_filters, init_factor=np.sqrt(1.0 / (32*32*3 + 16*16*40))),
    #     ConvLayer((5, 5, 40, 120), strides=2, use_padding=True, activation_func=relu_act,
    #               filter_init_func=initializer.init_normal_filters, init_factor=np.sqrt(1.0 / (16*16*40 + 8*8*120))),
    #     # ConvLayer((5, 5, 6, 9), strides=2, use_padding=True, activation_func=ReluActivation,
    #     #           filter_init_func=initializer.init_normal_filters),
    #     FlatteningLayer((8, 8, 120)),
    #     #FlatteningLayer((30, 30, 3)),
    #     FcLayer((8 * 8 * 120, 512), activation_func=sigmoid_act, weight_init_func=initializer.init_normal_weights, init_factor=np.sqrt(1.0 / (8*8*120 + 512.))),
    #     FcLayer((512, 10), activation_func=linear_act, weight_init_func=initializer.init_normal_weights,
    #             last_layer=True, init_factor=np.sqrt(1.0 / (512.)))
    # ]

    # layers = [
    #     # widthX, widthY, number of input channels, number of output channels
    #     # ConvLayer((3, 3, 3, 96), strides=2, activation_func=ReluActivation, filter_init_func=initializer.init_normal_filters),
    #     ConvLayer((5, 5, 3, 40), strides=2, use_padding=True, activation_func=lkrelu_act,
    #               filter_init_func=initializer.init_normal_filters,
    #               init_factor=np.sqrt(1.0 / (32 * 32 * 3 + 16 * 16 * 40))),
    #     ConvLayer((5, 5, 40, 60), strides=2, use_padding=True, activation_func=lkrelu_act,
    #               filter_init_func=initializer.init_normal_filters,
    #               init_factor=np.sqrt(1.0 / (16 * 16 * 40 + 8 * 8 * 40))),
    #     ConvLayer((5, 5, 60, 120), strides=1, use_padding=True, activation_func=lkrelu_act,
    #               filter_init_func=initializer.init_normal_filters,
    #               init_factor=np.sqrt(1.0 / (8 * 8 * 120 + 4 * 4 * 120))),
    #     # ConvLayer((5, 5, 6, 9), strides=2, use_padding=True, activation_func=ReluActivation,
    #     #           filter_init_func=initializer.init_normal_filters),
    #     FlatteningLayer((8, 8, 120)),
    #     # FlatteningLayer((30, 30, 3)),
    #     FcLayer((8 * 8 * 120, 512), activation_func=sigmoid_act, weight_init_func=initializer.init_normal_weights,
    #             init_factor=np.sqrt(1.0 / (4 * 4 * 120 + 512.))),
    #     FcLayer((512, 10), activation_func=linear_act, weight_init_func=initializer.init_normal_weights,
    #             last_layer=True, init_factor=np.sqrt(1.0 / (512.)))
    # ]

    layers = [
        # widthX, widthY, number of input channels, number of output channels
        # ConvLayer((3, 3, 3, 96), strides=2, activation_func=ReluActivation, filter_init_func=initializer.init_normal_filters),
        ConvLayer((3, 3, 3, 16), strides=1, use_padding=True, activation_func=relu_act,
                  filter_init_func=initializer.init_normal_filters,
                  init_factor=np.sqrt(1.0 / (32 * 32 * 3 + 32 * 32 * 16))),
        MaxPoolLayer(pool_size=2, strides=2),
        ConvLayer((3, 3, 16, 48), strides=1, use_padding=True, activation_func=lkrelu_act,
                  filter_init_func=initializer.init_normal_filters,
                  init_factor=np.sqrt(1.0 / (16 * 16 * 16 + 8 * 8 * 48))),
        MaxPoolLayer(pool_size=2, strides=2),
        FlatteningLayer((8, 8, 48)),
        # FlatteningLayer((30, 30, 3)),
        FcLayer((8 * 8 * 48, 256), activation_func=lkrelu_act, weight_init_func=initializer.init_normal_weights,
                init_factor=np.sqrt(1.0 / (8 * 8 * 48 + 256.))),
        FcLayer((256, 10), activation_func=sigmoid_act, weight_init_func=initializer.init_normal_weights,
                last_layer=True, init_factor=np.sqrt(1.0 / (256.)))
    ]

    # layers = [
    #     FcLayer((3072, 100), activation_func=ReluActivation, weight_init_func=initializer.init_normal_weights),
    #     FcLayer((100, 10), activation_func=SigmoidActivation, weight_init_func=initializer.init_normal_weights, last_layer=True)
    # ]

    nnw = NeuralNetWrapper(training_set_features=r.get_training_set()[:, 1:],
                           training_set_labels=r.get_training_set()[:, 0],
                           validation_set_features=r.get_validation_set()[:, 1:],
                           validation_set_labels=r.get_validation_set()[:, 0],
                           testing_set_features=r.get_testing_set()[:, 1:],
                           #testing_set_labels=r.get_testing_set()[:, 0],
                           layers=layers,
                           loss_function=mse_loss,#cross_entropy_loss,#mse_loss,
                           max_epoch=73,#5000,#50,
                           learning_rate=0.01,#0.03,#0.05,
                           weight_decay=0.0012,
                           size_of_batch=32) #50

# network input pre-process
nnw.pre_process()

# train network
nnw.train()


# debug prints
if PRINT_LAST_EPOCH_RESULTS:
    w4 = Writer(output_path="get_last_valid_w.txt")
    w4.write(nnw.get_last_valid_w(), special=True)

# run network on test set after training and write to output file
w = Writer(output_path="output.txt")
w.write(nnw.test())

# debug prints
if PRINT_LAST_EPOCH_RESULTS:
    w0 = Writer(output_path="last_epoch_train_pred.txt")
    w0.write(nnw.get_last_epoch_train_pred())

    w1 = Writer(output_path="last_epoch_train_labels.txt")
    w1.write(nnw.get_last_epoch_train_labels())

    w2 = Writer(output_path="last_epoch_valid_pred.txt")
    w2.write(nnw.get_last_epoch_valid_pred())

    w3 = Writer(output_path="get_last_epoch_valid_labels.txt")
    w3.write(nnw.get_last_epoch_valid_labels())

    w5 = Writer(output_path="get_test_w.txt")
    w5.write(nnw.get_test_w(), special=True)




# to record max validation and the epoch it happened during the training
if PRINT_MAX_VALIDATION_ACC_AND_EPOCH:
    w4 = Writer(output_path="max_validation_acc_and_epoch.txt")
    w4.write_float(np.array(nnw.get_max_validation_acc_and_epoch()))

# show training progress graph
if DEBUG_GRAPH_TRAINING_PROGRESS:
    nnw.graph_training_progress()

