from abc import ABCMeta, abstractmethod
import warnings
import numpy as np

##################################################################
#                         DATA FORMATTING
##################################################################

# formatting methods aggregator
class Formatter(object):
    def __init__(self):
        pass

    """
        Conversion methods between int and one hot representations
    """
    def one_hot_1_to_10(self, x, num_of_values):
        #print(x)
        oh = np.zeros((len(x), num_of_values))
        #print(oh)
        oh[np.arange(len(x)), x.astype(np.int)-1] = 1
        #print(oh)
        return oh
        # x[x == 1] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        # x[x == 2] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        # x[x == 3] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        # x[x == 4] = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        # x[x == 5] = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        # x[x == 6] = np.array([0, 0, 0,  WEIGHT PRINT 0, 1, 0, 0, 0, 0, 0])
        # x[x == 7] = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        # x[x == 8] = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        # x[x == 9] = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        # x[x == 10] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    def one_hot_to_int(self, x):
        #print(x)
        #print(x[0])
        #print(np.argmax(x, axis=1) + 1)
        return np.argmax(x, axis=1) + 1


##################################################################
#                           DATA MANIPULATION
##################################################################

# manipulating methods aggregator
class Manipulator(object):
    def __init__(self):
        pass

    def data_standardization(self, data):
        #print(data.shape)
        return (data - data.mean(axis=0))/data.std(axis=0)

    def data_random_elements_zeroize(self, data, prob_of_zero):
        mask = np.random.choice(2, data.shape, p=[prob_of_zero, 1-prob_of_zero])
        return mask*data


##################################################################
#                           DATA INITIALER
##################################################################

class Initializer(object):
    def __init__(self):
        pass

    def init_normal_weights(self, size, init_factor=1):
        return np.random.randn(size[0], size[1])*init_factor #np.random.normal(size=size)

    def init_normal_filters(self, size, init_factor=1):
        return np.random.normal(size=size)*init_factor #* np.sqrt(1.0 / (32*32*3 + 32*32*3))

##################################################################
#                           MATH FUNCTIONS
##################################################################

# -----------------------------------------------------------------
# "math function" is an abstraction of mathematical function with
# range & image. It gives means to calculate the function for given
# input and in case the function is continuous it gives means to
# derive it. Here 2 use cases are activations and loss functions.
# -----------------------------------------------------------------

# general abstract class for math function
class GeneralMathFunc(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def calc(self, x):
        raise NotImplementedError()

    @abstractmethod
    def derivative(self, x):
        raise NotImplementedError()

# --------------------------------------
#   activations
# --------------------------------------
class GeneralActivation(GeneralMathFunc):
    def __init__(self):
        pass

# relu
class ReluActivation(GeneralActivation):
    def calc(self, x):
        return np.maximum(0, x)
    def derivative(self, x):
        return 1. * (x > 0)

# leaky relu
class LeakyReluActivation(GeneralActivation):
    def calc(self, x):
        return np.maximum(0.01, x)
    def derivative(self, x):
        g = 1. * (x > 0)
        g[g == 0.] = 0.01
        return g

# linear
class LinearActivation(GeneralActivation):
    def calc(self, x):
        return x

    def derivative(self, x):
        return 1.

# sigmoid
class SigmoidActivation(GeneralActivation):
    def calc(self, x):
        #return 1. / (1. + np.exp(-x))
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                cx = np.clip(x, -500, 500)
                y = float(1) / (1 + np.exp(-cx))
            except Warning as e:
                print('warning:', e)
                print(x)
                # for m in x:
                #    print(np.exp(-m))
                exit()
        return y

    def derivative(self, x):
        y = self.calc(x)
        return y * (1. - y)

class SoftmaxActivation(GeneralActivation):
    def calc(self, x):
        cx = np.clip(x, -30, 30)
        exp_x = np.exp(cx - np.max(cx, axis=1)[..., np.newaxis])
        return exp_x/np.sum(exp_x, axis=1, keepdims=True)



# def pad_with(vector, pad_width, iaxis, kwargs):
#     pad_value = kwargs.get('padder', 0)
#     vector[:pad_width[0]] = pad_value
#     vector[-pad_width[1]:] = pad_value
#     return vector



# --------------------------------------
#   loss functions
# --------------------------------------
class Loss(GeneralMathFunc):
    pass

X_MINUS_Y = True # When True, in gradient descent add to the weights/biases -lr*(...). When False, in gradient descent add to the weights/biases +lr*(...)

class MeanSquaredErrorLoss(Loss):
    def calc(self, X, Y): #(X, Y)):
        if X_MINUS_Y:
            return np.linalg.norm(X-Y)
        else:
            return np.linalg.norm(Y-X) #(1. / 2. * X.shape[0]) * ((X - Y) ** 2.)   #
    def calc_diag(self, X, Y):
        return self.calc(X,Y)
        #return np.linalg.norm(Y-X)
    def derivative(self, X, Y): #(X, Y)):
        if X_MINUS_Y:
            return X-Y
        else:
            return Y-X #(X - Y) / X.shape[0]

class CrossEntropyLoss(Loss):
    def _calc_softmax(self, X):
        cx = np.clip(X, -30, 30)
        exp_x = np.exp(cx - np.max(cx, axis=1)[..., np.newaxis])
        return exp_x/np.sum(exp_x, axis=1, keepdims=True)
    def calc(self, X, Y): #(X, Y)):
        if X_MINUS_Y:
            sf = self._calc_softmax(X)
            # print(X.shape)
            # print(Y.shape)
            # print(Y)
            #print(sf[np.arange(X.shape[0]), np.argmax(Y, axis=1)])
            #print(-np.log(sf[np.arange(X.shape[0]), np.argmax(Y, axis=1)]) / X.shape[0])
            #exit()
            return -np.log(sf[np.arange(X.shape[0]), np.argmax(Y, axis=1)]) / X.shape[0]
        else:
            sf = self._calc_softmax(Y)
            # print(X.shape)
            # print(Y.shape)
            # print(Y)
            # print(sf[np.arange(X.shape[0]), np.argmax(Y, axis=1)])
            # print(-np.log(sf[np.arange(X.shape[0]), np.argmax(Y, axis=1)]) / X.shape[0])
            # exit()
            return -np.log(sf[np.arange(Y.shape[0]), np.argmax(X, axis=1)]) / Y.shape[0]
    def calc_diag(self, X, Y):
        return np.sum(self.calc(X,Y))
    def derivative(self, X, Y): #(X, Y)):
        if X_MINUS_Y:
            err = self._calc_softmax(X)
            return (err - Y) / X.shape[0]
        else:
            err = self._calc_softmax(Y)
            return (err - X) / Y.shape[0]


##################################################################
#                           REALIZATIONS
##################################################################

# here are several objects that implement the classes above and
# can be used by the software stack

# formatter
formatter = Formatter()
# manipulator
manipulator = Manipulator()
# initializer
initializer = Initializer()
# activation objects
relu_act = ReluActivation()
lkrelu_act = LeakyReluActivation()
sigmoid_act = SigmoidActivation()
linear_act = LinearActivation()
# loss functions objects
mse_loss = MeanSquaredErrorLoss()
cross_entropy_loss = CrossEntropyLoss()
