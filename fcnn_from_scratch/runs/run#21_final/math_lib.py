from config import *
import warnings

# wrapper to math functions

# relu
def relu(x):
    DEBUG_PRINT("relu")
    #print(x)
    x_copy = np.copy(x)
    x_copy[x_copy <= 0] = 0
    #x_copy = np.abs((x_copy-np.mean(x_copy))/np.std(x_copy))
    return x_copy

# relu derivative
def relu_derivative(x):
    DEBUG_PRINT("relu der")
    x_copy = np.copy(x)
    x_copy[x_copy <= 0] = 0
    x_copy[x_copy > 0] = 1
    return x_copy


# softmax
def softmax(x):
    DEBUG_PRINT("softmax")
    #return np.e ** x / sum(np.e ** x)
    cx = np.clip(x, -30, 30)#-500, 500)
    return np.exp(cx) / np.sum(np.exp(cx))

# softmax derivative
def softmax_derivative(x):
    DEBUG_PRINT("softmax der")
    return softmax(x) * (1 - softmax(x))

# sigmoid
def sigmoid(x):
    DEBUG_PRINT("sigmoid")
    #return np.exp(x) / (1 + np.exp(x))
    #print(x)
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            cx = np.clip(x, -500, 500)
            y = float(1)/ (1 + np.exp(-cx))
        except Warning as e:
            print('Houston, we have a warning:', e)
            print(x)
            #for m in x:
            #    print(np.exp(-m))
            exit()
    return y

# sigmoid derivative
def sigmoid_derivative(x):
    DEBUG_PRINT("sigmoid der")
    return sigmoid(x) * (1 - sigmoid(x))

# cross entropy
def cross_entropy(output, expected):
    # - sum of expected.log(output) + (1 – expected).log(1 – output)
    DEBUG_PRINT("ce")
    to_sum = expected*np.log(output) + (1 - expected) * np.log(1 - output) # -np.log(p[range(m),y])
    loss = -np.sum(to_sum)
    return loss

# cross entropy derivative
def cross_entropy_derivative(output, expected):
    DEBUG_PRINT("ce der")
    #print(expected)
    #print(output)
    #∂E /∂pi = – expected / output + (1 – expected) / (1 – output)
    return -(expected / output) + (1 - expected) / (1 - output)


# squared error
def squared_error(output, expected):
    DEBUG_PRINT("squared")
    return np.linalg.norm(expected - output)

# squared error derivative
def squared_error_derivative(output, expected):
    DEBUG_PRINT("squared der")
    return expected - output

"""
    Conversion methods between int and one hot representations
"""
def one_hot_1_to_10(x, num_of_values):
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

def one_hot_to_int(x):
    #print(x.shape)
    #print(x[0])
    #print(np.argmax(x, axis=1) + 1)
    return np.argmax(x, axis=1) + 1

"""
    Data preprocess methods
"""
def data_standardization(data):
    #print(data.shape)
    return (data - data.mean(axis=0))/data.std(axis=0)

def data_random_elements_zeroize(data, prob_of_zero):
    mask = np.random.choice(2, data.shape, p=[prob_of_zero, 1-prob_of_zero])
    return mask*data