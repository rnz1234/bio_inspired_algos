from abc import ABCMeta, abstractmethod
import numpy as np
from config import *
import math_lib

# ------------------------------------------------------
# Abstract class for a layer (skeleton)
# ------------------------------------------------------
class GeneralLayer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def back_propagate(self, layer_err):
        raise NotImplementedError()

    @abstractmethod
    def gradient(self, inputs, layer_err, size_of_batch):
        raise NotImplementedError()

    @abstractmethod
    def update_weights(self, lr, deltas, weight_decay=None):
        raise NotImplementedError()

    @abstractmethod
    def feed_forward(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def feed_forward_train(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def error(self, z, backwarded_err, dv):
        raise NotImplementedError()

    @abstractmethod
    def get_weights(self):
        raise NotImplementedError()


# ------------------------------------------------------
# Fully connected layer
# ------------------------------------------------------
class FcLayer(GeneralLayer):
    def __init__(self, w_shape, activation_func, weight_init_func, last_layer=False, init_factor=1):
        self.w_shape = w_shape
        self.activation_func = activation_func
        self.weights = weight_init_func((self.w_shape[1], self.w_shape[0]), init_factor).T
        self.bias = np.zeros((1, w_shape[1]))
        self.last_layer = last_layer
        self.inputs = []

    def back_propagate(self, layer_err):
        # print(layer_err.dot(self.weights.T).shape)
        # exit()
        return layer_err.dot(self.weights.T)

    def gradient(self, inputs, layer_err, size_of_batch):
        #print(size_of_batch)
        #print(inputs.shape)
        # print(inputs.shape)
        # print(inputs[0, 0:10])
        # print("----")
        # print(self.inputs.shape)
        # print(self.inputs[0, 0:10])
        # exit()
        activation_on_inputs = self.activation_func.calc(inputs)#self.inputs  #inputs #
        #print("act on inp " + str(activation_on_inputs.shape))
        #print(activation_on_inputs.T.dot(layer_err))
        #print("------------------")
        #print(activation_on_inputs.T.dot(layer_err)/size_of_batch)
        #exit()
        # result : [weights gradient, bias gradient]
        #print(activation_on_inputs.T.shape)
        ones_vec = np.ones((size_of_batch, 1))
        #print(ones_vec.T.shape)
        #print(layer_err.shape)
        #exit()
        return [activation_on_inputs.T.dot(layer_err)/size_of_batch, ones_vec.T.dot(layer_err)/size_of_batch]                      # was originally inputs.T.dot(layer_err)

    def update_weights(self, lr, deltas, weight_decay=None):
        # update bias
        if USED_WEIGHT_UPDATE_RULE == WEIGHT_UPDATE_RULE.BASIC:
            self.weights -= lr*deltas[0]
        elif USED_WEIGHT_UPDATE_RULE == WEIGHT_UPDATE_RULE.WEIGHT_DECAY_L1:
            self.weights -= lr*(deltas[0] - weight_decay*np.sign(self.weights))
        elif USED_WEIGHT_UPDATE_RULE == WEIGHT_UPDATE_RULE.WEIGHT_DECAY_L2:
            self.weights -= lr*(deltas[0] - weight_decay*self.weights)
            #print(self.weights)
        else:
            exit("bad weights update rule")

        # update bias
        self.bias -= lr*deltas[1]
        #self.bias = np.zeros((1, self.w_shape[1]))

    def feed_forward(self, inputs):
        bias_for_batch = np.repeat(self.bias, inputs.shape[0], axis=0)
        return self.activation_func.calc(inputs.dot(self.weights) + bias_for_batch)

    def feed_forward_train(self, inputs):
        self.inputs =  inputs
        bias_for_batch = np.repeat(self.bias, inputs.shape[0], axis=0)
        #print(self.weights.shape)
        #print(inputs.shape)
        z = inputs.dot(self.weights) + bias_for_batch

        if USE_DROPOUT:
            if not self.last_layer:
                dv = np.random.binomial(1, DROPOUT_PROB, size=z.shape) / DROPOUT_PROB               
                z = z*dv
            else:
                dv = 0
        else:
            dv = 0

        return (z, self.activation_func.calc(z), dv)

    def error(self, z, backwarded_err, dv):
        if USE_DROPOUT:
            if not self.last_layer:
                return backwarded_err * self.activation_func.derivative(z) * dv
            else:
                return backwarded_err * self.activation_func.derivative(z)
        else:
            return backwarded_err * self.activation_func.derivative(z)

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

# ------------------------------------------------------
# Flattening layer (put between convolutional layers and fully connected layers)
# ------------------------------------------------------
class FlatteningLayer(GeneralLayer):
    def __init__(self, shape):
        self.shape = shape

    def back_propagate(self, layer_err):
        return np.reshape(layer_err, (layer_err.shape[0],) + self.shape)

    def gradient(self, inputs, layer_err, size_of_batch):
        return 0.

    def update_weights(self, lr, deltas, weight_decay=None):
        pass

    def feed_forward(self, inputs):
        return np.reshape(inputs, (inputs.shape[0],-1))

    def feed_forward_train(self, inputs):
        #print(inputs.shape)
        z = np.reshape(inputs, (inputs.shape[0],-1))
        return (z, z, 0)

    def error(self, z, next_layer_err, dv):
        return next_layer_err

    def get_weights(self):
        return None


# ------------------------------------------------------
# Max Pooling layer
# ------------------------------------------------------
class MaxPoolLayer(GeneralLayer):
    def __init__(self, pool_size, strides):
        self.pool = pool_size
        self.s = strides
        self.batch_s = 0
        self.filters_n = 0
        self.h = 0
        self.w = 0
        self.inputs_cache = []


    def feed_forward_train(self, inputs):
        out = self.feed_forward(inputs)
        return (out, out, 0)

    def feed_forward(self, inputs):
        self.inputs_cache = inputs
        self.batch_s = inputs.shape[0]
        self.filters_n = inputs.shape[3]
        self.h = inputs.shape[1]
        self.w = inputs.shape[2]
        h = self.h
        w = self.w
        hn = int(1 + (h - self.pool)/self.s)
        wn = int(1 + (w - self.pool)/self.s)
        out = np.zeros((self.batch_s,hn,wn,self.filters_n))
        for n in range(self.batch_s): # batch
            for depth in range(self.filters_n): # input depth
                for r in range(0,h,self.s):
                    for c in range(0,w,self.s):
                        out[n,int(r/self.s),int(c/self.s),depth] = np.max(inputs[n,r:r+self.pool, c:c+self.pool, depth])



        #print(out.shape)
        return out

    def back_propagate(self, layer_err):
        h = self.h
        w = self.w
        #layer_err.shape[0]
        hn = layer_err.shape[1]
        wn = layer_err.shape[2]
        bwerr = np.zeros((self.batch_s, h, w, self.filters_n))
        for n in range(self.batch_s):
            for depth in range(self.filters_n):
                for r in range(hn):
                    for c in range(wn):
                        inputs_pool = self.inputs_cache[n,r*self.s:r*self.s+self.pool,c*self.s:c*self.s+self.pool,depth]
                        mask_vec = (inputs_pool == np.max(inputs_pool))

                        bwerr[n,r*self.s:r*self.s+self.pool,c*self.s:c*self.s+self.pool,depth] = mask_vec*layer_err[n,r,c,depth]

        #print(bwerr.shape)
        return bwerr

    def gradient(self, inputs, layer_err, size_of_batch):
        return 0

    def update_weights(self, lr, deltas, weight_decay=None):
        pass

    def error(self, z, backwarded_err, dv):
        return backwarded_err

    def get_weights(self):
        return 0

# ------------------------------------------------------
# Convolution layer
# ------------------------------------------------------
class ConvLayer(GeneralLayer):
    def __init__(self, f_shape, activation_func, filter_init_func, strides, use_padding, init_factor=1):
        self.f_shape = f_shape
        self.strides = strides
        self.filters = filter_init_func(self.f_shape, init_factor)
        self.activation_func = activation_func
        self.use_padding = use_padding
        self.padding_size = int((self.f_shape[0]-1)/2)
        self.inputs_unpadded_size = 0
        self.inputs = []

    def back_propagate(self, layer_err):

        #if self.use_padding:
        # print("-------------------")
        #print(layer_err.shape)
        #exit()
        if self.use_padding:
            bfmap_shape = self.inputs_unpadded_size + 2*self.padding_size #(layer_err.shape[1] - 1) * self.strides + self.f_shape[0] #layer_err.shape[1] * self.strides
        else:
            bfmap_shape = self.inputs_unpadded_size
        #print(bfmap_shape)
        fmap_bwd = np.zeros((layer_err.shape[0], bfmap_shape, bfmap_shape, self.f_shape[-2]))
        #print(layer_err.shape)
        #print(fmap_bwd.shape)
        #exit()
        #print(fmap_bwd.shape)
        #size = int(fmap_bwd.shape[1] / self.strides) #int((fmap_bwd.shape[1] - self.f_shape[0]) / self.strides + 1)
        size = int((fmap_bwd.shape[1] - self.f_shape[0]) / self.strides + 1)
        #print(size)
        # else:
        #     #print(layer_err.shape)
        #     bfmap_shape = (layer_err.shape[1] - 1) * self.strides + self.f_shape[0]
        #     #bfmap_shape = layer_err.shape[1] * self.strides
        #     #print(bfmap_shape)
        #     fmap_bwd = np.zeros((layer_err.shape[0], bfmap_shape, bfmap_shape, self.f_shape[-2]))
        #     #print(fmap_bwd.shape)
        #     size = int((fmap_bwd.shape[1] - self.f_shape[0]) / self.strides + 1)
        #     #print(size)

        #exit()
        s = self.strides
        wn = self.f_shape[0]
        hn = self.f_shape[1]
        #print(self.filters.shape)
        #exit()
        #print("-------------------")


        for n in range(layer_err.shape[0]):   # over batch
            for depth in range(layer_err.shape[3]): # filters
                for r in range(0,self.inputs_unpadded_size,s): # H = self.inputs_unpadded_size
                    for c in range(0,self.inputs_unpadded_size,s): # W = self.inputs_unpadded_size
                        #print(self.filters[:,:,:,depth].shape)
                        fmap_bwd[n,r:r+hn,c:c+wn,:] += layer_err[n, int(r/s), int(c/s), depth] * self.filters[:,:,:,depth]
                        #print(fmap_bwd.shape)

        # for j in range(size):
        #     for i in range(size):
        #         # print(self.filters[np.newaxis, ...])
        #         # print("-----")
        #         # print(layer_err[:, j:j + 1, i:i + 1, np.newaxis, :])
        #         # print("-----")
        #         #print(self.filters[np.newaxis, ...].shape)
        #         #print(layer_err[:, j:j + 1, i:i + 1, np.newaxis, :].shape)
        #         #print((self.filters[np.newaxis, ...] * layer_err[:, j:j + 1, i:i + 1, np.newaxis, :]).shape)
        #         #print(np.sum(self.filters[np.newaxis, ...] * layer_err[:, j:j + 1, i:i + 1, np.newaxis, :], axis=4).shape)
        #         # exit()
        #         #print(self.f_shape[0])
        #         #print(self.f_shape[1])
        #         #print(fmap_bwd[:, j * self.strides:j * self.strides + self.f_shape[0], i * self.strides:i * self.strides + self.f_shape[1]].shape)
        #
        #         #print((j * s,  j * s + w,  i * s , i * s + h))
        #         fmap_bwd[:, j * s : j * s + ww, i * s : i * s + hn] += np.sum(self.filters[np.newaxis, ...] * layer_err[:, j:j + 1, i:i + 1, np.newaxis, :], axis=4)

        if self.use_padding:
            h = size - 2*self.padding_size
            w = size - 2*self.padding_size
            p = self.padding_size
            r_to_del = list(range(self.padding_size)) + list(range(h + p, h + 2*p,1))
            c_to_del = list(range(self.padding_size)) + list(range(h + p, h + 2*p, 1))
            fmap_bwd = np.delete(fmap_bwd, r_to_del, axis=1)
            fmap_bwd = np.delete(fmap_bwd, c_to_del, axis=2)
            #print(fmap_bwd.shape)
            #exit()

        return fmap_bwd

    def gradient(self, inputs, layer_err, size_of_batch):
        # print(inputs.shape)
        # print(inputs[0,0:4,0:4,0])
        # print("----")
        # print(self.inputs.shape)
        # print(self.inputs[0,0:4,0:4,0])
        # exit()
        activation_on_inputs = self.activation_func.calc(inputs) #self.inputs #inputs #
        #inputs_sum = np.sum(activation_on_inputs, axis=0)              #   was originally np.sum(inputs, axis=0)
        #print(activation_on_inputs.shape)
        #print(layer_err.shape)

        #total_layer_err = np.sum(layer_err, axis=(0, 1, 2))
        #print(total_layer_err)

        filters_err = np.zeros(self.f_shape)
        #print(filters_err.shape)
        #exit()
        # size = int((inputs.shape[1] - self.f_shape[0]) / self.strides + 1)
        #
        # for j in range(size):
        #     for i in range(size):
        #         #print(inputs_sum[j * self.strides:j * self.strides + self.f_shape[0],
        #         #               i * self.strides:i * self.strides + self.f_shape[1], :, np.newaxis].shape)
        #         filters_err += inputs_sum[j * self.strides:j * self.strides + self.f_shape[0],
        #                        i * self.strides:i * self.strides + self.f_shape[1], :, np.newaxis]
        # return (filters_err * total_layer_err)/size_of_batch

        if self.use_padding:
            # ---------------------
            #activation_on_inputs = self.activation_func.calc(inputs)
            npad = ((0, 0), (int((self.f_shape[0] - 1) / 2), int((self.f_shape[0] - 1) / 2)),
                    (int((self.f_shape[0] - 1) / 2), int((self.f_shape[0] - 1) / 2)), (0, 0))
            # print(npad)
            input_to_take = np.pad(activation_on_inputs, pad_width=npad, mode='constant', constant_values=0)
        else:
            input_to_take = activation_on_inputs


        s = self.strides
        hn = self.f_shape[0]
        wn = self.f_shape[1]
        #dw = np.zeros(self.f_shape)
        for n in range(layer_err.shape[0]):            # over batch
            for depth in range(layer_err.shape[3]):      # over filters (output channels)
                for r in range(layer_err.shape[1]):
                    for c in range(layer_err.shape[2]):
                        filters_err[:, :, :, depth] += layer_err[n,r,c, depth] * input_to_take[n,r*s:r*s+hn, c*s:c*s+wn, :]

        return filters_err

    def update_weights(self, lr, deltas, weight_decay=None):
        if USED_WEIGHT_UPDATE_RULE == WEIGHT_UPDATE_RULE.BASIC:
            #print(deltas)
            self.filters -= lr*deltas
        elif USED_WEIGHT_UPDATE_RULE == WEIGHT_UPDATE_RULE.WEIGHT_DECAY_L1:
            self.filters -= lr*(deltas - weight_decay*np.sign(self.filters))
        elif USED_WEIGHT_UPDATE_RULE == WEIGHT_UPDATE_RULE.WEIGHT_DECAY_L2:
            self.filters -= lr*(deltas - weight_decay*self.filters)
            #print(self.filters)
            # print("filters")
            # print(self.filters)
        else:
            exit("bad weights update rule")

    def feed_forward(self, inputs):
        if self.use_padding:
            #print(inputs.shape)
            #print(int((self.f_shape[0]-1)/2))
            npad = ((0, 0), (self.padding_size, self.padding_size), (self.padding_size, self.padding_size), (0, 0))
            #print(npad)
            input_to_take = np.pad(inputs, pad_width=npad, mode='constant', constant_values=0)
            #print(input_to_take.shape)
        else:
            input_to_take = inputs

        size = int((input_to_take.shape[1] - self.f_shape[0]) / self.strides + 1)

        fmap = np.zeros((inputs.shape[0], size, size, self.f_shape[-1]))

        for j in range(size):
            for i in range(size):
                fmap[:, j, i, :] = np.sum(input_to_take[:, j * self.strides:j * self.strides + self.f_shape[0],
                                          i * self.strides:i * self.strides + self.f_shape[1], :,
                                          np.newaxis] * self.filters, axis=(1, 2, 3))
        return self.activation_func.calc(fmap)

    def feed_forward_train(self, inputs):
        self.inputs = inputs
        # print(inputs.shape)
        # exit()
        self.inputs_unpadded_size = inputs.shape[1]
        if self.use_padding:
            #print(inputs.shape)
            #print(int((self.f_shape[0]-1)/2))
            npad = ((0, 0), (int((self.f_shape[0]-1)/2), int((self.f_shape[0]-1)/2)), (int((self.f_shape[0]-1)/2), int((self.f_shape[0]-1)/2)), (0, 0))
            #print(npad)
            input_to_take = np.pad(inputs, pad_width=npad, mode='constant', constant_values=0)
            #print(input_to_take.shape)
        else:
            input_to_take = inputs

        size = int((input_to_take.shape[1] - self.f_shape[0]) / self.strides + 1)
        #print(size)
        # print("layer size = " + str(self.f_shape[-1]) + " channels of "+ str(size) + "X" + str(size))
        # print(size)
        # print(self.filters.shape)
        fmap = np.zeros((inputs.shape[0], size, size, self.f_shape[-1]))
        #print(fmap.shape)
        for j in range(size):
            for i in range(size):
                # print(inputs[:, j * self.strides:j * self.strides + self.f_shape[0],
                #                           i * self.strides:i * self.strides + self.f_shape[1], :,
                #                           np.newaxis].shape)
                # #print(self.filters.shape)
                # x = inputs[:, j * self.strides:j * self.strides + self.f_shape[0],
                #     i * self.strides:i * self.strides + self.f_shape[1], :,
                #         np.newaxis]
                #
                # print(x.shape)
                # #print(self.filters[0])
                # npad = ((0, 0), (1, 1), (1, 1), (0, 0), (0, 0))
                # y = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
                # print(y.shape)
                #print(self.filters.shape)
                # prinbackwarded_errt((inputs[:, j * self.strides:j * self.strides + self.f_shape[0],
                #                           i * self.strides:i * self.strides + self.f_shape[1], :,
                #                           np.newaxis] * self.filters).shape)
                #print((y * self.filters).shape)


                fmap[:, j, i, :] = np.sum(input_to_take[:, j * self.strides:j * self.strides + self.f_shape[0],
                                          i * self.strides:i * self.strides + self.f_shape[1], :,
                                          np.newaxis] * self.filters, axis=(1, 2, 3))

        #print(help(self.activation_func))
        #print(self.activation_func.calc(fmap).shape)
        #exit()
        return (fmap, self.activation_func.calc(fmap), 0)

        #print(inputs.shape, self.f_shape, self.strides)
        #print(inputs.shape[2], self.f_shape[0], self.strides)
        # size = int((inputs.shape[2] - self.f_shape[0]) / self.strides + 1)
        # #print(inputs.shape[0], size, size, self.f_shape[-1])
        # fmap = np.zeros((inputs.shape[0], size, size, self.f_shape[-1]))
        # #print(inputs.shape)
        # #print(self.filters.shape)
        # for j in range(size):
        #     for i in range(size):
        #         print(inputs[0, 0, j * self.strides:j * self.strides + self.f_shape[0],
        #                                   i * self.strides:i * self.strides + self.f_shape[1],
        #                                   np.newaxis].shape)
        #         print(self.filters.shape)
        #         fmap[:, j, i, :] = np.sum(inputs[:, :, j * self.strides:j * self.strides + self.f_shape[0],
        #                                   i * self.strides:i * self.strides + self.f_shape[1],
        #                                   np.newaxis] * self.filters, axis=(1, 2, 3))
        # return (fmap, self.activation_func.calc(fmap))

    def error(self, z, backwarded_err, dv):
        # print(self.activation_func.derivative(z).shape)
        # print(backwarded_err.shape)
        # print((backwarded_err * self.activation_func.derivative(z)).shape)
        # exit()
        return backwarded_err * self.activation_func.derivative(z)

    def get_weights(self):
        return self.filters