import numpy as np
from numpy.ma.core import zeros_like


def conv_forward(x,weights,bias,conv_param):
    pass
    '''
    bias is of shape( F, ) where F is the number of filters
    weights is of shape (F, C, FH, FW) FH and FW are the filters' height and width
    x is of shape (N,C,H,W) where N is the number of images and C is the depth layer
    conv_param is a dictionary with setting for the convolutional pass
    the output should be of size (N, F, H ,W) where H and W are the new height and width for the activation map
    '''
    pad = conv_param['pad']
    stride = conv_param['stride']
    N = x.shape[0]
    C = x.shape[1]
    F = weights.shape[0]


    input_height = x.shape[2]
    filter_height = weights.shape[2]
    output_height = (input_height - filter_height + 2 * pad)// stride
    #Because both the filter and input are square we will not need to recalculate the same for width
    output_width = output_height
    filter_width = filter_height

    xpad = np.pad(x,pad_width=((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant')

    out = np.zeros(N,F,output_height,output_width)

    for i in range(output_height):
        for j in range(output_width):
                out[:,:,i,j] = bias + np.sum(xpad[:,:,i*stride:i*stride + filter_height,j*stride:j*stride + filter_width].reshape(N,1,-1) * weights.reshape(1,F,-1),axis=2)

    cache = x,weights,bias,conv_param,xpad
    return out, cache

def max_pool_forward(x,pool_param):
    '''
    pool_param is a dictionary with settings for the convolutional pass
    x is the input of size (N,F,H,W)
    out is of size (N,F,OH,OW) where OH and OW are the new height and width
    '''

    stride = pool_param['stride']
    pool_size = pool_param['size']
    N = x.shape[0]
    F = x.shape[1]

    input_height = x.shape[2]
    output_height = (input_height - pool_size) // stride + 1
    output_width = output_height

    out = np.zeros(N,F,output_height,output_width)

    for i in range(output_height):
        for j in range(output_width):
            out[:,:,i,j] = np.max(x[:,:,i*stride:i*stride + pool_size,j*stride:j*stride + pool_size].reshape(N,F,-1),axis = 2)

    cache = (x,pool_param)
    return out, cache

def conv_batch_norm_forward(x,gamma,beta,norm_param):
    '''
    x is of size (N,F,H,W)
    gamma is size (F,)
    beta is size (F,)
    '''
    eps = norm_param['eps']
    cache = None

    if norm_param['test']:
        x -= norm_param['running_mean'].reshape(1, -1, 1, 1)
        x /= np.sqrt((norm_param['running_var'] + eps).reshape(1, -1, 1, 1))
        x *= gamma.reshape(1, -1, 1, 1)
        x += beta.reshape(1, -1, 1, 1)
        return x,cache

    m = norm_param['momentum']
    mean = np.mean(x, axis = (0,2,3))
    var = np.var(x, axis = (0,2,3))
    stab = var + eps

    out = x -  mean.reshape(1,-1,1,1)
    out /= np.sqrt((stab).reshape(1, -1, 1, 1))
    out *= gamma.reshape(1, -1, 1, 1)
    out += beta.reshape(1, -1, 1, 1)

    norm_param['running_mean'] = norm_param['running_mean'] * m + (1-m) * mean
    norm_param['running_var'] = norm_param['running_var'] * m + (1 - m) * mean

    cache = x,gamma,beta,mean,eps,stab

    return out, cache

def relu(x):
    mask = (x>0)
    out = mask * x
    cache = x,mask
    return out,cache

def fc_layer_forward(x,w,b):
    '''
    x is an array of activation maps (N,F) where N is the number of examples and F is the features of each example
    w is an array containing the weights for the fully-connected layer (F,H) where H is the number of neurons
    b is an array of biases for the fully-connected layer (H,)
    out is of size (N,H)
    '''
    out = x.dot(w) + b
    cache = x,w,b
    return out, cache

def flatten(x):
    '''
    x should be of size N,F,H,W
    '''
    N = x.shape[0]
    return x.reshape(N,-1), x

def conv_backward(dout,cache):
    '''
    dout is of shape (N, F, H ,W)
    cache = x,weights,bias,conv_param,xpad
    returns dx,dw,db
    '''

    x, weights, bias, conv_param, xpad = cache

    dw = np.zeros_like(weights)
    db = np.ones_like(bias)
    dx = np.zeros_like(xpad)

    pad = conv_param['pad']
    stride = conv_param['stride']

    N = x.shape[0]
    C = x.shape[1]
    F = weights.shape[0]


    input_height = x.shape[2]
    filter_height = weights.shape[2]
    output_height = (input_height - filter_height + 2 * pad)// stride + 1
    #Because both the filter and input are square we will not need to recalculate the same for width
    output_width = output_height
    filter_width = filter_height

    for i in (range(output_height)):
        for j in (range(output_width)):
            dx[:, :, stride * i:stride * i + filter_height, stride * j:stride * j + filter_width] += np.sum(
                    weights.reshape(1,F,C,filter_height,-1) * dout[:, :, i, j].reshape(N,-1, 1, 1, 1), axis=1)
            dw += np.sum(xpad[:, :, stride * i:stride * i + filter_height, stride * j:stride * j + filter_width].reshape(N,1,C,filter_height,-1) * dout[:,:,i,j].reshape(N,F,-1,1,1),axis=0)

    db *= np.sum(dout, axis=(0, 3, 2))
    if pad>0:
        dx = dx[:, :, pad:-pad, pad:-pad]
    return dx, dw, db

def flatten_backward(dout,cache):
    #dout is of shape (N, F)
    #reshapes dout to (N, F, H, W)
    x = cache
    return dout.reshape(x.shape)

def relu_backward(dout,cache):
    #returns dx
    x, mask = cache
    return mask * dout

def fc_layer_backward(dout,cache):
    '''
    dout is of shape (N, H) Where N is the number of examples and H is the number of neurons in the layer
    returns dx,dw,db
    cache = x,w,b
    x is of shape (N,F)
    w is of shape (F,H)
    b is of shape (H,)
    '''
    x,w,b = cache
    dx = np.dot(dout,w.T)
    dw = np.dot(x.T,dout)
    db = np.sum(dout,axis=0)
    return dx, dw, db

def max_pool_backward(dout,cache):
    '''
    dout is of shape (N, F, OH, OW)
    cache is (x,pool_param)
    pool_param is a dictionary with settings for the convolutional pass
    x is the input of size (N,F,H,W)
    :param dout:
    :param cache:
    :return:
    '''
    N = dout.shape[0]
    x,pool_param = cache
    stride = pool_param['stride']
    pool_size = pool_param['size']

    dx = np.zeros_like(x)
    #remove after checking
    output_height = dout.shape[2]
    output_width = dout.shape[3]
    for i in range(output_height):
        for j in range(output_width):
            x_slice = x[:,:,i*stride:i*stride+pool_size,j*stride:j*stride+pool_size]
            max_vals = np.max(x_slice,axis = (2,3))
            mask = max_vals.reshape(N,-1,1,1) == x_slice
            dx[:,:,i*stride:i*stride+pool_size,j*stride:j*stride+pool_size] += dout[:,:,i,j].reshape(N,-1,1,1) * mask
    return dx

def conv_batch_norm_backward(dout,cache):
    '''
    x is of size(N, F, H, W)
    gamma is size(F, )
    beta is size(F, )
    '''
    x, gamma, beta, mean, eps, stab = cache

    NHW = x.shape[0] * x.shape[2] * x.shape[3]
    dx = np.zeros_like(x)
    dout_gamma = dout * gamma.reshape(1,-1,1,1)

    dout_mean_x = np.sum(dout_gamma * (x - mean.reshape(1,-1,1,1)),axis=(0,2,3))
    dout_sum = np.sum(dout_gamma,axis=(0,2,3))
    dx = (np.sqrt(stab).reshape(1,-1,1,1) * dout_gamma + (-dout_sum * np.sqrt(stab)/NHW).reshape(1,-1,1,1) - dout_mean_x.reshape(1,-1,1,1) * (x - mean.reshape(1,-1,1,1))/(NHW * np.sqrt(stab)).reshape(1,-1,1,1))/stab.reshape(1,-1,1,1)

    return dx