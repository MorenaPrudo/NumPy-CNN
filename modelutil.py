import copy
from layers import *
import numpy as np
import gc


class Model:
    def __init__(self,reg=0,learning_rate=1e-3, num_classes = 12,epochs=40,batch_size=16,hidden_dim=100):
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.initialization = 1e-2
        self.params = { 'w1': self.initialization * np.random.randn(32,3,3,3), #(F,D,H,W)
                        'b1': np.zeros(32),
                        'w2' :self.initialization * np.random.randn(64,32,4,4),
                        'b2': np.zeros(64),
                        'w3' :self.initialization * np.random.randn(64,64,4,4),
                        'b3': np.zeros(64),
                        'w4' : self.initialization * np.random.randn(26*26*64,self.hidden_dim),
                        'b4': np.zeros(self.hidden_dim),
                        'w5': self.initialization * np.random.randn(self.hidden_dim, self.num_classes),
                        'b5': np.zeros(self.num_classes)
        }

        for k, v in self.params.items():
            self.params[k] = v.astype(np.float32)

        self.grads = {}
        self.learning_rate = learning_rate
        self.reg = reg
        self.adam_momentum = {}
        self.adam_velocity = {}
        self.loss = 0
        self.running_values = {'running_mean':0,'running_var':0}
        self.running_values2 = {'running_mean': 0, 'running_var': 0}
        self.best_val_accuracy = 0
        self.best_params = {}

    def train(self,X,y,X_val,y_val,X_test,y_test):
        total_iter = 0
        for epoch in range(self.epochs):
            print("Epoch {}/{}".format(epoch+1,self.epochs))
            self.learning_rate = self.learning_rate * 0.9
            num_of_training = X.shape[0]
            num_iter_per_epoch = num_of_training // self.batch_size if num_of_training % self.batch_size == 0 else num_of_training // self.batch_size + 1

            random_indices = np.random.permutation(X.shape[0])
            X_shuffled = X[random_indices]
            y_shuffled = y[random_indices]

            for batch in range(num_iter_per_epoch):
                gc.collect()
                np.get_default_memory_pool().free_all_blocks()

                batch_size = min(self.batch_size, num_of_training)
                self.train_pass(X_shuffled[batch*self.batch_size:batch*self.batch_size + batch_size],y_shuffled[batch*self.batch_size:batch*self.batch_size+batch_size])
                num_of_training -= batch_size
                total_iter +=1
                self.update_params(total_iter)
                print("Current loss: " + str(self.loss))
            self.test(X_val,y_val,False,"val")
            #self.test(X, y, False, "train")
        self.params = self.best_params
        self.test(X_test, y_test, False, "test")

    def train_pass(self,X,y):
        params = self.params
        reg = self.reg

        cache_array = []
        conv_param1 = {'pad': 0, 'stride': 1}
        conv_param2 = {'pad': 0, 'stride': 1}
        conv_param3 = {'pad': 0, 'stride': 2}
        pool_param = {'size': 2, 'stride': 2}
        #norm_param1 = {'test': False, 'running_values':self.running_values, 'momentum': 0.9, 'eps': 1e-8}

        out, cache = conv_forward(X, params['w1'], params['b1'], conv_param1)
        cache_array.append(cache)

        out, cache = relu(out)
        cache_array.append(cache)

        out, cache = max_pool_forward(out, pool_param)
        cache_array.append(cache)

        out, cache = conv_forward(out, params['w2'], params['b2'], conv_param2)
        cache_array.append(cache)

        out, cache = relu(out)
        cache_array.append(cache)

        out, cache = max_pool_forward(out, pool_param)
        cache_array.append(cache)

        out, cache = conv_forward(out, params['w3'], params['b3'], conv_param3)
        cache_array.append(cache)

        out, cache = relu(out)
        cache_array.append(cache)

        out, cache = flatten(out)
        cache_array.append(cache)

        out, cache = fc_layer_forward(out, params['w4'], params['b4'])
        cache_array.append(cache)

        scores, cache = fc_layer_forward(out, params['w5'], params['b5'])
        cache_array.append(cache)

        loss, dx = softmax_loss(scores, y)


        #regularization

        for name, param in params.items():
            if 'w' in name:
                loss += reg * np.sum(param**2)/2
                self.loss = loss

        dx, dw5, db5 = fc_layer_backward(dx, cache_array.pop())
        dx, dw4, db4 = fc_layer_backward(dx, cache_array.pop())
        dx = flatten_backward(dx, cache_array.pop())
        dx = relu_backward(dx, cache_array.pop())
        dx, dw3, db3 = conv_backward(dx, cache_array.pop())
        dx = max_pool_backward(dx, cache_array.pop())
        dx = relu_backward(dx, cache_array.pop())
        dx, dw2, db2 = conv_backward(dx, cache_array.pop())
        dx = max_pool_backward(dx, cache_array.pop())
        dx = relu_backward(dx, cache_array.pop())
        dx, dw1, db1 = conv_backward(dx, cache_array.pop())

        del cache_array
        del cache

        grads = {
                  'w1': dw1,
                  'b1': db1,
                  'w2': dw2,
                  'b2': db2,
                  'w3': dw3,
                  'b3': db3,
                  'w4': dw4,
                  'b4': db4,
                  'w5': dw5,
                  'b5': db5
                }

        self.grads = grads
        for name, grad in grads.items():
            self.grads[name] = grad.astype(np.float32)
            if 'w' in name:
                self.grads[name] -= np.float32(reg) * grad.astype(np.float32)

    def update_params(self,t):
        adam_momentum = self.adam_momentum
        adam_velocity = self.adam_velocity
        params = self.params
        grads = self.grads
        eps = 1e-8
        learning_rate = self.learning_rate
        beta1 = 0.9
        beta2 = 0.999

        for name, grad in grads.items():
            adam_momentum[name] = beta1 * adam_momentum.get(name,0) + (1 - beta1) * grad
            mt = adam_momentum[name]/(1-beta1**t)

            adam_velocity[name] = beta2 * adam_velocity.get(name, 0) + (1 - beta2) * (grad**2)
            vt = adam_velocity[name] / (1 - beta2 ** t)

            params[name] -= (learning_rate * mt / (np.sqrt(vt) + eps)).astype(np.float32)

        self.grads = {}

    def test(self,X,y,present=False,training_type=""):
        if present:
            self.params = self.best_params
        params = self.params
        N = X.shape[0]

        conv_param1 = {'pad': 0, 'stride': 1}
        conv_param2 = {'pad': 0, 'stride': 1}
        conv_param3 = {'pad': 0, 'stride': 2}
        pool_param = {'size': 2, 'stride': 2}


        out, cache = conv_forward(X, params['w1'], params['b1'], conv_param1)
        out, cache = relu(out)
        out, cache = max_pool_forward(out, pool_param)
        out, cache = conv_forward(out, params['w2'], params['b2'], conv_param2)
        out, cache = relu(out)
        out, cache = max_pool_forward(out, pool_param)
        out, cache = conv_forward(out, params['w3'], params['b3'], conv_param3)
        out, cache = relu(out)
        out, cache = flatten(out)
        out, cache = fc_layer_forward(out, params['w4'], params['b4'])
        scores, cache = fc_layer_forward(out, params['w5'], params['b5'])

        if present:
            self.params = self.best_params
            return np.argmax(scores,axis = 1)
        else:
            accuracy = np.mean(np.argmax(scores, axis=1) == y) * 100
            if accuracy > self.best_val_accuracy and "val" in training_type:
                self.best_val_accuracy = accuracy
                self.best_params = copy.deepcopy(params)
            print(training_type + " accuracy: %" + str(accuracy))
            return None