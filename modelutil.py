import layers
import numpy as np
import copy

from layers import flatten, fc_layer_forward


class Model:
    def __init__(self,reg=0,learning_rate=1e-3, num_classes = 6,epochs=10,batch_size=150,hidden_dim=100):
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.initialization = 1e-2
        self.params = { 'w1': self.initialization * np.random.randn(12,3,7,7),
                        'b1': np.zeros(12),
                        'g1' : np.ones(12),
                        'beta1': np.zeros(12),
                        'w2' :self.initialization * np.random.randn(12,12,4,4),
                        'b2': np.zeros(12),
                        'g2': np.ones(12),
                        'beta2': np.zeros(12),
                        'w3' : self.initialization * np.random.randn(17*17*12,self.hidden_dim),
                        'b3': np.zeros(self.hidden_dim),
                        'w4': self.initialization * np.random.randn(self.hidden_dim, self.num_classes),
                        'b4': np.zeros(self.num_classes)
        }
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
                batch_size = min(self.batch_size, num_of_training)
                self.train_pass(X_shuffled[batch*self.batch_size:batch*self.batch_size + batch_size],y_shuffled[batch*self.batch_size:batch*self.batch_size+batch_size])
                num_of_training -= batch_size
                total_iter +=1
                self.update_params(total_iter)
                print("Current loss: " + str(self.loss))
            self.test(X_val,y_val,False,"val")
            self.test(X, y, False, "train")
        self.test(X_test, y_test, False, "test")

    def train_pass(self,X,y):
        params = self.params
        reg = self.reg

        cache_array = []
        conv_param1 = {'pad': 1, 'stride': 3}
        conv_param2 = {'pad': 0, 'stride': 2}
        pool_param = {'size': 2, 'stride': 2}
        norm_param1 = {'test': False, 'running_values':self.running_values, 'momentum': 0.9, 'eps': 1e-8}
        norm_param2 = {'test': False, 'running_values':self.running_values2, 'momentum': 0.9, 'eps': 1e-8}

        out, cache = layers.conv_forward(X, params['w1'], params['b1'], conv_param1)
        cache_array.append(cache)

        out, cache = layers.conv_batch_norm_forward(out, params['g1'], params['beta1'], norm_param1)
        cache_array.append(cache)

        out, cache = layers.relu(out)
        cache_array.append(cache)

        out, cache = layers.max_pool_forward(out, pool_param)
        cache_array.append(cache)

        out, cache = layers.conv_forward(out, params['w2'], params['b2'], conv_param2)
        cache_array.append(cache)

        out, cache = layers.conv_batch_norm_forward(out, params['g1'], params['beta1'], norm_param2)
        cache_array.append(cache)

        out, cache = layers.relu(out)
        cache_array.append(cache)

        out, cache = layers.flatten(out)
        cache_array.append(cache)

        out, cache = layers.fc_layer_forward(out, params['w3'], params['b3'])
        cache_array.append(cache)

        scores, cache = layers.fc_layer_forward(out, params['w4'], params['b4'])
        cache_array.append(cache)

        loss, dx = layers.softmax_loss(scores, y)

        #regularization

        for name, param in params.items():
            if 'w' in name:
                loss += reg * np.sum(param**2)/2
                self.loss = loss

        dx, dw4, db4 = layers.fc_layer_backward(dx, cache_array.pop())
        dx, dw3, db3 = layers.fc_layer_backward(dx, cache_array.pop())
        dx = layers.flatten_backward(dx, cache_array.pop())
        dx = layers.relu_backward(dx, cache_array.pop())
        dx, dg2, dbeta2 = layers.conv_batch_norm_backward(dx, cache_array.pop())
        dx, dw2, db2 = layers.conv_backward(dx, cache_array.pop())
        dx = layers.max_pool_backward(dx, cache_array.pop())
        dx = layers.relu_backward(dx, cache_array.pop())
        dx, dg1, dbeta1 = layers.conv_batch_norm_backward(dx, cache_array.pop())
        dx, dw1, db1 = layers.conv_backward(dx, cache_array.pop())

        grads = {'w1': dw1,
                      'b1': db1,
                      'g1': dg1,
                      'beta1': dbeta1,
                      'w2': dw2,
                      'b2': db2,
                      'g2': dg2,
                      'beta2': dbeta2,
                      'w3': dw3,
                      'b3': db3,
                      'w4': dw4,
                      'b4': db4
                      }
        self.grads = grads
        for name, grad in grads.items():
            if 'w' in name:
                self.grads[name] -= reg * grad

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

            params[name] -= learning_rate * mt / (np.sqrt(vt) + eps)

    def test(self,X,y,present=False,training_type=""):
        params = self.params
        N = X.shape[0]

        conv_param1 = {'pad': 1, 'stride': 3}
        conv_param2 = {'pad': 0, 'stride': 2}
        pool_param = {'size': 2, 'stride': 2}
        norm_param1 = {'test': True, 'running_values':self.running_values, 'momentum': 0.9, 'eps': 1e-8}
        norm_param2 = {'test': True, 'running_values': self.running_values2, 'momentum': 0.9, 'eps': 1e-8}


        out, cache = layers.conv_forward(X, params['w1'], params['b1'], conv_param1)
        out, cache = layers.conv_batch_norm_forward(out, params['g1'], params['beta1'], norm_param1)
        out, cache = layers.relu(out)
        out, cache = layers.max_pool_forward(out, pool_param)
        out, cache = layers.conv_forward(out, params['w2'], params['b2'], conv_param2)
        out, cache = layers.conv_batch_norm_forward(out, params['g1'], params['beta1'], norm_param2)
        out, cache = layers.relu(out)
        out, cache = layers.flatten(out)
        out, cache = layers.fc_layer_forward(out, params['w3'], params['b3'])
        scores, cache = layers.fc_layer_forward(out, params['w4'], params['b4'])

        if present:
            return np.argmax(scores,axis = 1)
        else:
            accuracy = np.mean(np.argmax(scores, axis=1) == y) * 100
            if accuracy > self.best_val_accuracy and "val" in training_type:
                self.best_val_accuracy = accuracy
                self.best_params = copy.deepcopy(params)
            print(training_type + " accuracy: %" + str(accuracy))
            return None