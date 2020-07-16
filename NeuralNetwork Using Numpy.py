#Here we build neural net from scratch using numpy

import numpy as np

np.random.seed(42)

#lets build a layer that can do both forward and backward pass

class Layer:
    def __init__(self):
        '''Initialize layer parameters'''
        pass

    def forward(self, input):
        '''Takes an input of shape [batch , input_size] and returns output data [batch, output_units]'''
        return input

    def backward(self, input, grad_output):
        '''Performs backpropagation step through the layer ,with respect to the given input'''

        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)

        return np.dot(grad_output, d_layer_d_input)

    def backward_with_regularization(self, input, grad_output, lambd):
        '''backpropagation with regularization parameter'''
        return self.backward(input, grad_output)

#Lets build Non Linearity Layer

class ReLU(Layer):
    def __init__(self):
        '''This layer applies elementwise rectified linear unit to all input'''
        pass
    def forward(self, input):
        return np.maximum(0, input)

    def forward_with_regularization(self, input, parameters):
        return self.forward(input)

    def backward(self, input, grad_output):
        '''Compute gradient of loss w.r.t. ReLU input'''
        relu_grad = input > 0
        return grad_output * relu_grad

    def backward_with_regularization(self, input, grad_output, lambd):
        return self.backward(input, grad_output)

#Lets test this
'''from util import eval_numerical_gradient
x = np.linspace(-1,1,10*32).reshape([10,32])
l = ReLU()
grads = l.backward(x,np.ones([10,32])/(32*10))
numeric_grads = eval_numerical_gradient(lambda x: l.forward(x).mean(), x=x)
assert np.allclose(grads, numeric_grads, rtol=1e-3, atol=0),\
    "gradient returned by your layer does not match the numerically computed gradient"'''


class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        """
         f(x) = <W*x> + b
        """
        self.learning_rate = learning_rate

        # initialize weights with small random numbers. We use normal initialization.

        self.weights = np.random.randn(input_units, output_units) * 0.01
        self.biases = np.zeros(output_units)

    def forward(self, input):
        """
        Perform an affine transformation:
        f(x) = <W*x> + b

        input shape: [batch, input_units]
        output shape: [batch, output units]
        """
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)  # <your code here>

        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output) / input.shape[0]  # <your code here>
        grad_biases = grad_output.mean(axis=0)  # <your code here>

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        # Here we perform a stochastic gradient descent step.
        # Later on, you can try replacing that with something better.
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases

        return grad_input
#testing dense layer
'''
l = Dense(128, 150)

assert -0.05 < l.weights.mean() < 0.05 and 1e-3 < l.weights.std() < 1e-1,\
    "The initial weights must have zero mean and small variance. "\
    "If you know what you're doing, remove this assertion."
assert -0.05 < l.biases.mean() < 0.05, "Biases must be zero mean. Ignore if you have a reason to do otherwise."

# To test the outputs, we explicitly set weights with fixed values. DO NOT DO THAT IN ACTUAL NETWORK!
l = Dense(3,4)

x = np.linspace(-1,1,2*3).reshape([2,3])
l.weights = np.linspace(-1,1,3*4).reshape([3,4])
l.biases = np.linspace(-1,1,4)

assert np.allclose(l.forward(x),np.array([[ 0.07272727,  0.41212121,  0.75151515,  1.09090909],
                                          [-0.90909091,  0.08484848,  1.07878788,  2.07272727]]))
print("Well done!")'''

#tesing gradients
'''def compute_out_given_wb(w,b):
    l = Dense(32,64,learning_rate=1)
    l.weights = np.array(w)
    l.biases = np.array(b)
    x = np.linspace(-1,1,10*32).reshape([10,32])
    return l.forward(x)
    
def compute_grad_by_params(w,b):
    l = Dense(32,64,learning_rate=1)
    l.weights = np.array(w)
    l.biases = np.array(b)
    x = np.linspace(-1,1,10*32).reshape([10,32])
    l.backward(x,np.ones([10,64]) / 10.)
    return w - l.weights, b - l.biases
    
w,b = np.random.randn(32,64), np.linspace(-1,1,64)

numeric_dw = eval_numerical_gradient(lambda w: compute_out_given_wb(w,b).mean(0).sum(),w )
numeric_db = eval_numerical_gradient(lambda b: compute_out_given_wb(w,b).mean(0).sum(),b )
grad_w,grad_b = compute_grad_by_params(w,b)

assert np.allclose(numeric_dw,grad_w,rtol=1e-3,atol=0), "weight gradient does not match numeric weight gradient"
assert np.allclose(numeric_db,grad_b,rtol=1e-3,atol=0), "weight gradient does not match numeric weight gradient"
print("Well done!")
Well done!'''


def softmax_crossentropy_with_logits(logits, reference_answers):
    """Compute crossentropy from logits[batch,n_classes] and ids of correct answers"""
    logits_for_answers = logits[np.arange(len(logits)), reference_answers]

    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits), axis=-1))

    return xentropy


def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    """Compute crossentropy gradient from logits[batch,n_classes] and ids of correct answers"""
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)), reference_answers] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (- ones_for_answers + softmax) / logits.shape[0]

