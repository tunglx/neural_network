import numpy as np

try:
    # PYPY hasn't got scipy
    from scipy.special import expit
except:
    expit = lambda x: 1.0 / (1 + np.exp(-x))


def softmax_function( signal, derivative=False ):
    # Calculate activation signal
    e_x = np.exp( signal - np.max(signal, axis=1, keepdims = True) )
    signal = e_x / np.sum( e_x, axis = 1, keepdims = True )
    
    if derivative:
        return np.ones( signal.shape )
    else:
        # Return the activation signal
        return signal
#end activation function


def sigmoid_function( signal, derivative=False ):
    # Prevent overflow.
    signal = np.clip( signal, -500, 500 )
    
    # Calculate activation signal
    signal = expit( signal )
    
    if derivative:
        # Return the partial derivation of the activation function
        return np.multiply(signal, 1 - signal)
    else:
        # Return the activation signal
        return signal
#end activation function

def ReLU_function( signal, derivative=False ):
    if derivative:
        return (signal > 0).astype(float)
    else:
        # Return the activation signal
        return np.maximum( 0, signal )
#end activation function



def tanh_function( signal, derivative=False ):
    # Calculate activation signal
    signal = np.tanh( signal )
    
    if derivative:
        # Return the partial derivation of the activation function
        return 1-np.power(signal,2)
    else:
        # Return the activation signal
        return signal
#end activation function


def linear_function( signal, derivative=False ):
    if derivative:
        # Return the partial derivation of the activation function
        return np.ones( signal.shape )
    else:
        # Return the activation signal
        return signal
#end activation function