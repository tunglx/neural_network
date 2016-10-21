import numpy as np
import math

def sum_squared_error( outputs, targets, derivative=False ):
    if derivative:
        return outputs - targets 
    else:
        return 0.5 * np.power(outputs - targets,2)
#end cost function