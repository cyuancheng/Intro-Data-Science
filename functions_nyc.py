# statistic model
import numpy as np
import pandas as pd

def normalize_features(array):
    """
    Normalize features (substract mean and divide by starndar deviation).
    """
    mu = array.mean()
    sigma = array.std()
    array_normalized = (array - mu)/sigma
    return array_normalized, mu, sigma

def compute_cost(features, values, theta):
    import numpy as np
    m = len(values)
    sum_of_square_errors = np.square(np.dot(features, theta) - values).sum()
    cost = sum_of_square_errors / (2*m)
    return cost

def gradient_descent(features, values, theta, alpha, num_iterations):
    m = len(values)
    cost_history = []
    for i in range(0,num_iterations):
        theta = theta - (alpha/m)*np.dot((np.dot(features,theta)-values),features)
        cost = compute_cost(features, values, theta)
        cost_history.append(cost)
    return theta, pd.Series(cost_history)

#prediction function for GD
def predictions_GD(features, values, alpha, num_iterations):
    m = len(values)
    theta = np.zeros(features.shape[1])
    theta, cost_history = gradient_descent(features, values, theta, alpha, num_iterations)
    pred_GD = features.dot(theta) # hypothesis function
    pred_GD[predictions_GD<0] = 0 # Predictions should be greater than 0
    return theta, pd.Series(pred_GD)


def getPredictions(features, intercept, params):
    """
    Calculate the predictions using the feature values and the model parameters (including intercept).
    """
    import numpy as np
    predictions = intercept + np.dot(features.values, params)
    return predictions

def compute_r_squared(data, predictions):
    """
    Computes the R squared using the real data and the predicted data.
    """
    r_squared = 1 - (np.sum((data - predictions) ** 2) / np.sum((data - np.mean(data)) ** 2))
    return r_squared
