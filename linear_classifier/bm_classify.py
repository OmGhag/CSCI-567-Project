import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################

def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
	- w0: initial weight vector (a numpy array)
	- b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        for it in range(max_iterations):
            # Compute the predictions
            preds = X.dot(w) + b
            preds = np.where(preds >= 0, 1, 0)

            # Compute the errors
            errors = y - preds

            # Compute the gradients
            dw = np.dot(X.T, errors) / N
            db = np.sum(errors) / N

            # Update the weights and bias
            w -= step_size * dw
            b -= step_size * db

        
        
    elif loss == "logistic":
        for it in range(max_iterations):
            # Compute the predictions
            logits = X.dot(w) + b
            preds = 1 / (1 + np.exp(-logits))

            # Compute the errors
            errors = preds - y

            # Compute the gradients
            dw = np.dot(X.T, errors) / N
            db = np.sum(errors) / N

            # Update the weights and bias
            w -= step_size * dw
            b -= step_size * db

    else:
        raise "Undefined loss function."

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """
    value = 1 / (1 + np.exp(-z))
    return value
    


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    
    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape
    pred = sigmoid(X.dot(w) + b)
    pred = np.where(pred >= 0.5, 1, 0)
    
    assert pred.shape == (N,) 
    return pred


def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes
	
    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.
	
    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42) #DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    if gd_type == "sgd":

        for it in range(max_iterations):
            n = np.random.choice(N)
            # Compute the logits
            logits = X[n].dot(w.T) + b  # Shape (C,)
            # Compute the softmax probabilities
            exp_logits = np.exp(logits - np.max(logits))  # For numerical stability
            probs = exp_logits / np.sum(exp_logits)  # Shape (C,)
            # Create one-hot encoding for the true label
            y_one_hot = np.zeros(C)
            y_one_hot[y[n]] = 1
            # Compute the gradients
            errors = probs - y_one_hot  # Shape (C,)
            dw = np.outer(errors, X[n])  # Shape (C, D)
            db = errors  # Shape (C,)
            # Update the weights and bias
            w -= step_size * dw
            b -= step_size * db
            
    elif gd_type == "gd":
        for it in range(max_iterations):
            # Compute the logits
            logits = X.dot(w.T) + b  # Shape (N, C)
            # Compute the softmax probabilities
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # For numerical stability
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # Shape (N, C)
            # Create one-hot encoding for the true labels
            y_one_hot = np.zeros((N, C))
            y_one_hot[np.arange(N), y] = 1
            # Compute the gradients
            errors = probs - y_one_hot  # Shape (N, C)
            dw = errors.T.dot(X) / N  # Shape (C, D)
            db = np.sum(errors, axis=0) / N  # Shape (C,)
            # Update the weights and bias
            w -= step_size * dw
            b -= step_size * db
        
    else:
        raise "Undefined algorithm."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    logits = X.dot(w.T) + b  # Shape (N, C)
    preds = np.zeros((N,))
    preds = np.argmax(logits, axis=1)  # Shape (N,)
    
    
    assert preds.shape == (N,)
    return preds




        