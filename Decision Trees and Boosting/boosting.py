import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Part 2: Implementation of AdaBoost with decision trees as weak learners

class AdaBoost:
  def __init__(self, n_estimators=60, max_depth=10):
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.betas = []
    self.models = []
    
  def fit(self, X, y):
    ###########################TODO#############################################
    # In this part, please implement the adaboost fitting process based on the 
    # lecture and update self.betas and self.models, using decision trees with 
    # the given max_depth as weak learners

    # Inputs: X, y are the training examples and corresponding (binary) labels
    
    # Hint 1: remember to convert labels from {0,1} to {-1,1}
    # Hint 2: DecisionTreeClassifier supports fitting with a weighted training set
    
    y_orig = y.copy()  # Keep original {0, 1}
    y_boost = np.where(y == 0, -1, 1)  # Convert to {-1, +1} for math
    w = np.ones(X.shape[0]) / X.shape[0]
    
    for i in range(self.n_estimators):
      model = DecisionTreeClassifier(max_depth=self.max_depth)
      model.fit(X, y_orig, sample_weight=w)
      self.models.append(model)
      
      preds = model.predict(X)
      preds_boost = np.where(preds == 0, -1, 1)
      
      err = np.sum(w * (preds_boost != y_boost)) / np.sum(w)
      
      # Handle edge cases
      if err >= 0.5:
          err = 0.49  # Force error < 0.5
      err = np.clip(err, 1e-10, 1 - 1e-10)
      
      beta = 0.5 * np.log((1 - err) / err)  # Note the 0.5 factor
      self.betas.append(beta)
      
      w = w * np.exp(-beta * preds_boost * y_boost)
      w = w / np.sum(w)
        
    return self
    
  def predict(self, X):
    ###########################TODO#############################################
    # In this part, make prediction on X using the learned ensemble
    # Note that the prediction needs to be binary, that is, 0 or 1.
    preds = np.zeros(X.shape[0])
    for i, (beta, model) in enumerate(zip(self.betas, self.models)):
        pred = model.predict(X)
        pred_boost = np.where(pred == 0, -1, 1)
        preds += beta * pred_boost

    print(f"Final score range: [{preds.min():.4f}, {preds.max():.4f}]")
    preds = np.where(preds > 0, 1, 0)
    
    return preds
    
  def score(self, X, y):
    accuracy = accuracy_score(y, self.predict(X))
    return accuracy

