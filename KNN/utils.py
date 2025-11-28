import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    f1_score = 0.0
    tp = sum(1 for r, p in zip(real_labels, predicted_labels) if r == 1 and p == 1)
    fp = sum(1 for r, p in zip(real_labels, predicted_labels) if r == 0 and p == 1)
    fn = sum(1 for r, p in zip(real_labels, predicted_labels) if r == 1 and p == 0)
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        result = 0
        p = 3
        for i in range(len(point1)):
            result += abs(point1[i] - point2[i]) ** p
        return result ** (1/p)

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        result = 0
        for i in range(len(point1)):
            result += (point1[i] - point2[i]) ** 2
        return result ** 0.5

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        dot_product = sum(a * b for a, b in zip(point1, point2))
        magnitude1 = sum(a ** 2 for a in point1) ** 0.5
        magnitude2 = sum(b ** 2 for b in point2) ** 0.5
        if magnitude1 == 0 or magnitude2 == 0:
            return 1.0  # Cosine similarity is not defined for zero vectors
        cosine_similarity = dot_product / (magnitude1 * magnitude2)
        return 1 - cosine_similarity  # Convert similarity to distance



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[float]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[float]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        for distance_func_name, distance_func in distance_funcs.items():
            for k in range(1,30,2):
                model = KNN(k, distance_func)
                model.train(x_train, y_train)
                y_pred = model.predict(x_val)
                current_f1 = f1_score(y_val, y_pred)
                if self.best_model is None:
                    self.best_model = model
                    self.best_k = k
                    self.best_distance_function = distance_func_name
                    best_f1 = current_f1
                else:
                    if current_f1 > best_f1:
                        self.best_model = model
                        self.best_k = k
                        self.best_distance_function = distance_func_name
                        best_f1 = current_f1
                    elif current_f1 == best_f1:
                        if list(distance_funcs.keys()).index(distance_func_name) < list(distance_funcs.keys()).index(self.best_distance_function):
                            self.best_model = model
                            self.best_k = k
                            self.best_distance_function = distance_func_name
                            best_f1 = current_f1
                        elif list(distance_funcs.keys()).index(distance_func_name) == list(distance_funcs.keys()).index(self.best_distance_function):
                            if k < self.best_k:
                                self.best_model = model
                                self.best_k = k
                                self.best_distance_function = distance_func_name
                                best_f1 = current_f1
        

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[float]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[float]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        
        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        for scaler_name, scaler_class in scaling_classes.items():
            scaler = scaler_class()
            x_train_scaled = scaler(x_train)
            x_val_scaled = scaler(x_val)
            for distance_func_name, distance_func in distance_funcs.items():
                for k in range(1,30,2):
                    model = KNN(k, distance_func)
                    model.train(x_train_scaled, y_train)
                    y_pred = model.predict(x_val_scaled)
                    current_f1 = f1_score(y_val, y_pred)
                    if self.best_model is None:
                        self.best_model = model
                        self.best_k = k
                        self.best_distance_function = distance_func_name
                        self.best_scaler = scaler_name
                        best_f1 = current_f1
                    else:
                        if current_f1 > best_f1:
                            self.best_model = model
                            self.best_k = k
                            self.best_distance_function = distance_func_name
                            self.best_scaler = scaler_name
                            best_f1 = current_f1
                        elif current_f1 == best_f1:
                            if list(scaling_classes.keys()).index(scaler_name) < list(scaling_classes.keys()).index(self.best_scaler):
                                self.best_model = model
                                self.best_k = k
                                self.best_distance_function = distance_func_name
                                self.best_scaler = scaler_name
                                best_f1 = current_f1
                            elif list(scaling_classes.keys()).index(scaler_name) == list(scaling_classes.keys()).index(self.best_scaler):
                                if list(distance_funcs.keys()).index(distance_func_name) < list(distance_funcs.keys()).index(self.best_distance_function):
                                    self.best_model = model
                                    self.best_k = k
                                    self.best_distance_function = distance_func_name
                                    self.best_scaler = scaler_name
                                    best_f1 = current_f1
                                elif list(distance_funcs.keys()).index(distance_func_name) == list(distance_funcs.keys()).index(self.best_distance_function):
                                    if k < self.best_k:
                                        self.best_model = model
                                        self.best_k = k
                                        self.best_distance_function = distance_func_name
                                        self.best_scaler = scaler_name
                                        best_f1 = current_f1


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        self.n_samples = len(features)
        self.n_features = len(features[0]) if self.n_samples > 0 else 0
        normalized_features = []
        for feature in features:
            magnitude = sum(x ** 2 for x in feature) ** 0.5
            if magnitude == 0:
                normalized_features.append([0.0] * self.n_features)
            else:
                normalized_features.append([x / magnitude for x in feature])
        return normalized_features


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        n_samples = len(features)
        n_features = len(features[0]) if n_samples > 0 else 0
        if n_samples == 0 or n_features == 0:
            return features
        features_np = np.array(features)
        min_vals = features_np.min(axis=0)
        max_vals = features_np.max(axis=0)
        ranges = max_vals - min_vals
        scaled_features = np.zeros_like(features_np)
        for j in range(n_features):
            if ranges[j] == 0:
                scaled_features[:, j] = 0.0
            else:
                scaled_features[:, j] = (features_np[:, j] - min_vals[j]) / ranges[j]
        return scaled_features.tolist()