import numpy as np
from collections import Counter

def find_best_split(feature_vector, target_vector):
    if len(np.unique(feature_vector)) == 1:
        return None, None, None, None
    
    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    thresholds = (sorted_features[1:] + sorted_features[:-1]) / 2
    
    left_counts = np.cumsum(sorted_targets)
    left_sizes = np.arange(1, len(sorted_targets) + 1)

    p1_left = left_counts / left_sizes
    p0_left = 1 - p1_left
    H_left = 1 - p1_left**2 - p0_left**2

    right_counts = left_counts[-1] - left_counts
    right_sizes = left_sizes[-1] - left_sizes

    valid = right_sizes > 0

    p1_right = np.zeros_like(right_counts, dtype=float)
    p0_right = np.zeros_like(right_counts, dtype=float)
    H_right = np.zeros_like(right_counts, dtype=float)
    
    p1_right[valid] = right_counts[valid] / right_sizes[valid]
    p0_right[valid] = 1 - p1_right[valid]
    H_right[valid] = 1 - p1_right[valid]**2 - p0_right[valid]**2

    ginis = -(left_sizes/len(target_vector)) * H_left - (right_sizes/len(target_vector)) * H_right

    valid_indices = (left_sizes[:-1] > 0) & (right_sizes[:-1] > 0)
    if not np.any(valid_indices):
        return None, None, None, None

    valid_ginis = ginis[:-1][valid_indices]
    valid_thresholds = thresholds[valid_indices]

    if len(valid_ginis) == 0:
        return None, None, None, None
    
    best_idx = np.argmax(valid_ginis)
    threshold_best = valid_thresholds[best_idx]
    gini_best = valid_ginis[best_idx]
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if len(np.unique(sub_y)) == 1:
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                unique_values = np.unique(sub_X[:, feature])
                value_to_index = {val: idx for idx, val in enumerate(unique_values)}

                ratios = {}
                for val in unique_values:
                    mask = (sub_X[:, feature] == val)
                    if np.sum(mask) > 0:
                        ratios[val] = np.mean(sub_y[mask])
                    else:
                        ratios[val] = 0.0
                
                sorted_categories = sorted(unique_values, key=lambda x: ratios[x])
                categories_map = {cat: i for i, cat in enumerate(sorted_categories)}
                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])
            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            if len(np.unique(feature_vector)) < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if threshold is None:
                continue
            
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = [cat for cat in categories_map if categories_map[cat] < threshold]
                else:
                    raise ValueError
        
        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        
        node["left_child"] = {}
        node["right_child"] = {}

        left_samples = np.sum(split)
        right_samples = len(split) - left_samples
        
        if (self._min_samples_leaf is not None and 
            (left_samples < self._min_samples_leaf or 
             right_samples < self._min_samples_leaf)):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        if self._feature_types[feature_best] == "real":
            split_mask = sub_X[:, feature_best] < threshold_best
        else:
            split_mask = np.isin(sub_X[:, feature_best], threshold_best)
        
        if np.sum(split_mask) == 0 or np.sum(~split_mask) == 0:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        self._fit_node(sub_X[split_mask], sub_y[split_mask], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split_mask], sub_y[~split_mask], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            if x[feature] in node.get("categories_split", []):
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])

    def get_tree_depth(self):
        return self._get_tree_depth(self._tree)

    def _get_tree_depth(self, node):
        if node["type"] == "terminal":
            return 1
        return 1 + max(self._get_tree_depth(node["left_child"]), self._get_tree_depth(node["right_child"]))
