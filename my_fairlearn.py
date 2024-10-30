import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def demographic_parity_difference_mine(y_true, y_pred, *, sensitive_features, method="between_groups", sample_weight=None) -> float:

    # Ensure all inputs are arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_features = np.asarray(sensitive_features)

    # Combine y_true, y_pred, and sensitive_features into a DataFrame
    data = {'y_true': y_true, 'y_pred': y_pred, 'sensitive_features': sensitive_features}
    df = pd.DataFrame(data)

    # Compute confusion matrix for each sensitive group
    confusion_matrices = {}
    for group_value in df['sensitive_features'].unique():
        group_data = df[df['sensitive_features'] == group_value]
        confusion_matrices[group_value] = confusion_matrix(group_data['y_true'], group_data['y_pred'], sample_weight=sample_weight)

    # Calculate demographic parity difference based on the selected method
    if method == "between_groups":
        #selection_rates = [confusion_matrix.sum(axis=0)[1] / confusion_matrix.sum() for confusion_matrix in confusion_matrices.values()]
        selection_rates = [confusion_matrix.sum(axis=0)[1] / confusion_matrix.sum()  for confusion_matrix in confusion_matrices.values() if confusion_matrix.shape[0] > 1]
        result = max(selection_rates) - min(selection_rates)
    else:
        # Handle other methods if needed
        result = 0.0

    return result

def demographic_parity_ratio_mine(y_true, y_pred, *, sensitive_features, method="between_groups", sample_weight=None) -> float:

    # Ensure all inputs are arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_features = np.asarray(sensitive_features)

    # Combine y_true, y_pred, and sensitive_features into a DataFrame
    data = {'y_true': y_true, 'y_pred': y_pred, 'sensitive_features': sensitive_features}
    df = pd.DataFrame(data)

    # Compute confusion matrix for each sensitive group
    confusion_matrices = {}
    for group_value in df['sensitive_features'].unique():
        group_data = df[df['sensitive_features'] == group_value]
        confusion_matrices[group_value] = confusion_matrix(group_data['y_true'], group_data['y_pred'], sample_weight=sample_weight)

    # Calculate demographic parity difference based on the selected method
    if method == "between_groups":
        #selection_rates = [confusion_matrix.sum(axis=0)[1] / confusion_matrix.sum() for confusion_matrix in confusion_matrices.values()]
        selection_rates = [confusion_matrix.sum(axis=0)[1] / confusion_matrix.sum()  for confusion_matrix in confusion_matrices.values() if confusion_matrix.shape[0] > 1]
        result = min(selection_rates) / max(selection_rates)
    else:
        # Handle other methods if needed
        result = 0.0

    return result

def true_positive_rate_difference(y_true, y_pred, sensitive_features, sample_weight=None):
    # Calculate true positive rate
    tp_rates = []
    for group_value in np.unique(sensitive_features):
        group_indices = (sensitive_features == group_value)
        tp = np.sum((y_true[group_indices] == 1) & (y_pred[group_indices] == 1), axis=0)
        tpr = tp / np.sum((y_true[group_indices] == 1), axis=0)
        tp_rates.append(tpr)

    # Return the difference between the largest and smallest true positive rates
    return np.max(tp_rates) - np.min(tp_rates)

def false_positive_rate_difference(y_true, y_pred, sensitive_features, sample_weight=None):
    # Calculate false positive rate
    fp_rates = []
    for group_value in np.unique(sensitive_features):
        group_indices = (sensitive_features == group_value)
        fp = np.sum((y_true[group_indices] == 0) & (y_pred[group_indices] == 1), axis=0)
        fpr = fp / np.sum((y_true[group_indices] == 0), axis=0)
        fp_rates.append(fpr)

    # Return the difference between the largest and smallest false positive rates
    return np.max(fp_rates) - np.min(fp_rates)

def equalized_odds_difference_mine(y_true, y_pred, *, sensitive_features, method="between_groups", sample_weight=None) -> float:
    # Ensure all inputs are arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_features = np.asarray(sensitive_features)

    # Calculate true positive rate difference and false positive rate difference
    tpr_diff = true_positive_rate_difference(y_true, y_pred, sensitive_features, sample_weight)
    fpr_diff = false_positive_rate_difference(y_true, y_pred, sensitive_features, sample_weight)

    # Return the maximum difference between TPRs and FPRs
    return max(tpr_diff, fpr_diff)

def true_positive_rate_ratio(y_true, y_pred, sensitive_features, sample_weight=None):
    # Calculate true positive rate
    tp_rates = []
    for group_value in np.unique(sensitive_features):
        group_indices = (sensitive_features == group_value)
        tp = np.sum((y_true[group_indices] == 1) & (y_pred[group_indices] == 1), axis=0)
        tpr = tp / np.sum((y_true[group_indices] == 1), axis=0)
        tp_rates.append(tpr)

    # Return the ratio between the smallest and largest true positive rates
    min_tpr = np.min(tp_rates)
    max_tpr = np.max(tp_rates)

    # Handle potential division by zero
    if max_tpr == 0:
        return np.nan
    return min_tpr / max_tpr

def false_positive_rate_ratio(y_true, y_pred, sensitive_features, sample_weight=None):
    # Calculate false positive rate
    fp_rates = []
    for group_value in np.unique(sensitive_features):
        group_indices = (sensitive_features == group_value)
        fp = np.sum((y_true[group_indices] == 0) & (y_pred[group_indices] == 1), axis=0)
        fpr = fp / np.sum((y_true[group_indices] == 0), axis=0)
        fp_rates.append(fpr)

    # Return the ratio between the smallest and largest false positive rates
    min_fpr = np.min(fp_rates)
    max_fpr = np.max(fp_rates)

    # Handle potential division by zero
    if max_fpr == 0:
        return np.nan
    return min_fpr / max_fpr

def equalized_odds_ratio_mine(y_true, y_pred, *, sensitive_features, method="between_groups", sample_weight=None) -> float:
    """
    Calculate the equalized odds ratio.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) labels.

    y_pred : array-like
        Predicted labels h(X) returned by the classifier.

    sensitive_features : array-like
        The sensitive features over which demographic parity should be assessed.

    method : str
        How to compute the differences. See fairlearn.metrics.MetricFrame.difference
        for details.

    sample_weight : array-like
        The sample weights.

    Returns
    -------
    float
        The equalized odds ratio
    """

    # Ensure all inputs are arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_features = np.asarray(sensitive_features)

    # Calculate true positive rate ratio and false positive rate ratio
    tpr_ratio = true_positive_rate_ratio(y_true, y_pred, sensitive_features, sample_weight)
    fpr_ratio = false_positive_rate_ratio(y_true, y_pred, sensitive_features, sample_weight)
    #print("tpr_ratio:", tpr_ratio)
    #print("fpr_ratio:", fpr_ratio)

    # Return the smaller of the two ratios
    return min(tpr_ratio, fpr_ratio)



from itertools import combinations

def true_positive_rate(conf_matrix):
    """
    Calculate the true positive rate (sensitivity) from a confusion matrix.
    """
    return conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])

def equalized_odds_difference_multi_groups(y_true, y_pred, sf_data):
    """
    Calculate the aggregated Equalized Odds Difference (EOD) across multiple protected groups.
    """
    unique_groups = np.unique(sf_data)
    eod_values = []

    # Calculate EOD for each pair of protected groups
    for group1, group2 in combinations(unique_groups, 2):
        indices_group1 = np.where(sf_data == group1)[0]
        indices_group2 = np.where(sf_data == group2)[0]

        conf_matrix_group1 = confusion_matrix(np.array(y_true)[indices_group1], np.array(y_pred)[indices_group1])
        conf_matrix_group2 = confusion_matrix(np.array(y_true)[indices_group2], np.array(y_pred)[indices_group2])

        tpr_group1 = true_positive_rate(conf_matrix_group1)
        tpr_group2 = true_positive_rate(conf_matrix_group2)

        eod_values.append(abs(tpr_group1 - tpr_group2))

    # Aggregate EOD values
    aggregated_eod = np.mean(eod_values)

    return aggregated_eod
