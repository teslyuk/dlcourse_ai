import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification
    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    TP = np.sum((prediction[i] == True and ground_truth[i] == True) for i in range(prediction.shape[0]))
    TN = np.sum((prediction[i] == False and ground_truth[i] == False) for i in range(prediction.shape[0]))
    FP = np.sum((prediction[i] == True and ground_truth[i] == False) for i in range(prediction.shape[0]))
    FN = np.sum((prediction[i] == False and ground_truth[i] == True) for i in range(prediction.shape[0]))
    
    if TP + FP != 0:
        precision = TP / (TP + FP) 
    if TP + FN != 0:
        recall = TP / (TP + FN)
    if TP + TN + FP + FN != 0:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
    if precision + recall != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification
    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    TP_or_TN = np.sum((prediction[i] == ground_truth[i]) for i in range(prediction.shape[0]))
    accuracy = TP_or_TN / len(prediction)
    return accuracy