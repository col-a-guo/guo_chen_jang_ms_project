import numpy as np
import scipy.stats as st

def calculate_prediction_counts(report):
    """
    Calculates the number of predicted samples for each class based on a classification report.

    Args:
        report (dict): A dictionary representation of the classification report, where
                       'precision', 'recall', and 'support' are keys containing lists
                       of respective values for each class.

    Returns:
        dict: A dictionary where keys are class labels (e.g., '0.0', '1.0') and
              values are the number of predicted samples for each class.
    """
    predicted_counts = {}
    for class_label in report.keys():
        if class_label in ['accuracy', 'macro avg', 'weighted avg']:
            continue

        precision = report[class_label]['precision']
        recall = report[class_label]['recall']
        support = report[class_label]['support']


        tp = recall * support  # True Positives
        if precision > 0:
            fp = (tp * (1 - precision)) / precision  # False Positives
            predicted_count = tp + fp
        else:
            predicted_count = 0  # Handle the case where precision is zero

        predicted_counts[class_label] = int(round(predicted_count))  # Round to nearest integer

    return predicted_counts

classification_report_data = {
    '0.0': {'precision': 0.6798, 'recall': 0.8214, 'f1-score': 0.7439, 'support': 504},
    '1.0': {'precision': 0.4512, 'recall': 0.2751, 'f1-score': 0.3418, 'support': 269},
    'accuracy': {'precision': None, 'recall': None, 'f1-score': 0.6313, 'support': 773},
    'macro avg': {'precision': 0.5655, 'recall': 0.5483, 'f1-score': 0.5429, 'support': 773},
    'weighted avg': {'precision': 0.6003, 'recall': 0.6313, 'f1-score': 0.6040, 'support': 773}
}

predicted_counts = calculate_prediction_counts(classification_report_data)
print(predicted_counts)


def calculate_confidence_interval(precision, recall, predicted_count, support, confidence_level=0.95):
    """Calculates confidence intervals for precision and recall.

    Args:
        precision (float): The precision score.
        recall (float): The recall score.
        predicted_count (int): Number of predicted samples for the class
        support (int): The number of actual samples for the class (support).
        confidence_level (float): The desired confidence level (e.g., 0.95 for 95%).

    Returns:
        tuple: A tuple containing confidence intervals for precision and recall.
               Each interval is a tuple (lower_bound, upper_bound).  Returns
               (None, None) if predicted_count or support is zero.
    """
    if predicted_count == 0 or support == 0:
        return (None, None), (None, None)


    # Confidence interval for precision
    precision_interval = st.norm.interval(confidence_level, loc=precision, scale=np.sqrt((precision * (1 - precision)) / predicted_count))


    # Confidence interval for recall
    recall_interval = st.norm.interval(confidence_level, loc=recall, scale=np.sqrt((recall * (1 - recall)) / support))

    return precision_interval, recall_interval

# Example usage:
for class_label in ['0.0', '1.0']:
    precision = classification_report_data[class_label]['precision']
    recall = classification_report_data[class_label]['recall']
    support = classification_report_data[class_label]['support']
    predicted_count = predicted_counts[class_label]

    precision_interval, recall_interval = calculate_confidence_interval(precision, recall, predicted_count, support)

    print(f"Class {class_label}:")
    print(f"  Precision: {precision:.4f}  Confidence Interval: {precision_interval}")
    print(f"  Recall:    {recall:.4f}  Confidence Interval: {recall_interval}")