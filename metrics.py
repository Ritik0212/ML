import numpy as np

def mean_squared_error(y, y_hat):
    m = len(y)
    return (np.sum((y - y_hat) ** 2)) / m

def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.append(y_true, y_pred)
        labels = np.unique(labels)

    label_num = len(labels)
    print(labels)

    output = np.zeros((label_num, label_num), dtype=int)

    for t, p in zip(y_true, y_pred):
        t_id = np.where(labels == t)
        p_id = np.where(labels == p)
        output[t_id, p_id] += 1

    return output


def precision_recall_fscore_support(y_true, y_pred, labels=None):
    if labels is None:
        labels = np.append(y_true, y_pred)
        labels = np.unique(labels)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # print(cm.shape)
    precision = np.diagonal(cm) / np.sum(cm, axis=0)

    recall = np.diagonal(cm) / np.sum(cm, axis=1)

    a = 2 * precision * recall
    b = precision + recall
    f_score = np.divide(a, b, out=np.zeros_like(a), where=(b != 0))

    support = np.zeros((len(cm)), dtype=int)

    for i in range(len(cm)):
        support[i] = np.sum(y_true == labels[i])

    return precision, recall, f_score, support


