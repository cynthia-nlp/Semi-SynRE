from sklearn.metrics import precision_recall_fscore_support


def compute_PRF(predicted_idx, gold_idx, nota):
    return {
        "NoTA": precision_recall_fscore_support(y_true=gold_idx, y_pred=predicted_idx,
                                                labels=list(set(gold_idx) - {nota}), average='micro'),
        "micro": precision_recall_fscore_support(y_true=gold_idx, y_pred=predicted_idx,
                                                 labels=list(set(gold_idx)), average='micro'),
        "macro": precision_recall_fscore_support(y_true=gold_idx, y_pred=predicted_idx,
                                                 labels=list(set(gold_idx)), average='macro'),
    }


def evaluate(preds, true_label, nota):
    return compute_PRF(preds, true_label, nota)
