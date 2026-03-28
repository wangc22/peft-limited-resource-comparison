import evaluate
import numpy as np

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(
        predictions=preds,
        references=labels
    )["accuracy"]

    f1 = f1_metric.compute(
        predictions=preds,
        references=labels,
        average="binary"
    )["f1"]

    return {
        "accuracy": accuracy,
        "f1": f1,
    }