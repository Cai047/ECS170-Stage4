"""Binary positive-class metrics and macro-F1."""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from local_code.base_class.evaluate import evaluate

def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
    macro = precision_recall_fscore_support(y, pred, average='macro', zero_division=0)[2]
    return dict(accuracy=float(accuracy_score(y, pred)), precision=float(precision), recall=float(recall),
                f1=float(f1), macro_f1=float(macro), confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())

class Evaluate_Metrics(evaluate):
    def evaluate(self):
        # The experiment has evaluated each frozen checkpoint on the official test set.
        keys = ['accuracy', 'precision', 'recall', 'f1', 'macro_f1']
        return {row['model']: {key: row[key] for key in keys} for row in self.data}
