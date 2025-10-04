import numpy as np
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class to evaluate model performance with comprehensive metrics"""
    
    def __init__(self):
        pass
    
    @staticmethod
    def safe_divide(numerator, denominator):
        """Safely divide arrays, handling division by zero"""
        return np.divide(
            numerator, 
            denominator, 
            out=np.zeros_like(numerator, dtype=float), 
            where=denominator != 0
        )
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate TP, FP, FN, TN for multiclass
        if cm.ndim == 1:
            TP = cm[0]
            FP = 0
            FN = 0
            TN = 0
        else:
            TP = np.diag(cm)
            FP = np.sum(cm, axis=0) - TP
            FN = np.sum(cm, axis=1) - TP
            TN = np.sum(cm) - (TP + FP + FN)
        
        # Calculate additional metrics
        specificity = self.safe_divide(TN, (TN + FP))
        specificity = np.mean(specificity)
        
        mcc = matthews_corrcoef(y_true, y_pred)
        
        npv = self.safe_divide(TN, (TN + FN))
        npv = np.mean(npv)
        
        fpr = self.safe_divide(FP, (FP + TN))
        fpr = np.mean(fpr)
        
        fnr = self.safe_divide(FN, (FN + TP))
        fnr = np.mean(fnr)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'mcc': mcc,
            'npv': npv,
            'fpr': fpr,
            'fnr': fnr
        }
        
        return metrics
    
    def print_metrics(self, metrics, model_name="Model"):
        """Print metrics in a formatted way"""
        logger.info(f"\n=== {model_name} Performance Metrics ===")
        logger.info(f"Accuracy:                    {metrics['accuracy']:.4f}")
        logger.info(f"Precision:                   {metrics['precision']:.4f}")
        logger.info(f"Recall (Sensitivity):        {metrics['recall']:.4f}")
        logger.info(f"F1-Score:                    {metrics['f1_score']:.4f}")
        logger.info(f"Specificity:                 {metrics['specificity']:.4f}")
        logger.info(f"Matthews Correlation Coef:   {metrics['mcc']:.4f}")
        logger.info(f"Negative Predictive Value:   {metrics['npv']:.4f}")
        logger.info(f"False Positive Rate:         {metrics['fpr']:.4f}")
        logger.info(f"False Negative Rate:         {metrics['fnr']:.4f}")


def main():
    """Example usage"""
    # This is just for demonstration
    evaluator = ModelEvaluator()
    
    # Example predictions
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1])
    
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    evaluator.print_metrics(metrics, "Example Model")


if __name__ == "__main__":
    main()