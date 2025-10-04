import pandas as pd
import numpy as np
import yaml
import logging
import mlflow
import mlflow.sklearn
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Add current directory to path for imports
sys.path.insert(0, '.')
from evaluate import ModelEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path='params.yaml'):
        """Initialize model trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.target_column = self.config['preprocessing']['target_column']
        self.test_size = self.config['training']['test_size']
        self.random_state = self.config['training']['random_state']
        self.n_features = self.config['feature_extraction']['n_features']
        
        # MLflow configuration
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
    
    def load_data(self, filepath):
        """Load vectorized data in chunks to avoid memory issues"""
        logger.info(f"Loading data from {filepath}")
        
        # Read in chunks
        chunk_size = 50000
        chunks = []
        
        logger.info("Reading data in chunks...")
        for i, chunk in enumerate(pd.read_json(filepath, lines=True, chunksize=chunk_size)):
            chunks.append(chunk)
            logger.info(f"  Chunk {i+1}: Loaded {len(chunk)} rows")
            import sys
            sys.stdout.flush()  # Force output to display
        
        logger.info("Combining chunks...")
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Data loaded successfully. Final shape: {df.shape}")
        return df
    
    def prepare_data(self, df):
        """Prepare train and test sets with feature selection"""
        logger.info("Preparing train/test split...")
        
        # Separate features and target
        X = df.drop(['words', self.target_column], axis=1)
        y = df[self.target_column]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
        
        # Feature selection
        logger.info(f"Selecting top {self.n_features} features...")
        selector = SelectKBest(score_func=mutual_info_classif, k=self.n_features)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        selected_indices = selector.get_support(indices=True)
        logger.info(f"Selected features: {selected_indices.tolist()}")
        
        return X_train_selected, X_test_selected, y_train, y_test, selector
    
    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Train Logistic Regression model"""
        logger.info("\n=== Training Logistic Regression ===")
        
        with mlflow.start_run(run_name="Logistic_Regression"):
            # Get hyperparameters
            params = self.config['training']['logistic_regression']
            
            # Train model
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            evaluator = ModelEvaluator()
            metrics = evaluator.calculate_metrics(y_test, y_pred)
            
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Logistic Regression Accuracy: {metrics['accuracy']:.4f}")
            
            return model, metrics
    
    def train_decision_tree(self, X_train, X_test, y_train, y_test):
        """Train Decision Tree model"""
        logger.info("\n=== Training Decision Tree ===")
        
        with mlflow.start_run(run_name="Decision_Tree"):
            # Get hyperparameters
            params = self.config['training']['decision_tree']
            
            # Train model
            model = DecisionTreeClassifier(**params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            evaluator = ModelEvaluator()
            metrics = evaluator.calculate_metrics(y_test, y_pred)
            
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Decision Tree Accuracy: {metrics['accuracy']:.4f}")
            
            return model, metrics
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest model"""
        logger.info("\n=== Training Random Forest ===")
        
        with mlflow.start_run(run_name="Random_Forest"):
            # Get hyperparameters
            params = self.config['training']['random_forest']
            
            # Train model
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            evaluator = ModelEvaluator()
            metrics = evaluator.calculate_metrics(y_test, y_pred)
            
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"Random Forest Accuracy: {metrics['accuracy']:.4f}")
            
            return model, metrics
    
    def save_best_model(self, models_dict):
        """Save the best performing model"""
        # Find best model based on accuracy
        best_model_name = max(models_dict, key=lambda k: models_dict[k]['metrics']['accuracy'])
        best_model = models_dict[best_model_name]['model']
        best_metrics = models_dict[best_model_name]['metrics']
        
        logger.info(f"\n=== Best Model: {best_model_name} ===")
        logger.info(f"Accuracy: {best_metrics['accuracy']:.4f}")
        
        # Save model
        model_path = Path('models')
        model_path.mkdir(exist_ok=True)
        
        model_file = model_path / 'best_model.pkl'
        joblib.dump(best_model, model_file)
        logger.info(f"Best model saved to {model_file}")
        
        return best_model_name, best_model
    
    def train_all_models(self):
        """Train all models and compare"""
        logger.info("\n=== Starting Model Training Pipeline ===")
        
        # Load data
        data_path = self.config['data']['final_file']
        df = self.load_data(data_path)
        
        # Prepare data
        X_train, X_test, y_train, y_test, selector = self.prepare_data(df)
        
        # Train models
        models_dict = {}
        
        lr_model, lr_metrics = self.train_logistic_regression(X_train, X_test, y_train, y_test)
        models_dict['Logistic_Regression'] = {'model': lr_model, 'metrics': lr_metrics}
        
        dt_model, dt_metrics = self.train_decision_tree(X_train, X_test, y_train, y_test)
        models_dict['Decision_Tree'] = {'model': dt_model, 'metrics': dt_metrics}
        
        rf_model, rf_metrics = self.train_random_forest(X_train, X_test, y_train, y_test)
        models_dict['Random_Forest'] = {'model': rf_model, 'metrics': rf_metrics}
        
        # Save best model
        best_model_name, best_model = self.save_best_model(models_dict)
        
        # Save feature selector
        selector_path = Path('models') / 'feature_selector.pkl'
        joblib.dump(selector, selector_path)
        logger.info(f"Feature selector saved to {selector_path}")
        
        logger.info("\n=== Training Complete ===")
        return models_dict, best_model_name


def main():
    """Main function to run training"""
    trainer = ModelTrainer()
    models_dict, best_model_name = trainer.train_all_models()
    
    # Print summary
    logger.info("\n=== Model Performance Summary ===")
    for name, data in models_dict.items():
        logger.info(f"{name}: Accuracy = {data['metrics']['accuracy']:.4f}")
    
    return models_dict


if __name__ == "__main__":
    main()