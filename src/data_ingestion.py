import pandas as pd
import yaml
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataIngestion:
    def __init__(self, config_path='params.yaml'):
        """Initialize data ingestion with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_path = self.config['data']['raw_path']
        self.text_column = self.config['preprocessing']['text_column']
        self.target_column = self.config['preprocessing']['target_column']
    
    def load_data(self):
        """Load raw CSV data"""
        logger.info(f"Loading data from {self.raw_path}")
        
        try:
            df = pd.read_csv(self.raw_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, df):
        """Validate that required columns exist"""
        required_cols = [self.text_column, self.target_column]
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in dataset")
        
        logger.info(f"Data validation passed. Columns: {df.columns.tolist()}")
        return True
    
    def get_data_info(self, df):
        """Print data information"""
        logger.info("\n=== Data Information ===")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"\nNull values:\n{df.isnull().sum()}")
        logger.info(f"\nSentiment distribution:\n{df[self.target_column].value_counts()}")
        
        return df.info()


def main():
    """Main function to run data ingestion"""
    ingestion = DataIngestion()
    df = ingestion.load_data()
    ingestion.validate_data(df)
    ingestion.get_data_info(df)
    
    logger.info("Data ingestion completed successfully!")
    return df


if __name__ == "__main__":
    main()