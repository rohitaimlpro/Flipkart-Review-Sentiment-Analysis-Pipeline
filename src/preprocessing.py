import pandas as pd
import numpy as np
import yaml
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextPreprocessor:
    def __init__(self, config_path='params.yaml'):
        """Initialize preprocessor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.text_column = self.config['preprocessing']['text_column']
        self.target_column = self.config['preprocessing']['target_column']
        self.sentiment_mapping = self.config['preprocessing']['sentiment_mapping']
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def load_data(self, filepath):
        """Load data from CSV"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        return df
    
    def handle_missing_values(self, df):
        """Drop rows with null values in Summary column"""
        logger.info(f"Null values before cleaning:\n{df.isnull().sum()}")
        
        initial_shape = df.shape[0]
        df = df.dropna(subset=[self.text_column])
        
        logger.info(f"Dropped {initial_shape - df.shape[0]} rows with null {self.text_column}")
        return df
    
    def tokenize_text(self, df):
        """Tokenize text into words"""
        logger.info("Tokenizing text...")
        
        # Strip whitespace
        df[self.text_column] = df[self.text_column].apply(lambda x: x.strip())
        
        # Tokenize
        df['words'] = df[self.text_column].apply(word_tokenize)
        
        logger.info(f"Tokenization completed. Sample: {df['words'].iloc[0][:5]}")
        return df
    
    def lowercase_words(self, df):
        """Convert all words to lowercase"""
        logger.info("Converting to lowercase...")
        df['words'] = [[word.lower() for word in sentence] for sentence in df['words']]
        return df
    
    def remove_stopwords(self, df):
        """Remove stopwords from tokenized text"""
        logger.info("Removing stopwords...")
        
        def filter_stopwords(sentence):
            return [word for word in sentence if word not in self.stop_words]
        
        df['words'] = df['words'].apply(filter_stopwords)
        logger.info(f"Stopwords removed. Sample: {df['words'].iloc[0][:5]}")
        return df
    
    def apply_stemming(self, df):
        """Apply stemming to reduce words to root form"""
        logger.info("Applying stemming...")
        
        def stem_words(sentence):
            return [self.stemmer.stem(word) for word in sentence]
        
        df['words'] = df['words'].apply(stem_words)
        logger.info(f"Stemming completed. Sample: {df['words'].iloc[0][:5]}")
        return df
    
    def encode_sentiment(self, df):
        """Encode sentiment labels to numeric values"""
        logger.info("Encoding sentiment labels...")
        logger.info(f"Mapping: {self.sentiment_mapping}")
        
        df[self.target_column] = df[self.target_column].map(self.sentiment_mapping)
        
        logger.info(f"Sentiment distribution after encoding:\n{df[self.target_column].value_counts()}")
        return df
    
    def preprocess_pipeline(self, df):
        """Run complete preprocessing pipeline"""
        logger.info("\n=== Starting Preprocessing Pipeline ===")
        
        df = self.handle_missing_values(df)
        df = self.tokenize_text(df)
        df = self.lowercase_words(df)
        df = self.remove_stopwords(df)
        df = self.apply_stemming(df)
        df = self.encode_sentiment(df)
        
        # Keep only necessary columns
        processed_df = df[['words', self.target_column]].copy()
        
        logger.info(f"\nPreprocessing completed! Final shape: {processed_df.shape}")
        return processed_df
    
    def save_preprocessed_data(self, df, output_path):
        """Save preprocessed data"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_json(output_path, orient='records', lines=True)
        logger.info(f"Preprocessed data saved to {output_path}")


def main():
    """Main function to run preprocessing"""
    preprocessor = TextPreprocessor()
    
    # Load raw data
    raw_path = preprocessor.config['data']['raw_path']
    df = preprocessor.load_data(raw_path)
    
    # Run preprocessing
    processed_df = preprocessor.preprocess_pipeline(df)
    
    # Save preprocessed data
    output_path = preprocessor.config['data']['processed_path'] + 'preprocessed_data.json'
    preprocessor.save_preprocessed_data(processed_df, output_path)
    
    logger.info("\n=== Preprocessing Complete ===")
    return processed_df


if __name__ == "__main__":
    main()