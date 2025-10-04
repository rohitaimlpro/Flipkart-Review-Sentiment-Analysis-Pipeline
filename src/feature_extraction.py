import pandas as pd
import numpy as np
import yaml
import logging
import spacy
from pathlib import Path
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(self, config_path='params.yaml'):
        """Initialize feature extractor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.spacy_model = self.config['feature_extraction']['spacy_model']
        self.n_features = self.config['feature_extraction']['n_features']
        self.target_column = self.config['preprocessing']['target_column']
        
        logger.info(f"Loading spaCy model: {self.spacy_model}")
        self.nlp = spacy.load(self.spacy_model)
    
    def load_preprocessed_data(self, filepath):
        """Load preprocessed data"""
        logger.info(f"Loading preprocessed data from {filepath}")
        df = pd.read_json(filepath, lines=True)
        logger.info(f"Data loaded. Shape: {df.shape}")
        return df
    
    def vectorize_text(self, df):
        """Convert tokenized text to word vectors using spaCy"""
        logger.info("Starting vectorization with spaCy...")
        logger.info("This may take several minutes for large datasets...")
        
        def word_to_vector(word_list):
            """Convert list of words to average vector"""
            text = " ".join(word_list)
            doc = self.nlp(text)
            return doc.vector
        
        # Apply vectorization
        df['words'] = df['words'].apply(word_to_vector)
        
        logger.info("Vectorization completed!")
        return df
    
    def round_vectors(self, df):
        """Round vector values for consistency"""
        logger.info("Rounding vector values...")
        df['words'] = df['words'].apply(lambda x: [round(value, 3) for value in x])
        df['words'] = df['words'].apply(lambda x: np.array(x))
        return df
    
    def expand_vectors(self, df):
        """Expand vector column into multiple feature columns in batches"""
        logger.info("Expanding vectors into feature columns...")
        
        # Process in batches to avoid memory issues
        batch_size = 50000
        all_batches = []
        
        for i in range(0, len(df), batch_size):
            logger.info(f"Processing batch {i//batch_size + 1} (rows {i} to {min(i+batch_size, len(df))})")
            batch = df.iloc[i:i+batch_size].copy()
            
            # Convert vectors to DataFrame
            X_data = pd.DataFrame(batch['words'].tolist())
            
            # Concatenate with original dataframe
            batch_result = pd.concat([batch.reset_index(drop=True), X_data], axis=1)
            
            all_batches.append(batch_result)
            
            # Clear memory
            del X_data, batch
        
        # Combine all batches
        logger.info("Combining all batches...")
        result_df = pd.concat(all_batches, ignore_index=True)
        
        # Drop rows with NaN values
        initial_shape = result_df.shape[0]
        result_df = result_df.dropna()
        logger.info(f"Dropped {initial_shape - result_df.shape[0]} rows with NaN values")
        
        logger.info(f"Final shape after expansion: {result_df.shape}")
        return result_df
    
    def select_features(self, df, test_size=0.2, random_state=42):
        """Select top K features using mutual information"""
        logger.info(f"Selecting top {self.n_features} features...")
        
        from sklearn.model_selection import train_test_split
        
        # Separate features and target
        X = df.drop(['words', self.target_column], axis=1)
        y = df[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=self.n_features)
        selector.fit(X_train, y_train)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        logger.info(f"Selected feature indices: {selected_indices.tolist()}")
        
        return selected_indices.tolist()
    
    def save_vectorized_data(self, df, output_path):
        """Save vectorized data in chunks to avoid memory issues"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save in chunks
        chunk_size = 10000
        logger.info(f"Saving data in chunks of {chunk_size} rows...")
        
        mode = 'w'
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            chunk.to_json(output_path, orient='records', lines=True, mode=mode)
            mode = 'a'  # Append after first chunk
            if (i + chunk_size) % 50000 == 0:
                logger.info(f"  Saved {i + chunk_size} rows...")
        
        logger.info(f"Vectorized data saved to {output_path}")
    
    def feature_extraction_pipeline(self, input_path, output_path):
        """Run complete feature extraction pipeline"""
        logger.info("\n=== Starting Feature Extraction Pipeline ===")
        
        # Load preprocessed data
        df = self.load_preprocessed_data(input_path)
        
        # Vectorize text
        df = self.vectorize_text(df)
        
        # Save intermediate result
        intermediate_path = output_path.replace('final_data', 'vectorized_data')
        self.save_vectorized_data(df, intermediate_path)
        
        # Round and expand vectors
        df = self.round_vectors(df)
        df = self.expand_vectors(df)
        
        # Save final vectorized data
        self.save_vectorized_data(df, output_path)
        
        # Get selected features (for reference)
        selected_features = self.select_features(df)
        
        logger.info("\n=== Feature Extraction Complete ===")
        return df, selected_features


def main():
    """Main function to run feature extraction"""
    extractor = FeatureExtractor()
    
    # Get paths from config
    input_path = extractor.config['data']['processed_path'] + 'preprocessed_data.json'
    output_path = extractor.config['data']['final_file']
    
    # Run feature extraction
    df, selected_features = extractor.feature_extraction_pipeline(input_path, output_path)
    
    logger.info(f"\nSelected {len(selected_features)} features for modeling")
    return df, selected_features


if __name__ == "__main__":
    main()