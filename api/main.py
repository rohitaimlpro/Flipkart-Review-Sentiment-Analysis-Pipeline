from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import spacy
import yaml
import logging
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Flipkart Sentiment Analysis API",
    description="API for predicting sentiment of product reviews",
    version="1.0.0"
)

# Global variables for model and preprocessors
model = None
feature_selector = None
nlp = None
stemmer = None
stop_words = None
sentiment_labels = {0: "negative", 1: "neutral", 2: "positive"}


class ReviewInput(BaseModel):
    review_text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "review_text": "This product is amazing! Great quality and fast delivery."
            }
        }


class PredictionOutput(BaseModel):
    review_text: str
    sentiment: str
    sentiment_code: int
    confidence: float


@app.on_event("startup")
async def load_models():
    """Load models and preprocessors on startup"""
    global model, feature_selector, nlp, stemmer, stop_words
    
    try:
        logger.info("Loading models and preprocessors...")
        
        # Load configuration
        with open('params.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Load trained model
        model_path = Path(config['api']['model_path'])
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load feature selector
        selector_path = Path('models/feature_selector.pkl')
        if selector_path.exists():
            feature_selector = joblib.load(selector_path)
            logger.info("Feature selector loaded")
        
        # Load spaCy model
        spacy_model = config['feature_extraction']['spacy_model']
        nlp = spacy.load(spacy_model)
        logger.info(f"SpaCy model '{spacy_model}' loaded")
        
        # Initialize NLTK components
        stop_words = set(stopwords.words('english'))
        stemmer = PorterStemmer()
        logger.info("NLTK components initialized")
        
        logger.info("All models and preprocessors loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


def preprocess_text(text: str) -> list:
    """Preprocess text following the training pipeline"""
    # Strip whitespace
    text = text.strip()
    
    # Tokenize
    words = word_tokenize(text.lower())
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    # Apply stemming
    words = [stemmer.stem(word) for word in words]
    
    return words


def vectorize_text(words: list) -> np.ndarray:
    """Convert words to vector using spaCy"""
    text = " ".join(words)
    doc = nlp(text)
    vector = doc.vector
    
    # Round values
    vector = np.array([round(v, 3) for v in vector])
    
    return vector


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Flipkart Sentiment Analysis API",
        "status": "active",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "nlp_loaded": nlp is not None
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict_sentiment(review: ReviewInput):
    """
    Predict sentiment for a given review text
    
    Returns:
    - review_text: Original review text
    - sentiment: Predicted sentiment (positive/negative/neutral)
    - sentiment_code: Numeric code (0=negative, 1=neutral, 2=positive)
    - confidence: Prediction confidence score
    """
    try:
        # Validate input
        if not review.review_text or len(review.review_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Review text cannot be empty")
        
        # Preprocess text
        words = preprocess_text(review.review_text)
        
        if len(words) == 0:
            raise HTTPException(
                status_code=400, 
                detail="Review text contains no valid words after preprocessing"
            )
        
        # Vectorize
        vector = vectorize_text(words)
        
        # Reshape for model input
        features = vector.reshape(1, -1)
        
        # Apply feature selection if available
        if feature_selector is not None:
            features = feature_selector.transform(features)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get prediction probability (confidence)
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities))
        else:
            confidence = 1.0  # Default confidence for models without predict_proba
        
        # Get sentiment label
        sentiment = sentiment_labels.get(prediction, "unknown")
        
        return PredictionOutput(
            review_text=review.review_text,
            sentiment=sentiment,
            sentiment_code=int(prediction),
            confidence=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(reviews: list[ReviewInput]):
    """
    Predict sentiment for multiple reviews
    
    Returns list of predictions
    """
    try:
        results = []
        for review in reviews:
            prediction = await predict_sentiment(review)
            results.append(prediction)
        return results
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)