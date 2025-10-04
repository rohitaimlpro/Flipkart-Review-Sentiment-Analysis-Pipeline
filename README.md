# Flipkart Review Sentiment Analysis Pipeline

An end-to-end machine learning pipeline for sentiment analysis on product reviews, featuring experiment tracking, data versioning, and API inference capabilities.

## Project Overview

This project implements a complete ML workflow for classifying customer reviews into positive, negative, or neutral sentiments. Built with MLOps principles, it includes data preprocessing, model training with experiment tracking, and a REST API for inference.

**Dataset:** 200,000+ Flipkart product reviews  
**Best Model Performance:** 88.4% accuracy (Random Forest)  
**Technologies:** Python, scikit-learn, spaCy, MLflow, DVC, FastAPI

---

## Features

- Text preprocessing pipeline (tokenization, stopword removal, stemming)
- Feature extraction using spaCy word embeddings (96-dimensional vectors)
- Multiple model comparison (Logistic Regression, Decision Tree, Random Forest)
- Experiment tracking with MLflow
- Data versioning with DVC
- REST API for real-time predictions
- Integration with DagsHub for collaborative ML workflow

---

## Project Structure

```
flipkart-sentiment-mlops/
│
├── data/
│   ├── raw/                    # Original dataset (tracked by DVC)
│   └── processed/              # Processed data
│
├── models/                     # Trained models
│   ├── best_model.pkl         # Random Forest (88.4% accuracy)
│   └── feature_selector.pkl   # Feature selection transformer
│
├── src/
│   ├── preprocessing.py       # Text cleaning and tokenization
│   ├── feature_extraction.py  # spaCy vectorization
│   ├── train.py              # Model training with MLflow
│   ├── evaluate.py           # Metrics calculation
│   └── utils.py              # Helper functions
│
├── api/
│   └── main.py               # FastAPI application
│
├── params.yaml               # Configuration and hyperparameters
├── dvc.yaml                  # DVC pipeline definition
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.11
- Git
- DVC (optional, for data versioning)

### 1. Clone Repository

```bash
git clone https://dagshub.com/rs5294645/flipkart-sentiment-mlops.git
cd flipkart-sentiment-mlops
```

### 2. Create Virtual Environment

```bash
# Using conda
conda create -n sentiment python=3.11 -y
conda activate sentiment

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

# Download required NLP data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
python -m spacy download en_core_web_sm
```

### 4. Pull Data (Optional - if using DVC)

```bash
dvc pull
```

---

## Usage

### Training Models

```bash
# Run complete pipeline
python src/preprocessing.py      # ~2 minutes
python src/feature_extraction.py # ~24 minutes (one-time)
python src/train.py              # ~5 minutes
```

Or use DVC to run the entire pipeline:

```bash
dvc repro
```

### Starting the API

```bash
# Start FastAPI server
python api/main.py

# Or with uvicorn
uvicorn api.main:app --reload --port 8000
```

API will be available at: `http://localhost:8000`

### API Documentation

Interactive API docs: `http://localhost:8000/docs`

#### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"review_text": "Excellent product! Fast delivery and great quality."}'
```

#### Example Response

```json
{
  "review_text": "Excellent product! Fast delivery and great quality.",
  "sentiment": "positive",
  "sentiment_code": 2,
  "confidence": 0.92
}
```

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 84.2% | 82.4% | 84.2% | 82.1% |
| Decision Tree | 83.7% | 83.5% | 83.7% | 83.7% |
| **Random Forest** | **88.4%** | **87.5%** | **88.4%** | **86.6%** |

**Dataset Distribution:**
- Positive: 166,575 reviews (81%)
- Negative: 28,232 reviews (14%)
- Neutral: 10,234 reviews (5%)

---

## MLOps Components

### Experiment Tracking (MLflow)

All training runs are logged with:
- Hyperparameters
- Performance metrics
- Model artifacts

View experiments on DagsHub: [Project Experiments](https://dagshub.com/rs5294645/flipkart-sentiment-mlops/experiments)

### Data Versioning (DVC)

- Raw dataset tracked with DVC
- Pipeline stages defined in `dvc.yaml`
- Reproducible workflow with `dvc repro`

### Configuration Management

All hyperparameters centralized in `params.yaml`:
- Model parameters
- Feature extraction settings
- Train/test split ratios

---

## Pipeline Stages

1. **Preprocessing** (~2 min)
   - Load raw CSV data
   - Handle missing values
   - Tokenize text
   - Remove stopwords
   - Apply stemming
   - Encode sentiment labels

2. **Feature Extraction** (~24 min)
   - Vectorize text using spaCy (en_core_web_sm)
   - Generate 96-dimensional word embeddings
   - Select top 60 features using mutual information

3. **Training** (~5 min)
   - Train multiple models
   - Log experiments to MLflow
   - Save best performing model
   - Generate performance metrics

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | Health check status |
| `/predict` | POST | Predict sentiment for single review |
| `/predict_batch` | POST | Predict sentiment for multiple reviews |
| `/docs` | GET | Interactive API documentation |

---

## Configuration

Edit `params.yaml` to modify:

- Model hyperparameters
- Feature extraction settings
- Train/test split ratio
- MLflow tracking URI
- API settings

---

## Known Limitations

- Feature extraction is CPU-intensive (24 minutes for 200K reviews)
- DVC pipeline metrics validation has issues with MLflow artifacts
- Model serves locally only (no cloud deployment)
- No automated retraining pipeline
- Limited to English language reviews

---

## Future Improvements

- [ ] Implement transformer models (BERT, RoBERTa) for improved accuracy
- [ ] Add caching layer for feature extraction
- [ ] Deploy API to cloud platform (AWS/GCP/Heroku)
- [ ] Implement CI/CD pipeline for automated retraining
- [ ] Add comprehensive unit and integration tests
- [ ] Support multilingual sentiment analysis
- [ ] Implement model monitoring and drift detection

---

## Technologies Used

**ML/NLP:**
- scikit-learn - Model training
- spaCy - Text vectorization
- NLTK - Text preprocessing

**MLOps:**
- MLflow - Experiment tracking
- DVC - Data versioning
- DagsHub - Collaborative platform

**API:**
- FastAPI - REST API framework
- Uvicorn - ASGI server
- Pydantic - Data validation

---

## Contributing

This is a portfolio project and not actively maintained. Feel free to fork and adapt for your own use.

---

## License

MIT License

---

## Contact

**GitHub:** [rs5294645](https://github.com/rs5294645)  
**DagsHub:** [Project Repository](https://dagshub.com/rs5294645/flipkart-sentiment-mlops)

---

## Acknowledgments

- Dataset source: Flipkart product reviews
- MLflow and DVC communities for excellent documentation
- DagsHub for providing collaborative ML platform