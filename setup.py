"""
Setup script to initialize the project
"""
import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary project directories"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'mlruns',
        'logs',
        'api',
        'src',
        'tests'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {directory}")


def create_init_files():
    """Create __init__.py files"""
    init_dirs = ['src', 'api', 'tests']
    
    for directory in init_dirs:
        init_file = Path(directory) / '__init__.py'
        init_file.touch()
        logger.info(f"✓ Created: {init_file}")


def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        logger.info("Downloading NLTK data...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        logger.info("✓ NLTK data downloaded")
    except Exception as e:
        logger.error(f"✗ Failed to download NLTK data: {e}")


def download_spacy_model():
    """Download spaCy English model"""
    try:
        logger.info("Downloading spaCy model...")
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
            check=True,
            capture_output=True
        )
        logger.info("✓ spaCy model downloaded")
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to download spaCy model: {e}")


def initialize_dvc():
    """Initialize DVC"""
    try:
        if not Path('.dvc').exists():
            logger.info("Initializing DVC...")
            subprocess.run(["dvc", "init"], check=True, capture_output=True)
            logger.info("✓ DVC initialized")
        else:
            logger.info("✓ DVC already initialized")
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠ DVC initialization failed: {e}")
        logger.warning("  Please run 'dvc init' manually")
    except FileNotFoundError:
        logger.warning("⚠ DVC not found. Install it with: pip install dvc")


def initialize_git():
    """Initialize Git repository"""
    try:
        if not Path('.git').exists():
            logger.info("Initializing Git...")
            subprocess.run(["git", "init"], check=True, capture_output=True)
            logger.info("✓ Git initialized")
        else:
            logger.info("✓ Git already initialized")
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠ Git initialization failed: {e}")
    except FileNotFoundError:
        logger.warning("⚠ Git not found. Install it first.")


def main():
    """Run all setup tasks"""
    logger.info("\n=== Setting Up Flipkart Sentiment MLOps Project ===\n")
    
    create_directories()
    create_init_files()
    download_nltk_data()
    download_spacy_model()
    initialize_git()
    initialize_dvc()
    
    logger.info("\n=== Setup Complete! ===\n")
    logger.info("Next steps:")
    logger.info("1. Place your dataset in data/raw/Dataset-SA.csv")
    logger.info("2. Run: python src/preprocessing.py")
    logger.info("3. Run: python src/feature_extraction.py")
    logger.info("4. Run: python src/train.py")
    logger.info("5. Start API: uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()