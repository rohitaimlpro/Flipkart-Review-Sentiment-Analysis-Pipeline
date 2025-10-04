import time
import pandas as pd
import spacy

print("Loading spaCy model...")
start_load = time.time()
nlp = spacy.load('en_core_web_sm')
load_time = time.time() - start_load
print(f"Model loaded in {load_time:.2f}s")

print("\nLoading preprocessed data (first 10 reviews)...")
df = pd.read_json('data/processed/preprocessed_data.json', lines=True, nrows=10)

print(f"Processing {len(df)} reviews...")
start = time.time()

for idx, words in enumerate(df['words']):
    text = ' '.join(words)
    doc = nlp(text)
    vector = doc.vector
    if idx == 0:
        print(f"  Review 1 processed in {time.time() - start:.2f}s")

elapsed = time.time() - start
avg_per_review = elapsed / len(df)
total_reviews = 205041
estimated_total_seconds = avg_per_review * total_reviews
estimated_hours = estimated_total_seconds / 3600

print(f"\n=== Timing Results ===")
print(f"Time for 10 reviews: {elapsed:.2f}s")
print(f"Average per review: {avg_per_review:.3f}s")
print(f"Estimated total time: {estimated_hours:.1f} hours ({estimated_total_seconds/60:.0f} minutes)")
print(f"\nFor {total_reviews:,} reviews")