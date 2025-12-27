import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# 1. Load Data
url = "https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv"
try:
    data = pd.read_csv(url)
except Exception as e:
    print(f"Error downloading data: {e}")
    exit()

# --- DEBUGGING: Print original columns ---
print("Original Columns:", data.columns.tolist())

# --- THE FIX: Rename by index (0=Text, 1=Label) ---
# This ignores spelling/hidden character issues entirely
data.columns = ['text', 'label']

# 2. Preprocessing
# Force text to string to avoid errors if there are numbers interpreted as float
data['text'] = data['text'].astype(str)
data.dropna(inplace=True)

# 3. Split Data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], 
    data['label'], 
    test_size=0.2, 
    random_state=42
)

# 4. Build Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('clf', LogisticRegression(solver='liblinear'))
])

# 5. Train
print("Training model...")
pipeline.fit(X_train, y_train)
print(f"Model Accuracy: {pipeline.score(X_test, y_test):.2f}")

# 6. Save Model
joblib.dump(pipeline, 'sentiment_model.pkl')
print("Model saved as sentiment_model.pkl")