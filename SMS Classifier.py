# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the Dataset
data = pd.read_csv("train.csv")

# Step 3: Data Exploration
print(data.head())  # Display the first few rows of the dataset
print(data.info())  # Get information about the dataset

# Step 4: Data Preprocessing
# Assuming the dataset has columns named 'text' and 'label'
X = data['sms']
y = data['label']

# Step 5: Feature Engineering
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Step 6: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 8: Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
