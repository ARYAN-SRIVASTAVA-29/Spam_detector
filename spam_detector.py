# Import Libraries
import pandas as pd
import numpy as np  # Import numpy if needed
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load Data
data = pd.read_csv('sms.tsv', sep='\t', header=None, names=['label', 'message'])

# Shuffle the dataset with a fixed random state
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Display the first few rows (optional)
print(data.head())

# Split Data into Features and Labels
X = data['message']  # Messages
y = data['label']    # Labels (spam or ham)

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the Count Vectorizer
vectorizer = CountVectorizer()

# Fit the Vectorizer on the Training Data and Transform It
X_train_counts = vectorizer.fit_transform(X_train)

# Transform the Test Data
X_test_counts = vectorizer.transform(X_test)

# Initialize the Multinomial Naive Bayes Classifier
clf = MultinomialNB()

# Train the Classifier
clf.fit(X_train_counts, y_train)

# Make Predictions on the Test Data
y_pred = clf.predict(X_test_counts)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Display the Classification Report
print('\nClassification Report:\n', classification_report(y_test, y_pred))

# Function to Predict Whether a Message is Spam or Ham
def predict_spam(message):
    # Transform the Message Using the Fitted Vectorizer
    message_counts = vectorizer.transform([message])
    # Predict the Label
    prediction = clf.predict(message_counts)
    # Print the Result
    print('The message is:', prediction[0])

# Test the Function with Different Messages
# new_message = "Congratulations! You've won a $1000 gift card. Call now!"
# predict_spam(new_message)

new_message = "Hi, It's been a long time we met"
predict_spam(new_message)

