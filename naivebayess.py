import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load data
data = pd.read_excel('handphone_data_1000.xlsx')  # Use read_excel for Excel files

# Perform One-Hot Encoding for categorical feature 'merk'
data_encoded = pd.get_dummies(data, columns=['merk'])

# Split data into X (features) and y (target)
X = data_encoded.drop('layak_beli', axis=1)
y = data_encoded['layak_beli']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Visualize the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Confusion Matrix')
plt.imshow(confusion_matrix(y_test, y_pred), interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.xlabel('Predicted labels')
plt.ylabel('True labels')

plt.subplot(1, 2, 2)
plt.title('Classification Report')
report = classification_report(y_test, y_pred, output_dict=True)

# Calculate metrics
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1_score = report['weighted avg']['f1-score']

# Plot precision, recall, and f1-score
metrics = ['Precision', 'Recall', 'F1-score']
values = [precision, recall, f1_score]
plt.bar(metrics, values, color='green')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)  # Set y-axis limit to range of [0, 1] for clarity
plt.tight_layout()

plt.show()
