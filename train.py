# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Step 1: Load the dataset
df = pd.read_csv("phq9_dummy_dataset_1000.csv")  # Replace with correct path if needed

# Step 2: Prepare features and labels
X = df[[f'q{i}' for i in range(1, 10)]]  # q1 to q9
y = df['label']

# Step 3: Encode labels (e.g., "Mild" -> 0, "Moderate" -> 1, etc.)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Optional: Save label encoder for decoding later
joblib.dump(label_encoder, 'label_encoder.pkl')

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 5: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)

print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("🧱 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Save the trained model
joblib.dump(model, 'phq9_model.pkl')
print("\n✅ Model saved as 'phq9_model.pkl'")
print("✅ Accuracy Score:", accuracy_score(y_test, y_pred))
