from sklearn.dummy import DummyClassifier
import numpy as np
import joblib

# Dummy data
X = np.array([[0], [1], [0], [1]])
y = np.array([0, 1, 0, 1])

# Create simple dummy model
model = DummyClassifier(strategy="most_frequent")
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("model.pkl created!")