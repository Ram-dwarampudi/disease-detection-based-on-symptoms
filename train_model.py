import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load dataset
data = pd.read_csv("Training.csv")

# Split features and target
X = data.drop("prognosis", axis=1)
y = data["prognosis"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
with open("model copy.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")
