# ================================================
# Student Performance Prediction
# Author: Divyanshi Rana
# Tools Used: Python, Pandas, Matplotlib, Scikit-learn
# Algorithm: Linear Regression
# ================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ── Step 1: Load Dataset ─────────────────────────
df = pd.read_csv("C:/Users/dell/Desktop/student_data.csv")
print("Dataset loaded:", df.shape[0], "students")

# ── Step 2: Select Input and Output ─────────────
X = df[["G1", "G2", "studytime", "absences", "failures"]]
y = df["G3"]

# ── Step 3: Split into Train and Test ───────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

# ── Step 4: Train Model ──────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully!")

# ── Step 5: Check Accuracy ───────────────────────
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# ── Step 6: Predict for a New Student ───────────
new_student = pd.DataFrame({
    "G1"       : [12],
    "G2"       : [13],
    "studytime": [2],
    "absences" : [4],
    "failures" : [0]
})

result = model.predict(new_student)[0]
print("Predicted Final Grade:", round(result, 1), "/ 20")

# ── Step 7: Graph ────────────────────────────────
plt.scatter(y_test, y_pred, color="blue", label="Predicted vs Actual")
plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.title("Student Performance Prediction")
plt.legend()
plt.savefig("C:/Users/dell/Desktop/student_graph.png")
plt.show()
print("Graph saved on Desktop!")
