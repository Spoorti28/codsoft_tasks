
# IRIS FLOWER CLASSIFICATION
# Machine Learning Project using Decision Tree Classifier

# 1. Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# 2. Load the dataset
df = pd.read_csv("IRIS.csv")
print("IRIS FLOWER PROJECT STARTED")
print("Dataset loaded successfully!")
print(df.head())

# 3. Explore the dataset
print("\nDataset Information:")
print(df.info())

print("\nDataset Description:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSpecies Count:")
print(df["species"].value_counts())

# 4. Separate features and target
X = df.drop("species", axis=1)
y = df["species"]

# 5. Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# 6. Create and train Decision Tree model
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

print("\nDecision Tree trained successfully!")

# 7. Make predictions
y_pred = model.predict(X_test)

print("\nPredicted Species:")
print(y_pred)

# 8. Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# 9. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.title("Confusion Matrix - Iris Classification")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.tight_layout()
plt.show()

# 11. Visualize Decision Tree
plt.figure(figsize=(16, 9))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)

plt.title("Decision Tree - Iris Flower Classification")
plt.show()

# 12. Predict a new Iris flower
# Example measurements:
# Sepal Length = 5.1
# Sepal Width  = 3.5
# Petal Length = 1.4
# Petal Width  = 0.2

new_flower = pd.DataFrame(
    [[5.1, 3.5, 1.4, 0.2]],
    columns=X.columns
)

prediction = model.predict(new_flower)

print("\nNew Flower Prediction:")
print("Predicted Iris Species:", prediction[0])

print("\nIris Flower Classification Completed Successfully!")
