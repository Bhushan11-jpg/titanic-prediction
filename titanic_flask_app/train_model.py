import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

# Load and clean data
df = pd.read_csv("Titanic-Dataset.csv")
df = df.drop(columns=["Name", "sssCabin", "Ticket"])
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.dropna(inplace=True)

X = df[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = df['Survived']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "titanic_model.pkl")
print("✅ Model trained and saved as titanic_model.pkl")
