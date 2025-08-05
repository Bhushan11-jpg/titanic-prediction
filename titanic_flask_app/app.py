from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("titanic_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    pclass = int(request.form["pclass"])
    sex = int(request.form["sex"])
    age = float(request.form["age"])
    fare = float(request.form["fare"])
    embarked = int(request.form["embarked"])

    features = np.array([[pclass, sex, age, fare, embarked]])
    prediction = model.predict(features)[0]
    result = "🎉 Survived!" if prediction == 1 else "❌ Did not survive."

    return render_template("index.html", prediction_text=f"Prediction: {result}")

if __name__ == "__main__":
    app.run(debug=True) 
