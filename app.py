from flask import Flask, render_template, request
import joblib


# Create Flask application
app = Flask(__name__)


# Load trained model
model = joblib.load("model/crop_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from form
    nitrogen = float(request.form["N"])
    phosphorus = float(request.form["P"])
    potassium = float(request.form["K"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    # Arrange inputs in the same order
    features = [[
        nitrogen,
        phosphorus,
        potassium,
        temperature,
        humidity,
        ph,
        rainfall
    ]]

    # Make prediction
    prediction = model.predict(features)

    crop = prediction[0]

    return render_template(
        "index.html",
        prediction=crop
    )


if __name__ == "__main__":
    app.run(debug=True)