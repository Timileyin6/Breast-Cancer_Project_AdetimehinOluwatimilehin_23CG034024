from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and scaler with error handling
try:
    model = joblib.load("model/breast_cancer_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
except FileNotFoundError:
    print("Error: Model or scaler file not found. Please ensure model files exist.")
    model = None
    scaler = None

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None

    if request.method == "POST":
        try:
            if model is None or scaler is None:
                error = "Model not loaded. Please check server logs."
            else:
                values = [
                    float(request.form["radius"]),
                    float(request.form["texture"]),
                    float(request.form["perimeter"]),
                    float(request.form["area"]),
                    float(request.form["smoothness"])
                ]
                values_scaled = scaler.transform([values])
                pred = model.predict(values_scaled)[0]
                result = "Benign ✅" if pred == 1 else "Malignant ❌"
        except ValueError:
            error = "Invalid input. Please enter valid numbers."
        except Exception as e:
            error = f"Prediction error: {str(e)}"

    return render_template("index.html", result=result, error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

