from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

model = joblib.load("model/breast_cancer_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
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

    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

