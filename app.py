

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# Load the saved model files
model = joblib.load(BASE_DIR / "loan_model.pkl")
scaler = joblib.load(BASE_DIR / "scaler.pkl")
feature_names = joblib.load(BASE_DIR / "feature_names.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read the form values using the actual feature names
        input_values = []

        for feature in feature_names:
            value = request.form.get(feature)

            if value is None or value.strip() == "":
                return render_template(
                    "index.html",
                    error=f"Please enter a value for {feature}.",
                    features=feature_names,
                    form_data=request.form
                )

            input_values.append(float(value))

        # Preserve feature names and their training order
        input_df = pd.DataFrame(
            [input_values],
            columns=feature_names
        )

        # Apply the same scaler used during training
        input_scaled = scaler.transform(input_df)

        # Predict using the trained model
        prediction = model.predict(input_scaled)[0]

        # Display the actual model output
        result = str(prediction)

        return render_template(
            "index.html",
            prediction=result,
            features=feature_names,
            form_data=request.form
        )

    except ValueError:
        return render_template(
            "index.html",
            error="Please enter valid numeric values.",
            features=feature_names,
            form_data=request.form
        )

    except Exception as e:
        app.logger.exception("Prediction failed")
        return render_template(
            "index.html",
            error="Prediction failed. Check the terminal for details.",
            features=feature_names,
            form_data=request.form
        )


if __name__ == "__main__":
    app.run(debug=True)
