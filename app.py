from flask import Flask, render_template, request
import joblib
import numpy as np

# Step 1: Initialize Flask app
app = Flask(__name__)

# Step 2: Load the saved model
model = joblib.load('diabetes_model.pkl')

# Step 3: Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Step 4: Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Step 4a: Extract features from form
    features = [float(x) for x in request.form.values()]
    # Step 4b: Convert to NumPy array
    final_features = [np.array(features)]
    # Step 4c: Predict using loaded model
    prediction = model.predict(final_features)[0]
    
    # Step 4d: Display output
    output = "You are likely to have diabetes." if prediction == 1 else "You are unlikely to have diabetes."
    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(debug=True)
