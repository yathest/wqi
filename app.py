from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the saved LinearRegression model
with open('linear_regression_model.pkl', 'rb') as file:
    model1 = pickle.load(file)

# Load the saved StandardScaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input date from the form
        selected_date = request.form['date']

        # Create a datetime object with the selected date and fixed time at 12:00:00
        selected_datetime = pd.to_datetime(selected_date + ' 12:00:00')

        # Use the loaded scaler to transform the input feature
        new_datetime_numeric = selected_datetime.timestamp()
        new_datetime_scaled = scaler.transform([[new_datetime_numeric]])
        
        # Make a prediction using the loaded model
        predicted_wqi = model1.predict(new_datetime_scaled)
        
        rounded_wqi = round(predicted_wqi[0], 2)
        
        # Prepare the response
        response = {
            'date': selected_date,
            'predicted_wqi': rounded_wqi
        }

        return render_template('index.html', prediction=response)

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)

