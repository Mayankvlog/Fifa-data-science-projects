from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import plotly
import plotly.graph_objs as go
import json

app = Flask(__name__)

# Load the trained model and scaler
def load_model_and_scaler():
    model = tf.keras.models.load_model('model/fifa_player_model.h5')
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Create prediction plot
def create_prediction_plot(current_stats, predicted_stats):
    categories = ['Overall', 'Potential', 'Value (M)']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=current_stats,
        theta=categories,
        fill='toself',
        name='Current'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=predicted_stats,
        theta=categories,
        fill='toself',
        name='Predicted'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(current_stats), max(predicted_stats)) * 1.2]
            )),
        showlegend=True,
        title='Current vs Predicted Stats'
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        age = float(request.form['age'])
        overall = float(request.form['overall'])
        potential = float(request.form['potential'])
        value = float(request.form['value'])
        
        # Load model and scaler
        model, scaler = load_model_and_scaler()
        
        # Prepare input data
        input_data = np.array([[age, overall, potential, value]])
        input_scaled = scaler.transform(input_data)
        
        # Reshape for RNN (assuming time steps of 1)
        input_reshaped = input_scaled.reshape(1, 1, 4)
        
        # Make prediction
        prediction = model.predict(input_reshaped)
        prediction_unscaled = scaler.inverse_transform(prediction.reshape(1, -1))
        
        # Prepare prediction results
        result = {
            'overall': round(float(prediction_unscaled[0][0])),
            'potential': round(float(prediction_unscaled[0][1])),
            'value': round(float(prediction_unscaled[0][2]), 2)
        }
        
        # Create plot
        current_stats = [overall, potential, value]
        predicted_stats = [result['overall'], result['potential'], result['value']]
        plot_div = create_prediction_plot(current_stats, predicted_stats)
        
        return render_template('index.html', prediction=result, plot_div=plot_div)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
