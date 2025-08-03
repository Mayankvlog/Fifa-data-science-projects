import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import plotly.graph_objects as go
import os
from tensorflow.keras.losses import MeanSquaredError # Import MSE

# Set page configuration
st.set_page_config(
    page_title="FIFA Player Analysis",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
    }
    .stAlert {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .stTextInput, .stNumberInput {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        # Provide custom objects dictionary for loading the model
        model = tf.keras.models.load_model('fifa_player_model.h5', custom_objects={'mse': MeanSquaredError()})
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure to run the training notebook first to generate the model files.")
        return None, None

def create_radar_chart(current_stats, predicted_stats, categories):
    """Create a radar chart comparing current and predicted stats"""
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=current_stats,
        theta=categories,
        fill='toself',
        name='Current',
        line_color='#3498db'
    ))

    fig.add_trace(go.Scatterpolar(
        r=predicted_stats,
        theta=categories,
        fill='toself',
        name='Predicted',
        line_color='#2ecc71'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(current_stats), max(predicted_stats)) * 1.2]
            )
        ),
        showlegend=True,
        title='Current vs Predicted Stats'
    )

    return fig

def create_gauge_chart(value, title, min_val=0, max_val=100):
    """Create a gauge chart for displaying metrics"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': "#2ecc71"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_val, max_val/3], 'color': '#ff9999'},
                {'range': [max_val/3, 2*max_val/3], 'color': '#ffdd99'},
                {'range': [2*max_val/3, max_val], 'color': '#99ff99'}
            ],
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=50, b=10),
        font={'color': "#2c3e50", 'family': "Arial"}
    )

    return fig

def main():
    st.title("FIFA Player Analysis & Prediction System ⚽")
    st.markdown("---")

    # Load model and scaler
    model, scaler = load_model_and_scaler()

    if model is None or scaler is None:
        st.warning("⚠️ Model not loaded. Please check if model files exist in the 'model' directory.")
        return

    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Player Analysis", "About"])

    with tab1:
        st.header("Player Analysis")

        # Create two columns for input and results
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Player Information")

            with st.form("player_form"):
                # Basic Info
                name = st.text_input("Player Name", "")
                age = st.number_input("Age", min_value=16, max_value=45, value=20)
                position = st.selectbox("Position",
                    ["Forward", "Midfielder", "Defender", "Goalkeeper"])

                # Current Stats
                st.markdown("### Current Statistics")
                overall = st.slider("Overall Rating", 1, 99, 75)
                potential = st.slider("Potential", 1, 99, 80)
                value = st.number_input("Market Value (M €)",
                    min_value=0.0, max_value=200.0, value=1.0, step=0.1)

                # Additional Stats (These are not used for prediction but for display)
                pace = st.slider("Pace", 1, 99, 70)
                shooting = st.slider("Shooting", 1, 99, 70)
                passing = st.slider("Passing", 1, 99, 70)
                dribbling = st.slider("Dribbling", 1, 99, 70)
                defending = st.slider("Defending", 1, 99, 70)
                physical = st.slider("Physical", 1, 99, 70)


                submitted = st.form_submit_button("Analyze Player")

        if submitted:
            with col2:
                st.subheader("Analysis Results")

                with st.spinner('Analyzing player data...'):
                    try:
                        # Prepare input data - only use the features used for training
                        # Scale the features used for training (Overall, Potential, Value)
                        input_features_scaled = scaler.transform(np.array([[overall, potential, value]]))

                        # Combine age (unscaled) with scaled features
                        input_for_prediction = np.hstack((np.array([[age]]), input_features_scaled))

                        # Reshape the input for the RNN model (time_steps=1)
                        input_for_prediction = input_for_prediction.reshape(1, 1, input_for_prediction.shape[-1])

                        # Make prediction
                        predictions = model.predict(input_for_prediction)
                        predictions_unscaled = scaler.inverse_transform(predictions)

                        # Extract predictions
                        pred_overall = round(float(predictions_unscaled[0][0]))
                        pred_potential = round(float(predictions_unscaled[0][1]))
                        pred_value = round(float(predictions_unscaled[0][2]), 2)

                        # Display player info
                        st.markdown(f"### {name if name else 'Player'} Analysis")
                        st.markdown(f"**Position:** {position}")

                        # Create metrics
                        col2_1, col2_2, col2_3 = st.columns(3)

                        with col2_1:
                            st.plotly_chart(
                                create_gauge_chart(pred_overall, "Predicted Overall"),
                                use_container_width=True
                            )

                        with col2_2:
                            st.plotly_chart(
                                create_gauge_chart(pred_potential, "Predicted Potential"),
                                use_container_width=True
                            )

                        with col2_3:
                            st.plotly_chart(
                                create_gauge_chart(
                                    pred_value, "Predicted Value (M €)",
                                    min_val=0, max_val=max(200, pred_value)
                                ),
                                use_container_width=True
                            )

                        # Create radar charts
                        current_stats = [overall, potential, value]
                        predicted_stats = [pred_overall, pred_potential, pred_value]
                        categories = ['Overall', 'Potential', 'Value (M)']

                        fig1 = create_radar_chart(current_stats, predicted_stats, categories)
                        st.plotly_chart(fig1, use_container_width=True)

                        # Player attributes radar chart (Using input attributes directly for display)
                        attr_stats = [pace, shooting, passing, dribbling, defending, physical]
                        # Note: The prediction logic only uses Overall, Potential, Value.
                        # We can't directly predict the future of these individual attributes
                        # with the current model setup. Displaying current ones for context.
                        fig2 = create_radar_chart(
                            attr_stats,
                            attr_stats, # Displaying current attributes as predicted for now
                            ['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physical']
                        )
                        st.plotly_chart(fig2, use_container_width=True)

                        # Analysis summary
                        st.markdown("### Development Projection")

                        growth = pred_overall - overall
                        potential_diff = pred_potential - potential
                        value_change = pred_value - value

                        analysis = f"""
                        Based on the analysis:

                        🎯 **Overall Development:**
                        - Current: {overall} → Predicted: {pred_overall}
                        - Expected to {'increase' if growth > 0 else 'decrease'} by {abs(growth)} points

                        ⭐ **Potential Change:**
                        - Current: {potential} → Predicted: {pred_potential}
                        - Potential expected to {'improve' if potential_diff > 0 else 'decline'} by {abs(potential_diff)} points

                        💰 **Market Value Projection:**
                        - Current: €{value}M → Predicted: €{pred_value}M
                        - Value projected to {'rise' if value_change > 0 else 'fall'} by €{abs(round(value_change, 2))}M

                        """
                        # Remove attribute-specific analysis as the model doesn't predict them directly
                        # 💫 **Key Attributes Development:**
                        # - Most improved: {['Pace', 'Shooting', 'Passing', 'Dribbling', 'Defending', 'Physical'][np.argmax(np.array(pred_attrs) - np.array(attr_stats))]}
                        # - Average attribute change: {round(np.mean(np.array(pred_attrs) - np.array(attr_stats)), 2)} points


                        st.markdown(analysis)

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
    # Footer
    st.markdown("---")


if __name__ == "__main__":
    main()
