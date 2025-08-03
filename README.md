# FIFA Player Analysis & Prediction System 🎮⚽

An advanced machine learning-powered web application for analyzing and predicting football player development using Streamlit and TensorFlow.

## 🌟 Features

### Player Analysis
- **Basic Information Input**
  - Player name and age
  - Position selection (Forward, Midfielder, Defender, Goalkeeper)
  - Current market value assessment

### Comprehensive Stats Tracking
- **Core Attributes**
  - Overall rating (1-99)
  - Potential rating (1-99)
  - Market value (up to €200M)

- **Detailed Player Metrics**
  - Pace
  - Shooting
  - Passing
  - Dribbling
  - Defending
  - Physical attributes

### Advanced Visualizations
- **Interactive Gauge Charts**
  - Predicted overall rating
  - Future potential
  - Projected market value

- **Dual Radar Charts**
  - Current vs. predicted main stats
  - Attribute-specific development tracking

### Development Projections
- Detailed growth analysis
- Potential improvement tracking
- Market value forecasting
- Individual attribute development

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Required Packages
```bash
pip install -r requirements.txt
```

Key dependencies:
- Streamlit
- TensorFlow
- NumPy
- Pandas
- Plotly

### Running the Application
```bash
streamlit run fifa_analysis_app.py
```

## 💻 Usage Guide

1. **Input Player Data**
   - Enter basic player information
   - Set current attributes and ratings
   - Input market value

2. **Generate Analysis**
   - Click "Analyze Player"
   - View comprehensive predictions
   - Examine visual representations

3. **Interpret Results**
   - Review development projections
   - Analyze attribute changes
   - Assess market value trends

## 🎯 Model Architecture

The system uses a sophisticated machine learning model trained on historical player development data, featuring:
- Deep Neural Network architecture
- Multiple processing layers
- Advanced pattern recognition
- Specialized football metrics analysis

## 📊 Output Metrics

- **Performance Predictions**
  - Future overall rating
  - Potential development curve
  - Market value projections

- **Attribute Development**
  - Individual skill growth
  - Position-specific improvements
  - Comparative analysis

## ⚙️ Technical Requirements

- **System Requirements**
  - Modern web browser
  - 4GB RAM minimum
  - Internet connection

- **File Structure**
  ```
  fifa_mlops/
  ├── fifa_analysis_app.py
  ├── requirements.txt
  ├── README.md
  └── model/
      ├── fifa_player_model.h5
      └── scaler.pkl
  ```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, create issues, or suggest improvements.

## 📝 Note

This tool is designed to assist in player evaluation and should be used in conjunction with traditional scouting methods and professional assessment.

## ⭐ Support

If you find this project helpful, please give it a star on GitHub!