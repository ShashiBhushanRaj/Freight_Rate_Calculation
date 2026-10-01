PortCast — AI-Powered Dry Bulk Freight Forecasting & Chartering Decision Support

An AI-powered dry bulk freight forecasting and decision-support system that combines market data, time-series machine learning, vessel economics, cargo intelligence, and risk analysis to support smarter chartering decisions.

📌 Overview

PortCast is an AI-powered freight intelligence and decision-support platform developed as part of the Smart India Hackathon (SIH).

The system forecasts future dry bulk freight rates and combines those forecasts with vessel information, cargo data, fuel prices, trade information, economic analysis, and risk indicators to support chartering decisions.

Core Pipeline
Freight Market Data
        ↓
Data Processing
        ↓
Feature Engineering
        ↓
Machine Learning Models
   ┌────┴────┐
   ↓         ↓
XGBoost    LSTM
   └────┬────┘
        ↓
Freight Forecast
        ↓
Economic Analysis
        ↓
Risk Analysis
        ↓
Chartering Decision Support
🎯 Problem Statement

The dry bulk shipping market is highly dynamic and influenced by multiple factors:

Freight-rate fluctuations
Cargo demand
Commodity movements
Port activity
International trade
Fuel prices
Vessel characteristics
Market volatility
Supply and demand conditions

Traditional chartering decisions can depend heavily on historical market information and manual analysis.

PortCast aims to provide a unified AI-driven system that can:

Analyze historical freight-market data.
Engineer meaningful time-series features.
Forecast future freight rates.
Analyze vessel and voyage economics.
Incorporate cargo and market information.
Assess risk-related signals.
Provide chartering decision support.
🚀 Key Features
📈 1. Freight Rate Forecasting

PortCast uses historical freight-market information to forecast future freight rates.

The project explores multiple forecasting horizons:

7 Observations
30 Observations
60 Observations
90 Observations

Future targets are generated using shifted observations:

df["Target_7D"] = df["KDCI"].shift(-7)
df["Target_30D"] = df["KDCI"].shift(-30)
df["Target_60D"] = df["KDCI"].shift(-60)
df["Target_90D"] = df["KDCI"].shift(-90)

Note: These targets are generated using row/observation shifts. They represent future observations and should only be described as exact calendar-day forecasts when the underlying data has a continuous daily frequency.

🤖 2. Machine Learning Models
XGBoost

XGBoost is used as a tabular-data forecasting baseline.

Historical Freight Data
        ↓
Feature Engineering
        ↓
Lag Features
Rolling Statistics
Temporal Features
        ↓
XGBoost
        ↓
Freight Forecast
LSTM

LSTM is used to capture sequential and temporal patterns in the freight-rate data.

Input Sequence
      ↓
LSTM (128)
      ↓
Dropout (0.2)
      ↓
LSTM (64)
      ↓
Dropout (0.2)
      ↓
Dense (32, ReLU)
      ↓
Dense (1)
      ↓
Forecast

Separate LSTM models are trained for different forecasting horizons.

📊 Data Sources
🚢 Freight Market Data

KOBC freight-market data is used as the primary market signal.

The prototype dataset contains approximately:

~2,573 observations
~10 years of historical data

KDCI-related freight-rate information forms the foundation of the forecasting pipeline.

🇮🇳 Indian Port Cargo Data

Port and cargo information provides additional signals related to:

Cargo movement
Port activity
Commodity flows
Demand conditions
🌾 Commodity Data

Commodity-related information provides additional demand-side signals that can influence dry bulk shipping activity.

🌍 UN Comtrade Data

International trade data provides information related to:

Imports
Exports
Commodity trade flows
Country-level trade activity
🛢️ Brent / Fuel Price Data

Fuel prices are an important component of shipping economics because fuel costs directly affect voyage profitability.

🧮 Feature Engineering
Temporal Features
Year
Month
Quarter
Day of Week
Lag Features
KDCI Lag 1
KDCI Lag 7
KDCI Lag 14
KDCI Lag 30
Rolling Features
7-Period Moving Average
30-Period Moving Average
7-Period Rolling Standard Deviation
30-Period Rolling Standard Deviation

These features help capture:

Recent market trends
Market momentum
Short-term volatility
Longer-term market behavior
🔬 Data Preprocessing

The LSTM pipeline follows a chronological time-series workflow:

Historical Data
      ↓
Data Cleaning
      ↓
Chronological Train/Test Split
      ↓
Feature Scaling
      ↓
Lookback Sequence Creation
      ↓
LSTM Training
      ↓
Prediction
      ↓
Inverse Scaling
      ↓
Evaluation

The project uses an approximately 80/20 chronological train/test split.

A lookback window of 30 observations is used for the LSTM models.

Min-Max Scaling
$$ X_{scaled} = \frac{X-X_{min}}{X_{max}-X_{min}} $$

The scaler is fitted on the training data before transforming the test data.

📈 XGBoost Baseline Results

The prototype XGBoost model achieved approximately:

Metric	Result
MAE	1863.10
RMSE	2283.88
MAPE	10.46%
R²	0.596

These results represent the prototype baseline. Some supporting port, commodity, fuel, and trade features used during development contained dummy or repeated values, so these results should not be interpreted as production-level performance.

📊 Evaluation Metrics
MAE — Mean Absolute Error
$$ MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y_i}| $$

Measures the average absolute prediction error.

RMSE — Root Mean Squared Error
$$ RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2} $$

Penalizes larger prediction errors more strongly.

MAPE — Mean Absolute Percentage Error
$$ MAPE = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i-\hat{y_i}}{y_i}\right| $$

Expresses prediction error as a percentage.

R² Score

Measures the proportion of variance explained by the model relative to a baseline.

🔄 Complete Forecasting Pipeline
                     RAW DATA
                        ↓
                 Data Cleaning
                        ↓
              Feature Engineering
                        ↓
             ┌──────────┴──────────┐
             ↓                     ↓
          XGBoost                 LSTM
             ↓                     ↓
       Tabular Model        Temporal Model
             └──────────┬──────────┘
                        ↓
                 Forecast Results
                        ↓
                 Economic Analysis
                        ↓
                   Risk Analysis
                        ↓
             Decision Support Layer
                        ↓
              Chartering Support
🚢 Vessel Recommendation

Forecasting freight rates alone is not sufficient for a complete chartering decision.

PortCast combines forecast information with vessel-related information such as:

Vessel characteristics
Cargo requirements
Forecast freight rates
Voyage economics
Fuel costs
Market conditions
Risk indicators

The objective is to connect freight forecasting with the operational context of vessel chartering.

💰 Economic Analysis

The economic layer evaluates the financial implications of a potential voyage or chartering decision.

Freight Revenue
      ↓
Voyage Economics
      ↓
Fuel Costs
      ↓
Operating / Voyage Costs
      ↓
Estimated Economic Outcome

This allows the system to move from:

"What will the freight rate be?"

towards:

"What could the forecast mean for the chartering decision?"

⚠️ Risk Analysis

Freight markets are uncertain and future forecasts are not guaranteed.

PortCast considers signals such as:

Forecast movement
Historical volatility
Market conditions
Forecast uncertainty
Economic indicators

The risk layer is intended to support decisions rather than treat model predictions as guaranteed outcomes.

📊 Opportunity Score

The prototype includes an opportunity-scoring mechanism.

Example prototype output:

Current Rate:
19,506

Forecast:
23,692.83

Forecast Change:
+21.46%

Confidence:
41.27%

Opportunity Score:
74.82

Recommendation:
MONITOR

The confidence and opportunity-score mechanisms are prototype heuristics and require further statistical validation before real-world commercial use.

🏗️ System Architecture
                       ┌──────────────┐
                       │     USER     │
                       └──────┬───────┘
                              ↓
                   ┌────────────────────┐
                   │   React Frontend   │
                   └─────────┬──────────┘
                             ↓
                   ┌────────────────────┐
                   │   FastAPI Backend  │
                   └─────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
        Market Data    Forecast Engine   Decision Engine
                              │
                       ┌──────┴──────┐
                       ↓             ↓
                    XGBoost         LSTM
                       │             │
                       └──────┬──────┘
                              ↓
                      Forecast Results
                              ↓
                      Economic Analysis
                              ↓
                        Risk Analysis
                              ↓
                   Decision Support
🧰 Technology Stack
Machine Learning
Python
Pandas
NumPy
Scikit-learn
XGBoost
TensorFlow
Keras
LSTM
Joblib
Backend
FastAPI
Uvicorn
Python
REST APIs
Frontend
React
JavaScript
Axios
Vite
Data Processing
Pandas
NumPy
Time-Series Feature Engineering
Min-Max Scaling
📁 Project Structure
PortCast/
│
├── data/
│   ├── freight/
│   ├── ports/
│   ├── commodities/
│   ├── trade/
│   └── fuel/
│
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── xgboost_model.ipynb
│   └── lstm_model.ipynb
│
├── models/
│   ├── lstm_7d.keras
│   ├── lstm_30d.keras
│   ├── lstm_60d.keras
│   ├── lstm_90d.keras
│   ├── X_scaler.pkl
│   └── y_scaler.pkl
│
├── backend/
│   ├── main.py
│   ├── routes/
│   │   ├── market.py
│   │   ├── forecast.py
│   │   ├── vessel.py
│   │   ├── economics.py
│   │   └── decision.py
│   │
│   └── services/
│
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
│
├── requirements.txt
├── .gitignore
└── README.md
🔌 API Architecture
Market
GET /market/current

Returns current market information.

Forecast
GET /forecast

Returns freight-rate forecasts generated by the ML models.

Vessel Recommendation
GET /vessel/recommend

Provides vessel-related decision support.

Economics
GET /economics

Provides economic and voyage-related calculations.

Decision
GET /decision

Combines forecasting, economics, vessel information, and risk signals into a decision-support output.

💾 Model Saving

LSTM models can be saved separately for each forecasting horizon:

model.save("models/lstm_7d.keras")
model.save("models/lstm_30d.keras")
model.save("models/lstm_60d.keras")
model.save("models/lstm_90d.keras")

Scalers can be saved using Joblib:

import joblib

joblib.dump(X_scaler, "models/X_scaler.pkl")
joblib.dump(y_scaler, "models/y_scaler.pkl")
🔍 Model Comparison
Model	Primary Use	Main Strength
XGBoost	Tabular forecasting	Nonlinear feature relationships
LSTM	Time-series forecasting	Sequential and temporal patterns
🧠 Machine Learning Concepts Demonstrated
Time-Series Forecasting
Supervised Learning
Regression
XGBoost
LSTM Networks
Feature Engineering
Lag Features
Rolling Statistics
Min-Max Scaling
Chronological Train/Test Splitting
Model Evaluation
Multi-Horizon Forecasting
Decision Intelligence
🧪 Current Limitations

PortCast is currently a research and prototype system.

1. External Data Quality

Some port, commodity, fuel, and trade features used during prototyping contained dummy or repeated values.

Therefore, their contribution to predictive performance should not be interpreted as fully validated real-world predictive power.

2. Forecast Horizons

The forecasting targets were generated using:

shift(-7)
shift(-30)
shift(-60)
shift(-90)

These represent future observations in the dataset.

3. Opportunity Score

The current opportunity score and confidence logic are prototype heuristics.

A production implementation would require:

Statistical validation
Historical backtesting
Confidence calibration
Domain-expert validation
4. Production Deployment

Before production deployment, the system would require:

Larger datasets
Reliable real-time data
Complete vessel information
Robust backtesting
Data-quality validation
Model monitoring
Data-drift detection
Production-grade APIs
🔮 Future Improvements
📊 Data
Live freight-rate feeds
Real-time port congestion data
AIS vessel tracking
Real-time bunker fuel prices
Improved commodity datasets
Improved international trade datasets
Weather and maritime-condition data
🤖 Machine Learning
Temporal Fusion Transformers
Transformer-based time-series models
N-BEATS
Ensemble forecasting
Probabilistic forecasting
Prediction intervals
Automated hyperparameter optimization
Model stacking
🚢 Decision Intelligence
Advanced vessel-cargo matching
Voyage-level profitability optimization
Route optimization
Scenario analysis
Dynamic risk scoring
Uncertainty-aware recommendations
⚙️ Production
Docker deployment
Cloud deployment
MLflow experiment tracking
Automated model retraining
Model monitoring
Data-drift detection
Automated ETL pipelines
Real-time inference
🏆 Project Highlights
🚢 Dry Bulk Freight Forecasting
📈 Multi-Horizon Time-Series Forecasting
🤖 XGBoost + LSTM
📊 Time-Series Feature Engineering
🌍 International Trade Intelligence
🇮🇳 Port & Cargo Intelligence
🛢️ Fuel Economics
💰 Voyage Economics
⚠️ Risk-Aware Decision Support
🚢 Vessel Recommendation
⚡ FastAPI Backend
⚛️ React Frontend
💡 Core Idea

PortCast transforms raw shipping data into actionable decision intelligence.

             DATA
               ↓
         INFORMATION
               ↓
           FORECAST
               ↓
      ECONOMIC ANALYSIS
               ↓
             RISK
               ↓
       DECISION SUPPORT

The larger objective is to connect:

Freight Forecast
      +
Vessel Information
      +
Cargo Information
      +
Voyage Economics
      +
Risk Analysis
      ↓
Chartering Decision Support
📚 End-to-End Workflow
1. Collect Freight Market Data
                ↓
2. Collect Port & Cargo Data
                ↓
3. Collect Commodity Data
                ↓
4. Collect Trade Data
                ↓
5. Collect Fuel/Economic Indicators
                ↓
6. Clean and Integrate Datasets
                ↓
7. Exploratory Data Analysis
                ↓
8. Create Temporal Features
                ↓
9. Create Lag Features
                ↓
10. Create Rolling Statistics
                ↓
11. Create Future Forecast Targets
                ↓
12. Chronological Train/Test Split
                ↓
13. Train XGBoost Baseline
                ↓
14. Train LSTM Models
                ↓
15. Evaluate Forecasts
                ↓
16. Generate Forecast
                ↓
17. Economic Analysis
                ↓
18. Risk Analysis
                ↓
19. Decision Support
                ↓
20. API Integration
                ↓
21. Frontend Visualization
🚀 Installation
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/PortCast.git
cd PortCast
2. Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate
Linux / macOS
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Start FastAPI Backend
uvicorn backend.main:app --reload

Backend:

http://127.0.0.1:8000

API documentation:

http://127.0.0.1:8000/docs
5. Start React Frontend
cd frontend
npm install
npm run dev
🔐 Security

Never commit API keys, passwords, or environment variables.

Recommended .gitignore:

.env
venv/
__pycache__/
*.pyc
📌 Project Status
Status: Prototype / Research Project

✔ Freight-rate data processing
✔ Time-series feature engineering
✔ XGBoost baseline
✔ LSTM forecasting
✔ Multi-horizon forecasting
✔ Economic analysis
✔ Risk/decision layer
✔ Vessel recommendation concept
✔ FastAPI backend
✔ React frontend

🔄 Further validation and production data integration required
🎓 What I Learned

Through PortCast, I worked on:

Machine Learning
Time-series forecasting
Feature engineering
XGBoost
LSTM
Model evaluation
Data preprocessing
Data Engineering
Combining multiple datasets
Data cleaning
Temporal feature creation
Lag-based feature engineering
Rolling statistics
Backend Development
FastAPI
REST API development
Modular backend architecture
Machine-learning model serving
Frontend Development
React
API integration
Data visualization
Decision-support interfaces
System Design
End-to-end ML pipelines
Model integration
Forecasting services
Economic analysis
Risk-aware decision support
🌟 Why PortCast?

Traditional forecasting asks:

"What might the freight rate be?"

PortCast attempts to go further:

"What might the freight rate be?"
              +
"How could it affect voyage economics?"
              +
"What vessel and cargo context matters?"
              +
"What risk signals should be considered?"
              ↓
"How can this information support a chartering decision?"

Therefore, PortCast is designed as a decision-support system, rather than only a standalone forecasting model.

⚠️ Disclaimer

PortCast is a research and prototype decision-support system.

Forecasts and recommendations are not guaranteed future market outcomes and should not be considered financial, investment, or commercial advice.

Production deployment would require reliable real-time datasets, rigorous historical backtesting, uncertainty calibration, domain-expert validation, and continuous model monitoring.
