# 📊 Retail Sales Forecasting System (MicroGCC Case Study)

---

## 📌 Project Overview
This project is an end-to-end **machine learning-based time series forecasting system** designed to predict future retail sales across multiple states in the United States.

It applies multiple forecasting approaches including both statistical and deep learning models to analyze historical sales data and generate accurate predictions.

The system provides:
- State-wise future sales forecasts
- Model performance comparison
- Visualization of predictions
- REST API for predictions

---

## 🎯 Problem Statement
Retail businesses need accurate forecasting of future sales to:
- Optimize inventory
- Improve decision-making
- Reduce losses
- Understand regional demand patterns

This project addresses this using machine learning and deep learning models.

---

## 🧠 Models Used
The following models were implemented and compared:

- SARIMA (Statistical Time Series Model)
- Prophet (Meta/Facebook Forecasting Model)
- XGBoost Regressor (Machine Learning Model)
- LSTM (Long Short-Term Memory Neural Network)

---

## 📁 Project Structure


microgcc_case_study_final/
│
├── code/
│ ├── app.py # Flask API
│ ├── main.py # Model training & forecasting
│ └── data/
│ └── sales_data.xlsx
│
├── outputs/
│ ├── models/ # Trained models (.h5, .keras, .pkl)
│ ├── plots/ # Forecast visualizations
│ ├── future_forecasts/ # Predicted CSV files
│ ├── model_results.csv # Evaluation metrics
│ └── rmse_comparison.png # Model comparison chart
│
├── screenshots/
│ ├── home_api.png
│ ├── api_terminal.png
│ └── forecast_api_texas.png
│
├── deliverables/
│ ├── report.pdf # Final project report
│ ├── ppt.pptx # Presentation slides
│ ├── demo1.mp4 # Model execution demo
│ └── demo2.mp4 # API & prediction demo
│
├── requirements.txt
└── README.md


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/22K61A05B1/microgcc_case_study_final.git
cd microgcc_case_study_final
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Run the forecasting model
python code/main.py
4️⃣ Run the Flask API
python code/app.py
5️⃣ Access API in browser
http://127.0.0.1:5000/predict/Texas
📊 Outputs Generated

The system generates the following outputs:

Future sales predictions (CSV files per state)
Trained ML/DL models
Forecast graphs for each state
RMSE comparison between models
API response results
📸 Screenshots

Available inside the screenshots/ folder:

API terminal execution
Home API response
Forecast visualization output
🎥 Demo Videos

Available inside the deliverables/ folder:

Demo 1: Model training & execution flow
Demo 2: API prediction & output demonstration
📈 Results Summary
Successfully implemented multi-model forecasting system
LSTM and XGBoost showed strong performance on complex patterns
SARIMA and Prophet provided baseline comparisons
Accurate state-wise predictions generated
RMSE used for model evaluation
🚀 Key Features
Multi-model forecasting system
State-wise prediction capability
REST API integration using Flask
Automated ML pipeline
Data visualization and comparison
Scalable architecture for deployment
🧾 Conclusion

This project demonstrates a complete end-to-end machine learning pipeline for time series forecasting, integrating statistical models and deep learning approaches to achieve accurate and scalable predictions.

It is suitable for real-world business forecasting applications.

🔗 GitHub Repository

https://github.com/22K61A05B1/microgcc_case_study_final



👩‍💻 Author

Srija Kiran Nadipudi
Python Full Stack & Data Science Developer