# Energy Consumption Prediction using Machine Learning

This project predicts **hourly energy consumption** using various **machine learning regression models**.  
It uses the dataset from [Kaggle - Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption).

---

## 📁 Dataset
**File:** `AEP_hourly.csv`  
- Timestamp-based hourly energy consumption data.  
- Missing values handled using forward fill.  
- Features include time-based variables (hour, day, month, year, weekday).

---

## 🧠 Models Used
| Model | Description |
|--------|--------------|
| Linear Regression | Simple baseline model |
| Decision Tree | Non-linear splits |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting of weak learners |
| XGBoost | Optimized gradient boosting |
| LightGBM | Fast, efficient boosting library |
| SVR (RBF) | Support Vector Regression |
| Neural Network (MLP) | Deep learning approach |

---

## ⚙️ Model Evaluation Metrics
| Metric | Description |
|---------|--------------|
| MAE | Mean Absolute Error — average absolute prediction error |
| RMSE | Root Mean Squared Error — penalizes large errors |
| R² | Coefficient of determination (1 is perfect) |
| MAPE (%) | Mean Absolute Percentage Error |
| Accuracy (%) | Derived metric = 100 - MAPE |

---

## 📊 Model Training Logs

🔹 **Training Linear Regression...**  
Linear Regression → MAE: 1823.35, RMSE: 2276.10, R²: 0.2140  

🔹 **Training Decision Tree...**  
Decision Tree → MAE: 1596.19, RMSE: 2178.70, R²: 0.2798  

🔹 **Training Random Forest...**  
Random Forest → MAE: 1315.68, RMSE: 1714.83, R²: 0.5539  

🔹 **Training Gradient Boosting...**  
Gradient Boosting → MAE: 1330.99, RMSE: 1747.91, R²: 0.5365  

🔹 **Training XGBoost...**  
XGBoost → MAE: 1409.86, RMSE: 1860.21, R²: 0.4750  

🔹 **Training LightGBM...**  
LightGBM → MAE: 1349.19, RMSE: 1782.91, R²: 0.5177  

🔹 **Training SVR (RBF)...**  
SVR (RBF) → MAE: 1710.58, RMSE: 2223.91, R²: 0.2496  

🔹 **Training Neural Network (MLP)...**  
Neural Network (MLP) → MAE: 1594.37, RMSE: 2053.60, R²: 0.3602  

---

## 📈 Summary of Results

| Model | MAE | RMSE | R² |
|--------|------|------|----|
| Random Forest | 1315.68 | 1714.83 | 0.5539 |
| Gradient Boosting | 1330.99 | 1747.91 | 0.5365 |
| LightGBM | 1349.19 | 1782.91 | 0.5177 |
| XGBoost | 1409.86 | 1860.21 | 0.4750 |
| Neural Network (MLP) | 1594.37 | 2053.60 | 0.3602 |
| Decision Tree | 1596.19 | 2178.70 | 0.2798 |
| SVR (RBF) | 1710.58 | 2223.91 | 0.2496 |
| Linear Regression | 1823.35 | 2276.10 | 0.2140 |

---

## 🧩 Tech Stack
- Python (3.9+)
- Libraries: pandas, numpy, matplotlib, scikit-learn, xgboost, lightgbm
- IDE: PyCharm / Jupyter Notebook

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/energy-consumption-prediction.git
   cd energy-consumption-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python energy_prediction.py
   ```
4. View the model results printed in the console.

---

## 📈 Future Improvements
- Add hyperparameter tuning (GridSearchCV / Optuna)
- Incorporate weather data (temperature, humidity)
- Deploy as a web dashboard (Streamlit or Flask)

---

## Author
**Triston Aloyssius Marta** — Data Science & Statistics
📧 Contact: tristonmarta@yahoo.com.sg
