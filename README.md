# ⚖️ Odisha Crime Analytics & Prediction Dashboard

This Streamlit dashboard visualizes and predicts IPC crimes across districts of Odisha.

## 🧠 Features
- Upload CSV dataset of district-wise IPC crimes.
- Visualize crime patterns via dynamic bar plots.
- Predict *Total Cognizable IPC Crimes* using ML models (Linear Regression, Decision Tree, Random Forest).
- User enters year and district for custom predictions.

## 📁 Dataset Format
| District | Murder | Theft | Dowry Deaths | Riots | Total Cognizable IPC Crimes |
|-----------|--------|--------|---------------|--------|-----------------------------|
| Cuttack   | 90     | 230    | 45            | 22     | 3600 |
| Puri      | 40     | 190    | 26            | 18     | 2200 |

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
