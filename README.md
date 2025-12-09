# 🛠️ ModelForge: No-Code AutoML Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-ff4b4b)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)

**ModelForge** is a modular, end-to-end Machine Learning SaaS application that allows users to build, train, and evaluate models without writing a single line of code. It streamlines the ML lifecycle from data ingestion to model deployment.

---

## 🚀 Key Features

* **📂 Drag-and-Drop Interface:** Upload any CSV dataset and instantly view data statistics.
* **⚙️ Intelligent Preprocessing:**
    * Automatic detection of numerical and categorical columns.
    * Handles missing values (Median imputation for numbers, Mode for categories).
    * One-Hot Encoding and Standard Scaling applied automatically.
* **📊 Advanced Visualization:**
    * Interactive Correlation Heatmap to understand feature relationships.
    * Automated data distribution previews.
* **🧠 Multi-Model Training:**
    * Train multiple algorithms simultaneously (Linear Regression, Random Forest, XGBoost, SVM, etc.).
    * Real-time training logs streaming directly to the UI.
    * Dynamic Leaderboard allowing you to sort models by RMSE, MAE, or R2 Score.
* **💾 Deployment Ready:**
    * One-click download for trained models (`.pkl`).
    * **Crucial:** Downloads the fitting `preprocessor.pkl` to ensure new data can be transformed correctly.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (for rapid UI development)
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Visualization:** Matplotlib, Seaborn
* **Architecture:** Modular Python Scripts (`app.py`, `models.py`, `preprocessor.py`)

---

## 📂 Project Structure

The project follows a modular architecture to ensure scalability and maintainability:

```text
ModelForge/
│
├── app.py              # Main Streamlit application (UI & Logic)
├── preprocessor.py     # Data cleaning & transformation pipelines
├── models.py           # Dictionary of ML algorithms (Regression & Classification)
├── evaluate.py         # Metrics calculation (RMSE, R2, MAE, etc.)
├── requirements.txt    # Project dependencies
└── README.md           # Documentation