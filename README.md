# 🛠️ ModelForge: No-Code AutoML Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://model-forge-1.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-purple)

**ModelForge** is a modular, end-to-end Machine Learning SaaS application that empowers users to build, train, evaluate, and download models without writing a single line of code. It streamlines the entire ML lifecycle from data ingestion to deployment.

🔗 **[Live Demo: Click Here to Try ModelForge](https://model-forge-1.streamlit.app/)**

---

## 🚀 Key Features

* **📂 Drag-and-Drop Interface:** Instantly upload CSV datasets and view automated data summaries.
* **⚙️ Intelligent Preprocessing:**
    * **Auto-detection** of Numerical and Categorical columns.
    * **Imputation:** Median strategy for numbers, Most Frequent for categories.
    * **Encoding & Scaling:** Automatic One-Hot Encoding and Standard Scaling using Scikit-Learn Pipelines.
* **🔬 Advanced EDA:**
    * Interactive **Correlation Heatmap** to visualize feature relationships (cached for performance).
    * Data distribution previews.
* **🧠 Comprehensive Model Training:**
    * **Regression:** Linear Regression, Ridge, Lasso, ElasticNet, KNN, Decision Tree, Random Forest, XGBoost.
    * **Classification:** Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost, SVC, **Naive Bayes**.
* **⚡ Real-Time Monitoring:** Live training logs streaming RMSE, MAE, and R2 scores as models finish.
* **💾 Deployment Ready:**
    * **Leaderboard:** Rank models by performance metrics.
    * **Download Assets:** One-click download for trained `.pkl` models **AND** the fitted `preprocessor.pkl` (essential for real-world usage).

---
<img width="1360" height="640" alt="image" src="https://github.com/user-attachments/assets/b5647a61-3bee-4ba6-82c3-b0d92f8174da" />


## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Visualization:** Matplotlib, Seaborn
* **Architecture:** Modular Python (Separation of concerns: UI, Logic, Models, Evaluation)

---

## 📂 Project Structure

The project follows a modular architecture to ensure scalability and maintainability:

```text
ModelForge/
│
├── app.py              # Main Streamlit application (UI & Logic)
├── preprocessor.py     # Data cleaning & transformation pipelines
├── models.py           # Dictionary of ML algorithms (Regression & Classification)
├── evaluate.py         # Metrics calculation (RMSE, R2, MAE)
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
