# 🚗 Vehicle Insurance Churn Predictor — End-to-End MLOps Pipeline
---

## 📌 Project Overview

**Vehicle Insurance Churn Predictor** is a production-grade, end-to-end **Machine Learning Operations (MLOps)** pipeline that predicts whether a health insurance policyholder will also be interested in purchasing **vehicle insurance**.

The system automates the entire ML lifecycle — from raw data ingestion out of MongoDB Atlas, through validation, transformation, model training, evaluation, and deployment — all triggered automatically on every `git push` via **GitHub Actions**.

The best model between **XGBoost** and **LightGBM** is automatically selected based on F1-score and served through an interactive **Streamlit** dashboard.

---

## 🎯 Problem Statement

An insurance company wants to identify which of its existing health insurance customers are likely to purchase vehicle insurance. This is a **binary classification** problem:

- `1` → Customer is **interested** in vehicle insurance
- `0` → Customer is **not interested**

Accurately predicting this helps the company target the right customers, reducing marketing costs and increasing conversion rates.

---

## 🧰 Tech Stack

| Category | Tool |
|---|---|
| Language | Python 3.10 |
| ML Models | XGBoost, LightGBM |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Class Balancing | SMOTEENN (imbalanced-learn) |
| Database | MongoDB Atlas |
| Model Storage | DagsHub |
| UI | Streamlit |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Deployment | Render |

---

## 📊 Dataset

**Source:** [Health Insurance Cross Sell Prediction — Kaggle](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction)

| Feature | Description |
|---|---|
| Gender | Male / Female |
| Age | Age of the customer |
| Driving_License | 1 = has license, 0 = does not |
| Region_Code | Unique code for customer region |
| Previously_Insured | 1 = already has vehicle insurance |
| Vehicle_Age | Age of the vehicle |
| Vehicle_Damage | Yes = damaged before, No = never |
| Annual_Premium | Amount paid for health insurance annually |
| Policy_Sales_Channel | Channel through which customer was contacted |
| Vintage | Number of days customer has been associated |
| **Response** | **1 = Interested, 0 = Not Interested (Target)** |

**Dataset Size:** 381,109 rows × 11 columns

**Class Imbalance:** ~88% not interested (0), ~12% interested (1) — handled using SMOTEENN

---

## 👤 Author

**Aamir Raza**
- GitHub: [@AamirRaza01](https://github.com/AamirRaza01)
- LinkedIn: [aamirraza01](https://linkedin.com/in/aamirraza01)

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## ⭐ If you found this project helpful, please give it a star!