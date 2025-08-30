# 🏠 House Price Prediction using Linear Regression

A simple **Machine Learning project** that predicts house prices based on key features such as **square footage, number of bedrooms, and number of bathrooms** using **Linear Regression**.

---

## 📌 Project Overview
The aim of this project is to build a **Linear Regression Model** that can estimate the price of a house given its features.  
This project is beginner-friendly and helps in understanding:
- Data preprocessing
- Feature selection
- Training a linear regression model
- Model evaluation

---

## 📊 Dataset
For this project, we use a housing dataset containing features such as:
- **Square Footage** (`LotArea`)
- **Number of Bedrooms** (`BedroomAbvGr`)
- **Number of Bathrooms** (`Bathroom`)
- **Price** (`SalePrice`) → Target Variable  

## ⚙️ Tech Stack & Tools
- **Python 3.8+**
- **Libraries**:  
  - `pandas` → Data handling  
  - `numpy` → Numerical operations  
  - `matplotlib` → Visualization  
  - `scikit-learn` → Machine Learning  

## 🚀 Implementation Steps
1. Import and preprocess dataset  
2. Select features (`LotArea`, `BedroomAbvGR`, `Bathrooms`)  
3. Split data into **training** and **testing sets**  
4. Train Linear Regression model using **scikit-learn**  
5. Evaluate performance using **R² Score** and **Mean Squared Error (MSE)**  
6. Visualize predictions vs actual prices  
ore:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
