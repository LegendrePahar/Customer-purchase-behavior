# 🛒 Predicting Customer Purchase Behaviour

An end-to-end machine learning project that predicts whether a customer will make a purchase, based on demographic and behavioral features. Includes full EDA, model training/evaluation, and a live interactive Streamlit deployment.

**Live app:** https://customer-purchase-behavior-mcf396ffl4fryn9jwrzwq3.streamlit.app/

## Dataset

[Predict Customer Purchase Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset) (Kaggle) — 1,500 customer records with 8 features and a binary purchase outcome.

| Feature | Description |
|---|---|
| Age | Customer age |
| Gender | 0 = Female, 1 = Male |
| AnnualIncome | Annual income (USD) |
| NumberOfPurchases | Total past purchases |
| ProductCategory | 0: Electronics, 1: Clothing, 2: Home Goods, 3: Beauty, 4: Sports |
| TimeSpentOnWebsite | Minutes spent per session |
| LoyaltyProgram | 0 = not enrolled, 1 = enrolled |
| DiscountsAvailed | Number of discounts used |
| **PurchaseStatus** | **Target** — 0 = no purchase, 1 = purchase |

## Project Structure

```
customer-purchase-behavior/
├── app.py                              # Streamlit deployment app
├── requirements.txt                    # Python dependencies
├── Customer_Purchase_Behavior.ipynb    # Full EDA + model training notebook
├── data/
│   └── customer_purchase_data.csv
├── models/
│   ├── purchase_model.pkl              # Trained Random Forest model
│   └── feature_names.pkl
└── outputs/                            # Saved EDA charts
```

## Approach

1. **EDA** — checked for nulls (none), examined class balance (57/43), feature distributions, and correlation with the target.
2. **Preprocessing** — train/test split (80/20, stratified), feature scaling for the linear baseline.
3. **Modeling** — compared Logistic Regression vs. Random Forest.
4. **Result** — Random Forest selected as final model: **93% accuracy**, 0.91–0.94 F1-score on held-out test data. Strongest predictors: `LoyaltyProgram`, `DiscountsAvailed`, `TimeSpentOnWebsite`, `Age`.
5. **Deployment** — Streamlit app takes a customer profile as input and returns a live purchase-probability prediction with supporting visualizations.

## Run Locally

```bash
git clone https://github.com/LegendrePahar/customer-purchase-behavior.git
cd customer-purchase-behavior
pip install -r requirements.txt
streamlit run app.py
```

## Reproduce the Model

Open `Customer_Purchase_Behavior.ipynb` in Jupyter to rerun the full EDA and training pipeline from scratch.

## Tools

Python · pandas · scikit-learn · Streamlit · Plotly · Git · GitHub · Streamlit Community Cloud
