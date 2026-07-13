import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="Customer Purchase Predictor", page_icon="🛒", layout="centered")

PRODUCT_CATEGORIES = {0: "Electronics", 1: "Clothing", 2: "Home Goods", 3: "Beauty", 4: "Sports"}
GENDER_MAP = {0: "Female", 1: "Male"}


@st.cache_resource
def load_model():
    model = joblib.load("models/purchase_model.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, feature_names


@st.cache_data
def load_reference_data():
    return pd.read_csv("data/customer_purchase_data.csv")


def main():
    st.title("🛒 Customer Purchase Behavior Predictor")
    st.write(
        "Enter a customer's profile to predict whether they are likely to make a purchase. "
        "Model: Random Forest Classifier (93% test accuracy)."
    )

    model, feature_names = load_model()
    df = load_reference_data()

    st.divider()
    st.subheader("Customer Profile")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=18, max_value=70, value=35)
        gender_label = st.selectbox("Gender", options=list(GENDER_MAP.values()))
        gender = [k for k, v in GENDER_MAP.items() if v == gender_label][0]
        annual_income = st.number_input(
            "Annual Income ($)", min_value=10000, max_value=250000, value=60000, step=1000
        )
        num_purchases = st.slider("Number of Past Purchases", min_value=0, max_value=30, value=8)

    with col2:
        category_label = st.selectbox("Preferred Product Category", options=list(PRODUCT_CATEGORIES.values()))
        product_category = [k for k, v in PRODUCT_CATEGORIES.items() if v == category_label][0]
        time_on_site = st.slider("Time Spent on Website (minutes)", min_value=0.0, max_value=60.0, value=25.0)
        loyalty = st.radio("Enrolled in Loyalty Program?", options=["No", "Yes"], horizontal=True)
        loyalty_program = 1 if loyalty == "Yes" else 0
        discounts_availed = st.slider("Discounts Availed (past)", min_value=0, max_value=10, value=2)

    st.divider()

    if st.button("Predict Purchase Likelihood", type="primary"):
        input_df = pd.DataFrame(
            [[age, gender, annual_income, num_purchases, product_category,
              time_on_site, loyalty_program, discounts_availed]],
            columns=feature_names,
        )

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.success(f"✅ Likely to Purchase — probability: {probability:.1%}")
        else:
            st.warning(f"❌ Unlikely to Purchase — probability: {probability:.1%}")

        # Probability gauge
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Purchase Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2ca02c" if prediction == 1 else "#d62728"},
                    "steps": [
                        {"range": [0, 50], "color": "#fddede"},
                        {"range": [50, 100], "color": "#dbf5db"},
                    ],
                },
            )
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature importance context
        st.subheader("What drives this prediction?")
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
        fig_imp = px.bar(
            importances, orientation="h",
            title="Model Feature Importance (overall, not customer-specific)",
            labels={"value": "Importance", "index": "Feature"},
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # Where this customer sits vs the training data
        st.subheader("How this customer compares")
        compare_col = st.selectbox(
            "Compare against training data on:",
            options=["TimeSpentOnWebsite", "AnnualIncome", "NumberOfPurchases", "Age"],
        )
        fig_hist = px.histogram(
            df, x=compare_col, color="PurchaseStatus",
            barmode="overlay", nbins=30,
            title=f"Distribution of {compare_col} (colored by actual PurchaseStatus)",
            color_discrete_map={0: "#d62728", 1: "#2ca02c"},
        )
        input_val = input_df[compare_col].values[0]
        fig_hist.add_vline(x=input_val, line_dash="dash", line_color="black",
                            annotation_text="This customer")
        st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("ℹ️ About this app"):
        st.write(
            """
            This app predicts customer purchase likelihood using a Random Forest model trained on the
            [Predict Customer Purchase Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset)
            (1,500 customer records).

            **Model performance on held-out test data:** 93% accuracy, 0.91-0.94 F1-score.

            Built with Streamlit, scikit-learn, and Plotly.
            """
        )


if __name__ == "__main__":
    main()
