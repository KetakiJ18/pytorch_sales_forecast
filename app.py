import streamlit as st
import pandas as pd

from utils import (
    load_model,
    load_scalers,
    load_data,
    build_future_row,
    get_history,
    predict_sales
)

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------

st.set_page_config(
    page_title="Store Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------------
# Cache Expensive Objects
# -------------------------------------------------------

@st.cache_resource
def get_model():
    return load_model()

@st.cache_resource
def get_scalers():
    return load_scalers()

@st.cache_data
def get_dataset():
    return load_data()

# -------------------------------------------------------
# Load Resources
# -------------------------------------------------------

model = get_model()

x_scaler, y_scaler = get_scalers()

df = get_dataset()

# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.title("📈 Store Sales Forecasting using LSTM")

st.markdown(
    """
Predict the sales for a future date using the previous **30 days**
of historical sales along with future features like promotion,
holiday and store.
"""
)

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

st.sidebar.header("Prediction Inputs")

store_number = st.sidebar.selectbox(
    "Select Store",
    options=list(range(1,11)),
    format_func=lambda x: f"Store {x}"
)

prediction_date = st.sidebar.date_input(
    "Prediction Date"
)

promo = st.sidebar.radio(
    "Promotion",
    options=["No", "Yes"]
)

holiday = st.sidebar.radio(
    "Holiday",
    options=["No", "Yes"]
)

predict_button = st.sidebar.button(
    "Predict Sales",
    use_container_width=True
)

# -------------------------------------------------------
# Build Future Row
# -------------------------------------------------------

future_row = build_future_row(
    store_number,
    prediction_date,
    promo,
    holiday
)

# -------------------------------------------------------
# Get Previous 30 Days
# -------------------------------------------------------

try:

    history_df = get_history(
        df,
        store_number,
        prediction_date
    )

except ValueError as e:

    st.error(str(e))

    st.stop()

# -------------------------------------------------------
# Display Information
# -------------------------------------------------------

left, right = st.columns([2,1])

with left:

    st.subheader("Previous 30 Days Sales")

    chart = history_df[["date","sales"]].set_index("date")

    st.line_chart(chart)

with right:

    st.subheader("Selected Inputs")

    st.write(f"**Store:** Store {store_number}")

    st.write(f"**Prediction Date:** {prediction_date}")

    st.write(f"**Promotion:** {'Yes' if promo else 'No'}")

    st.write(f"**Holiday:** {'Yes' if holiday else 'No'}")

    st.write(f"**History Available:** {len(history_df)} Days")

# -------------------------------------------------------
# Prediction
# -------------------------------------------------------

if predict_button:

    prediction = None

    try:

        prediction = predict_sales(
            model=model,
            history_df=history_df,
            future_row=future_row,
            x_scaler=x_scaler,
            y_scaler=y_scaler
        )

        avg_sales = history_df["sales"].mean()

        difference = prediction - avg_sales

        percentage_change = (
            difference / avg_sales
        ) * 100 if avg_sales != 0 else 0

        st.divider()

        st.header("Prediction Results")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric(
            "Predicted Sales",
            f"{prediction:,.2f}"
        )

        metric2.metric(
            "30-Day Average",
            f"{avg_sales:,.2f}"
        )

        metric3.metric(
            "Change",
            f"{percentage_change:.2f}%",
            delta=f"{difference:.2f}"
        )

        st.success("Prediction completed successfully!")

        # Chart only after successful prediction
        chart_df = history_df[["date", "sales"]].copy()

        prediction_row = pd.DataFrame({
            "date": [pd.Timestamp(prediction_date)],
            "sales": [prediction]
        })


        with st.expander("Prediction Details"):

            st.write("### Future Features")

            st.json(future_row)

            st.write("### Previous 30 Days")

            st.dataframe(history_df)

    except Exception as e:

        st.error(f"Prediction Failed\n\n{e}")