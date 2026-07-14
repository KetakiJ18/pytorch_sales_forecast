import torch
import joblib
import numpy as np
import pandas as pd

from model import LSTMModel


INPUT_SIZE = 17
FUTURE_INPUT_SIZE = 16

HIDDEN_SIZE = 128
N_LAYERS = 1
DROPOUT = 0.2866623312653806
WINDOW_SIZE = 30

MODEL_PATH = "best_model_v2.pth"
X_SCALER_PATH = "scaler_X.pkl"
Y_SCALER_PATH = "scaler_y.pkl"
DATA_PATH = "store_sales.csv"

def load_data():

    df = pd.read_csv(
        DATA_PATH,
    )

    df["date"] = pd.to_datetime(
        df["date"],
        format="%d-%m-%Y"
    )


    df = df.sort_values(
        ["store", "date"]
    ).reset_index(drop=True)

    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    df["is_weekend"] = (
        df["dayofweek"] >= 5
    ).astype(int)

    # load saved encoder
    ohe = joblib.load("store_encoder.pkl")

    encoded = ohe.transform(
        df[["store"]]
    )

    store_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(["store"]),
        index=df.index
    )

    df = pd.concat(
        [
            df.drop(columns=["store"]),
            store_df
        ],
        axis=1
    )

    return df

def load_model():

    model = LSTMModel(
        input_size=INPUT_SIZE,
        future_input_size=FUTURE_INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        n_layers=N_LAYERS,
        dropout=DROPOUT
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only = False
        )
    )

    model.eval()

    return model

def load_scalers():

    x_scaler = joblib.load(X_SCALER_PATH)

    y_scaler = joblib.load(Y_SCALER_PATH)

    return x_scaler, y_scaler

FEATURE_COLUMNS = [

    "sales",

    "promo",

    "holiday",

    "dayofweek",

    "month",

    "day",

    "is_weekend",

    "store_1",
    "store_2",
    "store_3",
    "store_4",
    "store_5",
    "store_6",
    "store_7",
    "store_8",
    "store_9",
    "store_10"

]

def build_future_row(
    store_number,
    prediction_date,
    promo,
    holiday
):

    row = {}

    row["sales"] = 0

    row["promo"] = 1 if promo == "Yes" else 0
    row["holiday"] = 1 if holiday == "Yes" else 0

    row["day"] = prediction_date.day

    row["month"] = prediction_date.month

    row["dayofweek"] = prediction_date.weekday()

    row["is_weekend"] = int(
        prediction_date.weekday() >= 5
    )

    for i in range(1,11):

        row[f"store_{i}"] = 0

    row[f"store_{store_number}"] = 1

    return row


def get_history(
    df,
    store_number,
    prediction_date,
    window_size=WINDOW_SIZE
):

    store_col = f"store_{store_number}"

    history = df[
        (df[store_col] == 1) &
        (df["date"] < pd.Timestamp(prediction_date))
    ]

    history = (
        history
        .sort_values("date")
        .tail(window_size)
    )

    if len(history) != window_size:
        raise ValueError(
            f"Need {window_size} previous days of data, found {len(history)}."
        )

    return history

def predict_sales(
    model,
    history_df,
    future_row,
    x_scaler,
    y_scaler
):

    # Ensure exact column order
    history = history_df[FEATURE_COLUMNS].copy()

    history.columns = history.columns.astype(str)

    # Debug
    print("History columns:")
    print(history.columns.tolist())

    print("Scaler columns:")
    print(x_scaler.feature_names_in_.tolist())

    history_scaled = x_scaler.transform(history)

    history_tensor = torch.tensor(
        history_scaled,
        dtype=torch.float32
    ).unsqueeze(0)


    future_df = pd.DataFrame([future_row])

    # IMPORTANT: same order as training
    future_df = future_df[FEATURE_COLUMNS].copy()

    future_df.columns = future_df.columns.astype(str)

    print("Future columns:")
    print(future_df.columns.tolist())


    future_scaled = x_scaler.transform(
        future_df
    )


    sales_index = FEATURE_COLUMNS.index("sales")

    future_scaled = np.delete(
        future_scaled,
        sales_index,
        axis=1
    )


    future_tensor = torch.tensor(
        future_scaled,
        dtype=torch.float32
    )


    with torch.no_grad():

        prediction_scaled = model(
            history_tensor,
            future_tensor
        )


    prediction = y_scaler.inverse_transform(
        prediction_scaled.cpu().numpy()
    )

    return prediction[0][0]