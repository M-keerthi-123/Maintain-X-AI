import pandas as pd
import numpy as np
import os, joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# 1️⃣ LOAD DATA
# =====================================================

telemetry = pd.read_csv("../data/PdM_telemetry.csv", parse_dates=["datetime"])
failures  = pd.read_csv("../data/PdM_failures.csv", parse_dates=["datetime"])


failures["failure"] = 1

data = (telemetry
        .merge(failures[["machineID","datetime","failure"]],
               on=["machineID","datetime"], how="left")
       
        .sort_values(["machineID","datetime"])
        .fillna({"failure":0})
       )



# =====================================================
# 3️⃣ FEATURES
# =====================================================

features = [
    "volt","rotate","pressure","vibration"
]

X = data[features]
y = data["failure"]

# =====================================================
# 4️⃣ SCALING
# =====================================================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

os.makedirs("models",exist_ok=True)

joblib.dump(scaler,"models/scaler.pkl")

# =====================================================
# 5️⃣ XGBOOST MODEL
# =====================================================

X_train,X_test,y_train,y_test = train_test_split(
    X_scaled,y,test_size=0.2,random_state=42
)

pos_weight = (y_train==0).sum()/(y_train==1).sum()

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    scale_pos_weight=pos_weight,
    n_jobs=-1,
    tree_method="hist",
    early_stopping_rounds=10
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test,y_test)],
    verbose=False
)

print(classification_report(y_test,xgb_model.predict(X_test)))

joblib.dump(xgb_model,"models/xgboost_model.pkl")

# =====================================================
# 6️⃣ HEALTH SCORE
# =====================================================

failure_prob = xgb_model.predict_proba(X_scaled)[:,1]

data["failure_probability"] = failure_prob
data["health_score"] = (1-failure_prob)*100

print("Health score generated")

# =====================================================
# 7️⃣ LSTM DATASET
# =====================================================

seq_len = 30
X_lstm,y_lstm = [],[]

for machine,group in data.groupby("machineID"):

    X_values = scaler.transform(group[features])
    y_values = group["failure_probability"].values

    for i in range(len(group)-seq_len):

        X_lstm.append(X_values[i:i+seq_len])
        y_lstm.append(y_values[i+seq_len])

X_lstm = np.array(X_lstm)
y_lstm = np.array(y_lstm)

print("LSTM shape:",X_lstm.shape)

split = int(0.8*len(X_lstm))

X_train_lstm,X_test_lstm = X_lstm[:split],X_lstm[split:]
y_train_lstm,y_test_lstm = y_lstm[:split],y_lstm[split:]

# =====================================================
# 8️⃣ LSTM MODEL
# =====================================================

lstm_model = Sequential([
    LSTM(64,return_sequences=True,input_shape=(seq_len,len(features))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16,activation="relu"),
    Dense(1,activation="sigmoid")
])

lstm_model.compile(
    optimizer=Adam(0.001),
    loss="binary_crossentropy"
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

lstm_model.fit(
    X_train_lstm,y_train_lstm,
    epochs=15,
    batch_size=128,
    validation_data=(X_test_lstm,y_test_lstm),
    callbacks=[early_stop],
    verbose=1
)

lstm_model.save("models/lstm_model.h5")

print("🎯 TRAINING COMPLETED SUCCESSFULLY")