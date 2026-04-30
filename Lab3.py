import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, AdamW, Nadam

import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('processed_variant_13.csv')
X = df.drop(columns=['BMI'])
y = df['BMI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
input_shape = (X_train.shape[1],)

print(f"Розмір тренувальної вибірки: {X_train.shape}")
print(f"Розмір тестової вибірки: {X_test.shape}")

model_configs = [
    {"name": "M1_16_8_4_Elu_Adam", "units": [16, 8, 4], "activation": "elu", "optimizer": "adam", "dropout": 0.0,
     "bn": False},
    {"name": "M2_16_8_4_Elu_AdamW", "units": [16, 8, 4], "activation": "elu", "optimizer": "adamw", "dropout": 0.0, "bn": False},
    {"name": "M3_16_8_4_Elu_Nadam", "units": [16, 8, 4], "activation": "elu", "optimizer": "nadam", "dropout": 0.0,
     "bn": False},

    {"name": "M4_8_8_4_Gelu_Adam", "units": [8, 8, 4], "activation": "gelu", "optimizer": "adam", "dropout": 0.0,
     "bn": False},
    {"name": "M5_8_8_4_Gelu_AdamW", "units": [8, 8, 4], "activation": "gelu", "optimizer": "adamw", "dropout": 0.0,
     "bn": False},
    {"name": "M6_8_8_4_Gelu_Nadam", "units": [8, 8, 4], "activation": "gelu", "optimizer": "nadam", "dropout": 0.0,
     "bn": False},

    {"name": "M7_8_8_4_Gelu_Adam_Dropout_Bn", "units": [8, 8, 4], "activation": "gelu", "optimizer": "adam",
     "dropout": 0.2, "bn": True},
    {"name": "M8_8_8_4_Gelu_AdamW_Dropout_Bn", "units": [8, 8, 4], "activation": "gelu", "optimizer": "adamw",
     "dropout": 0.2, "bn": True},
    {"name": "M9_8_8_4_Gelu_Nadam_Dropout_Bn", "units": [8, 8, 4], "activation": "gelu", "optimizer": "nadam",
     "dropout": 0.2, "bn": True},

    {"name": "M10_16_16_16_Relu_Adam", "units": [16, 16, 16], "activation": "relu", "optimizer": "adam", "dropout": 0.0,
     "bn": False},
    {"name": "M11_16_16_16_Relu_AdamW", "units": [16, 16, 16], "activation": "relu", "optimizer": "adamw",
     "dropout": 0.0, "bn": False},
    {"name": "M12_16_16_16_Relu_Nadam", "units": [16, 16, 16], "activation": "relu", "optimizer": "nadam",
     "dropout": 0.0, "bn": False},

    {"name": "M13_32_16_8_Gelu_AdamW_Dropout_Bn", "units": [32, 16, 8], "activation": "gelu", "optimizer": "adamw",
     "dropout": 0.2, "bn": True},
    {"name": "M14_8_4_Relu_Nadam_Dropout_Bn", "units": [8, 4], "activation": "relu", "optimizer": "nadam",
     "dropout": 0.2, "bn": True},
]


def build_model(config, input_shape):
    model = Sequential(name=config["name"])
    model.add(Input(shape=input_shape))

    # Додаємо приховані шари
    for units in config["units"]:
        model.add(Dense(units, activation=config["activation"]))
        if config["bn"]:
            model.add(BatchNormalization())
        if config["dropout"] > 0.0:
            model.add(Dropout(config["dropout"]))

    model.add(Dense(1, name='output'))

    opt_name = config["optimizer"].lower()
    lr = 0.001
    if opt_name == "adam":
        opt = Adam(learning_rate=lr)
    elif opt_name == "adamw":
        opt = AdamW(learning_rate=lr, weight_decay=0.004)
    elif opt_name == "nadam":
        opt = Nadam(learning_rate=lr)

    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model


results = []
trained_models = {}
EPOCHS = 100
BATCH_SIZE = 32

print("\nПочинаємо навчання 10 нейронних мереж...")
for config in model_configs:
    model = build_model(config, input_shape)
    print(f"Навчання {model.name}...")

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    y_pred = model.predict(X_test, verbose=0).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({'Model': model.name, 'MAE': mae, 'MSE': mse, 'R2': r2})
    trained_models[model.name] = (model, y_pred)

results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
print("\nФінальні результати нейронних мереж:")
print(results_df.to_string(index=False))


top_3_models = results_df.head(3)['Model'].tolist()

for model_name in top_3_models:
    model, y_pred = trained_models[model_name]
    r2_val = results_df[results_df['Model'] == model_name]['R2'].values[0]

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Фактичні значення (BMI)")
    plt.ylabel("Передбачені значення (BMI)")
    plt.title(f"Графік кореляції ({model_name})\n$R^2$ = {r2_val:.4f}")

    residuals = y_test - y_pred
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Передбачені значення")
    plt.ylabel("Залишки (Похибка)")
    plt.title(f"Графік залишків ({model_name})")

    plt.tight_layout()
    plt.show()