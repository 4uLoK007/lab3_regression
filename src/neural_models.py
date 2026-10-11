import numpy as np
from tensorflow.keras import layers, models, optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping


def build_single_layer(n_features, activation="relu", lr=1e-3,
                       loss="mse", optimizer="adam"):
    inp = layers.Input(shape=(n_features,))
    out = layers.Dense(1, activation=activation, kernel_initializer="he_normal")(inp)
    model = models.Model(inp, out)
    opt = optimizers.get(optimizer); opt.learning_rate = lr
    model.compile(optimizer=opt, loss=loss, metrics=["mae"])
    return model


def build_mlp(n_features, hidden=(64, 32), activation="relu",
              lr=1e-3, loss="mse", optimizer="adam"):
    inp = layers.Input(shape=(n_features,))
    x = inp
    for h in hidden:
        x = layers.Dense(h, activation=activation)(x)
        x = layers.Dropout(0.1)(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    opt = optimizers.get(optimizer); opt.learning_rate = lr
    model.compile(optimizer=opt, loss=loss, metrics=["mae"])
    return model


def build_deep(n_features, loss="mse", lr=1e-3):
    inp = layers.Input(shape=(n_features,))
    x = layers.Dense(128, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=["mae"])
    return model


def build_multi_horizon_mlp(n_features, horizon=7, hidden=(128, 64),
                            activation="relu", lr=1e-3):
    inp = layers.Input(shape=(n_features,))
    x = inp
    for h in hidden:
        x = layers.Dense(h, activation=activation)(x)
    out = layers.Dense(horizon)(x)
    model = models.Model(inp, out)
    model.compile(optimizer=optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model


def train_model(model, X_tr, y_tr, X_val, y_val, epochs=100, batch_size=32):
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[es], verbose=0)
    return history


def count_params(model):
    return model.count_params()