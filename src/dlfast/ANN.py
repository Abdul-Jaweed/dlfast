import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping

def ANN(X, y, task='binary', scaler='standard', epochs=10, optimizer='Adam', activation='relu', layers=(100, 50, 3), early_stop=True, dropout=None, save=False):
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Data preprocessing with chosen scaler
    if scaler == 'standard':
        scaler_obj = StandardScaler()
    elif scaler == 'robust':
        scaler_obj = RobustScaler()
    elif scaler == 'minmax':
        scaler_obj = MinMaxScaler()
    else:
        raise ValueError("Invalid scaler name. Use 'standard', 'robust', or 'minmax'.")

    X_train = scaler_obj.fit_transform(X_train)
    X_test = scaler_obj.transform(X_test)

    # Create the neural network model
    model = Sequential()
    for i, units in enumerate(layers):
        if i == 0:
            # Input layer
            model.add(Dense(units, activation=activation.split('/')[0], input_shape=(X_train.shape[1],)))
        else:
            # Hidden layers
            model.add(Dense(units, activation=activation.split('/')[0]))
            # Add dropout regularization to hidden layers
            if dropout is not None and dropout > 0:
                model.add(Dropout(dropout))

    if task == 'binary':
        # Binary classification
        model.add(Dense(1, activation='sigmoid'))
        loss = 'binary_crossentropy'
    elif task == 'multiclass':
        # Multiclass classification
        model.add(Dense(len(set(y_train)), activation='softmax'))
        loss = 'sparse_categorical_crossentropy'
    elif task == 'regression':
        # Regression
        model.add(Dense(1, activation='linear'))
        loss = 'mean_squared_error'
    else:
        raise ValueError("Invalid task name. Use 'binary', 'multiclass', or 'regression'.")

    # Compile the model with the chosen optimizer and loss function
    if optimizer == 'Adam':
        optimizer = Adam()
    elif optimizer == 'SGD':
        optimizer = SGD()
    else:
        raise ValueError("Invalid optimizer name. Use 'Adam' or 'SGD'.")

    model.compile(optimizer=optimizer, loss=loss)

    # Define early stopping callback if early_stop is True
    callbacks = []
    if early_stop:
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        callbacks.append(early_stopping)

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, validation_split=0.1, callbacks=callbacks)

    # Evaluate the model on the test set
    evaluation_results = model.evaluate(X_test, y_test)

    # Save the model as an h5 file if save is True
    if save:
        model.save('ANN.h5')

    return model, evaluation_results


# Example usage:
# Assuming you have your X and y data loaded from your dataset
# X, y = load_data()

# Create and train the neural network with StandardScaler, Adam optimizer, relu activation function for input and hidden layers, sigmoid activation function for the output layer (for binary classification), early stopping, dropout, and save the model as ANN.h5 with dropout rate of 0.2
# model, results = ANN(X, y, task='binary', scaler='standard', epochs=20, optimizer='Adam', activation='relu/sigmoid', early_stop=True, dropout=0.2, save=True)

# For multiclass classification with softmax activation for the output layer, use:
# model, results = ANN(X, y, task='multiclass', scaler='standard', epochs=20, optimizer='Adam', activation='relu/softmax', early_stop=True, dropout=0.2, save=True)

# For regression with linear activation for the output layer, use:
# model, results = ANN(X, y, task='regression', scaler='standard', epochs=20, optimizer='Adam', activation='relu/linear', early_stop=True, dropout=0.2, save=True)
