import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def build_lstm_model(input_shape, cfg):
    """Constructs an LSTM model with AdamW based on the configuration."""
    model = Sequential([Input(shape=input_shape)])
    
    # Add multiple LSTM layers
    units_list = cfg['units']
    if not isinstance(units_list, list):
        units_list = [units_list]
        
    for i, u in enumerate(units_list):
        return_sequences = (i < len(units_list) - 1)
        model.add(LSTM(u, return_sequences=return_sequences, 
                       dropout=cfg['dropout'],
                       recurrent_dropout=cfg.get('recurrent_dropout', 0.0)))
    
    model.add(Dense(cfg['dense_units'], activation='relu'))
    model.add(Dropout(cfg['dropout']))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=cfg['lr'],
            weight_decay=cfg.get('weight_decay', 0.004)
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
