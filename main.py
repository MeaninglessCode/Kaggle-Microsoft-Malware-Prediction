from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping

from data import get_data

x_train, x_test, y_train, y_test = get_data()

model = Sequential([
    Dense(output_dim=100, activation='relu', input_dim=len(x_train.columns)),  # input layer
    Dense(output_dim=50, activation='sigmoid'),
    Dense(output_dim=50, activation='relu'),
    Dense(output_dim=1, activation='softmax'),  # output layer
])

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-6, decay=1e-2), metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100, batch_size=16, callbacks=[EarlyStopping(monitor='loss', patience=0)])  # backprop arguments

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=16)

print(loss_and_metrics)
