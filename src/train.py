
def train(m, x_train, y_train, x_test, y_test, params, callbacks):
    m.fit(x_train, y_train,
          batch_size=params['batch_size'],
          epochs=params['epochs'],
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[callbacks()])
