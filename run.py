"""
https://www.hhllcks.de/blog/2018/5/4/version-your-machine-learning-models-with-sacred
import keras from tensorflow not directly from keras
"""
# imports {{{
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver

from src.data_processing.data_loader import make_data
from src.config import params
from src.models.model import set_model
from src.train import train
from keras.callbacks import Callback
# }}}

# start mongdbservice by mongod --config /opt/homebrew/etc/mongod.conf --fork
ex = Experiment("mnist_cnn")
db_name = "sacred_db_1"
url = '127.0.0.1:27017'
ex.observers.append(MongoObserver.create(
    url=url,
    db_name=db_name)
)
# optional
ex.captured_out_filter = apply_backspaces_and_linefeeds

# decorate for sacred {{{


@ex.config
def my_config():
    params_selected = params[0]


@ex.capture
def my_metrics(_run, logs):
    """
    each time an epoch ends sacred will log the metrics,
    and plot in omniboard.
    """
    _run.log_scalar("loss", float(logs.get('loss')))
    _run.log_scalar("accuracy", float(logs.get('accuracy')))
    _run.log_scalar("val_loss", float(logs.get('val_loss')))
    _run.result = float(logs.get('val_loss'))


@ex.automain
def my_main(params_selected):
    x_train, y_train, x_test, y_test, input_shape = make_data(params_selected)
    model = set_model(params_selected, input_shape)

    class LogMetrics(Callback):
        def on_epoch_end(self, _, logs={}):
            my_metrics(logs=logs)
    train(m=model, x_train=x_train, y_train=y_train,
          x_test=x_test, y_test=y_test,
          params=params_selected, callbacks=LogMetrics)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print("open omniboard by the following command...",
          f"omniboard -m {url}:{db_name}", end="\n", sep='\n')
# }}}
