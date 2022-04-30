# a machine learning version control example

Thanks for the original contribution from the artical [Version your machine learning models with Sacred](https://www.hhllcks.de/blog/2018/5/4/version-your-machine-learning-models-with-sacred).

This example combines the `tensorflow.keras` job with [sacred](https://github.com/IDSIA/sacred) as version control and [omniboard](https://github.com/vivekratnavel/omniboard) as dashboard on Platform macOS-M1.

## structure

- mnist_cnn.py: no sacred version control. Updated to keras module from tensorflow.
- mnist_cnn_sacred.py: work with sacred. Updated to latest tensorflow and sacred setting.
- mongodb-README.md : the setup of mongodb service for sacred.
- omniboard-README.md : setup omniboard connection.


## More

- [mongodb macOS](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-os-x/)
- [sacred example](https://sacred.readthedocs.io/en/stable/examples.html)
- [omniboard quick-start](https://vivekratnavel.github.io/omniboard/#/quick-start)
