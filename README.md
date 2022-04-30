# a machine learning project management

Thanks for the original contribution from the artical [Version your machine learning models with Sacred](https://www.hhllcks.de/blog/2018/5/4/version-your-machine-learning-models-with-sacred) and [Managing Machine Learning projects](https://towardsdatascience.com/managing-machine-learning-projects-226a37fc4bfa).

This example reorganize the project [monchewharry/ML-manage-tfexample](https://github.com/monchewharry/ML-manage-tfexample) into general project structure as suggested by the mentioned credit above. The project running is similar, just by the following and read the printted hints to start `omniboard`.

```bash
python run.py
```
## todo

- add hyperparameter tune (no duplicate experiment control)
- add unittest
- add docker

## structure

```bash
.
├── README.md
├── run.py
├── src
│  ├── config.py
│  ├── data_processing
│  │  └── data_loader.py
│  ├── inference.py
│  ├── models
│  │  └── model.py
│  ├── train.py
│  ├── utils
│  └── visualization
│     └── explore.py
└── tests
   ├── data
   │  └── test-set.py
   ├── integration
   │  └── test_model.py
   └── unit
      └── test_data_loader.py

```
## More

- [mongodb macOS](https://www.mongodb.com/docs/manual/tutorial/install-mongodb-on-os-x/)
- [sacred example](https://sacred.readthedocs.io/en/stable/examples.html)
- [omniboard quick-start](https://vivekratnavel.github.io/omniboard/#/quick-start)
