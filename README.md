# Learning Gaussian Mixture Representations for Tensor Time Series Forecasting
This is a implementation of GMRL.

# Installation Dependencies

Python 3 (>= 3.6; Anaconda Distribution)

PyTorch (>= 1.6.0) 

Numpy >= 1.17.4

Pandas >= 1.0.3

torch-summary (>= 1.4.5)

# Model Training
``` python
python main.py -mode train -version 0 cuda_name
```

# Model Evaluation
``` python
python main.py -mode eval cuda_name
```
