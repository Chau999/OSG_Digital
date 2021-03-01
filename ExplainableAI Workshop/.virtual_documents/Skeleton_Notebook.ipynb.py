import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical

import matplotlib.pylab as plt

from sklearn.datasets import load_breast_cancer
from alibi.explainers import CounterFactualProto

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly())


# Load Boston dataset 
cancer = load_breast_cancer()
data = cancer.data
target = cancer.target
feature_names = cancer.feature_names


feature_names


from sklearn.datasets import load_diabetes


load_diabetes()



