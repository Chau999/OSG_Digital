








# Load Pacakges
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def generate_GP_simulation(n, noise):
    x = np.sort(np.random.normal(loc=0, scale=2, size=n))
    y = np.sin(x**2)*x + noise * np.random.normal(size=n)
    
    return x, y


# Simulate some data
n, noise = 30, 0.2
x, y = generate_GP_simulation(n, noise)

plt.plot(x, y, "r.")
plt.xlabel("x"), plt.ylabel("y"), plt.title("Simulated Data")
plt.show()


# Set up the kernel function
kernel_ = RBF(length_scale=2.) + WhiteKernel(noise_level=noise)

# Visualise the kernel matrix
K = kernel_(x.reshape(-1, 1))
sns.heatmap(K)
plt.title("Covariance matrix specifying similarity across data")
plt.show()


# Set up the GPR object and fit
gpr = GaussianProcessRegressor(kernel=kernel_)
gpr.fit(x.reshape(-1, 1), y)



# Set up x_test
x_test = np.linspace(np.min(x)-2, np.max(x)+2, 300)

# Run your prediction and retrieve the mean and uncertainity
y_pred, y_std = gpr.predict(x_test.reshape(-1, 1), return_std=True)


def plot_result(x, y, x_test, y_pred, y_std):
    plt.figure(figsize=(7,4))

    plt.plot(x, y,"rx", label="Data")
    plt.plot(x_test, y_pred, "b-",linewidth=2., label="Predictions")
    
    upper = y_pred + 1.96 * y_std
    lower = y_pred - 1.96 * y_std
    plt.fill_between(x_test, upper, lower, alpha=0.3)
    plt.title("GPR prediction and uncertainity.")
    plt.legend()
    plt.show()


# Visualisation
plot_result(x, y, x_test, y_pred, y_std)


# Import packages
import numpy as np
import matplotlib.pylab as plt

# This is the key
import shap

from sklearn.ensemble import RandomForestRegressor


shap.initjs()

X, y = shap.datasets.boston()

rf = RandomForestRegressor(n_estimators=500)
rf.fit(X, y)

explainer = shap.KernelExplainer(rf.predict, shap.sample(X, 10))
shap_values = explainer.shap_values(X.iloc[:30,:], nsamples=50)


shap.summary_plot(shap_values, X.iloc[:30, :])


shap.summary_plot(shap_values, X.iloc[:30, :], plot_type="bar")


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical

import matplotlib.pylab as plt

from sklearn.datasets import load_boston
from alibi.explainers import CounterFactualProto

print('TF version: ', tf.__version__)
print('Eager execution enabled: ', tf.executing_eagerly())


def boston_preprocessing():
    boston = load_boston()
    data, target, feature_names = boston.data, boston.target, boston.feature_names

    # Modify the feature
    y = np.zeros([target.shape[0], ])
    y[np.where(target > np.median(target))[0]] = 1

    # For simplicity, we will remove categorical feature
    data = np.delete(data, 3, 1)
    feature_names = np.delete(feature_names, 3)
    
    return data, target, feature_names


data, target, feature_names = boston_preprocessing()

# Standarise data
mu, sigma = data.mean(axis=0), data.std(axis=0)
data = (data - mu)/sigma

# Train-Test split
idx = 475
x_train,y_train = data[:idx,:], y[:idx]
x_test, y_test = data[idx:,:], y[idx:]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def nn_model():
    x_in = Input(shape=(12,))
    x = Dense(40, activation='relu')(x_in)
    x = Dense(40, activation='relu')(x)
    x_out = Dense(2, activation='softmax')(x)
    nn = Model(inputs=x_in, outputs=x_out)
    nn.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return nn


nn = nn_model()
nn.summary()
nn.fit(x_train, y_train, batch_size=64, epochs=500, verbose=0)


# Evaluation
score = nn.evaluate(x_test, y_test, verbose=0)
print('Test accuracy: ', score[1])


# Generate counterfactual 
X = x_test[1].reshape((1,) + x_test[1].shape)
shape = X.shape


tf.compat.v1.disable_eager_execution()
# initialize explainer, fit and generate counterfactual
cf = CounterFactualProto(nn, shape, use_kdtree=True, theta=10., max_iterations=1000,
                         feature_range=(x_train.min(axis=0), x_train.max(axis=0)),
                         c_init=1., c_steps=10)

cf.fit(x_train)
explanation = cf.explain(X)


print(f'Original prediction: {explanation.orig_class}')
print('Counterfactual prediction: {}'.format(explanation.cf['class']))


# Examine the explanation
explanation['cf']


orig = X * sigma + mu
counterfactual = explanation.cf['X'] * sigma + mu
delta = counterfactual - orig
for i, f in enumerate(feature_names):
    if np.abs(delta[0][i]) > 1e-4:
        print('{}: {}'.format(f, delta[0][i]))



