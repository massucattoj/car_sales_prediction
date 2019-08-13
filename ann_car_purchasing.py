"""
    Regression for Car Purchasing.

    Develop a model to predict the total dollar amount that customers
    are willing to pay given the following attributes
"""

# Libraries Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import dataset
df = pd.read_csv('car_purchasing_data.csv', sep=',', encoding='ISO-8859-1')
df.info()  # info about variables
df.head()  # get the first 5 rows of dataframe


## Visualize Dataset (correlation, distribuition,...)
#corr_mat = df.corr()  # correlation matrix
#fig = sns.pairplot(df)
#fig.savefig("output.png")

# Create testing and training dataset
# Split dependent variable from independets variables
X = df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis= 1)
y = df['Car Purchase Amount']

# get sizes from X and y
X.shape  # (500, 5)
y.shape  # 500 rows

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
std = MinMaxScaler()
X_scaled = std.fit_transform(X)
std.data_max_
std.data_min_

y = y.values.reshape(-1,1)
y_scaled = std.fit_transform(y)
y_scaled
y_scaled.shape

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                                                    y_scaled,
                                                    test_size = 0.3)



# Training the model
from keras.models import Sequential
from keras.layers import Dense

# create model
model = Sequential()
model.add(Dense(5, input_dim = 5, activation= 'relu'))
model.add(Dense(10, activation= 'relu'))
model.add(Dense(5, activation= 'relu'))
# output (predict certain values)
model.add(Dense(1, activation= 'linear'))

# infos about model parameters
model.summary()

"""
    x: Numpy array of training data (if the model has a single input), or
       list of Numpy arrays (if the model has multiple inputs).
    y: Numpy array of target (label) data (if the model has a single output),
       or list of Numpy arrays (if the model has multiple outputs)
    epochs: Integer. Number of epochs to train the model. An epoch is an
            iteration over the entire x and y data provided.
    batch_size: Integer or None. Number of samples per gradient update. If
                unspecified, batch_size will default to 32.
    validation_split: Float. Fraction of images reserved for validation
                      (strictly between 0 and 1).
    verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress
                      bar, 2 = one line per epoch.
"""
# compile model
model.compile(optimizer= 'adam', loss= 'mse')
epochs_hist = model.fit(X_train,
                        y_train,
                        epochs= 100,
                        batch_size= 25,
                        verbose= 1,
                        validation_split=0.2)


# Evaluation the model
epochs_hist.history.keys()
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress - During Training')
plt.ylabel('Training and Validation Loss')
plt.xlabel('Epoch Number')
plt.legend(['Training Loss', 'Validation Loss'])

# Gender, Age, Anual Salary, Credit Card Debt, Net Worth
X_testNEW = np.array([[1, 50, 50000, 10000, 600000]])
y_predict = model.predict(X_testNEW)
print('Expected Purchase Amount: ', y_predict)

