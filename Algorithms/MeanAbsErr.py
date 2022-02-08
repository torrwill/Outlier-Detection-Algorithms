# import pandas (read_csv)
# import sklearn (train_test_split, LinearRegression, mean_absolute_error)
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# load data set
url = 'https://raw.githubusercontent.com/torrwill/Outlier-Detection-Algorithms/main/Dataset/housing.csv'
df = read_csv(url, header = None)
data = df.values

# split data -- input and output
data_input, data_output = data[:, :-1], data[:, -1]

# split data -- train, test
input_train, input_test, output_train, output_test = train_test_split(data_input, data_output, test_size = 0.33, random_state = 1)

# fit model
model = LinearRegression()
model.fit(input_train, output_train)

# evaluate model
yhat = model.predict(input_test)
# evaluate prediction
mae = mean_absolute_error(output_test, yhat) # MAE (avg) = 3.417