import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt 
import pandas_datareader as data # library used to read financial data from interner in df
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

st.title('Stock prediction')
user_input = st.text_input('Enter stock Ticker', 'AAPL')

start = dt.datetime(2000,1,1)
end = dt.datetime(2021,12,31)

df = data.DataReader(user_input,'yahoo',start=start,end=end)

#describing data
st.write(df.describe())
st.write(df.head())

#visualization
st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

fig = plt.figure(figsize=(12,6))
plt.plot(ma100, 'g', label="100MA")
plt.plot(ma200, 'r', label="200MA")
plt.plot(df.Close, label="Price")
plt.legend()
st.pyplot(fig)

# scaling values
scaler = MinMaxScaler(feature_range=(0,1))
data = pd.DataFrame(df.Close)
data_testing_array = scaler.fit_transform(data)

x_test = []
y_test = []
for i in range (100, len(data_testing_array)):
  x_test.append(data_testing_array[i-100:i])
  y_test.append(data_testing_array[i,0])

#Loading Model
model = load_model('lstm_model.h5')
x_test, y_test = np.asarray(x_test), np.asarray(y_test)
# x_test = np.asarray(x_test)

#Prediction
y_predicted = model.predict(x_test)
scale_factor = 1/scaler.scale_[0]
y_predicted = y_predicted*scale_factor
y_test = y_test * scale_factor

#Final plot
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_predicted, 'r', label="predicted price")
plt.plot(y_test, 'b', label="original price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)