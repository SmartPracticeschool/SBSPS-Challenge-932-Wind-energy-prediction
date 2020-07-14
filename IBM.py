#importing Libraies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn import preprocessing, svm
import statsmodels.tsa.api as smt
import statsmodels.api as sm

#Importing dataset
data = pd.read_csv('T1.csv',engine="python")

#prediction for 72 hours
features_considered = ['LV ActivePower (kW)','Wind Speed (m/s)','Wind Direction()']
features = df[features_considered]
features.index = df['DateTime']
features.head()

features.plot(subplots=True)

TRAIN_SPLIT = 49000

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset-data_mean)/data_std

def multivariate_data(dataset, target, start_index, end_index, history_size,target_size,step,single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

past_history =1008
future_target = 72
STEP = 6

x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,TRAIN_SPLIT, past_history,future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],TRAIN_SPLIT, None, past_history,future_target, STEP)


BATCH_SIZE = 256
BUFFER_SIZE = 10000

print ('Single window of past history : {}'.format(x_train_multi[0].shape))
print ('\n Target power to predict : {}'.format(y_train_multi[0].shape))

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

def create_time_steps(length):
  return list(range(-length, 0))


def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 1]), label='History')
  plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro', label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()
  
 
     
multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(72))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')   

for x, y in val_data_multi.take(1):
    print (multi_step_model.predict(x).shape)
    
EVALUATION_INTERVAL = 200
EPOCHS = 10
    
    
def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()
  plt.show()
     
    
multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,steps_per_epoch=EVALUATION_INTERVAL,validation_data=val_data_multi,validation_steps=50)

plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

for x, y in val_data_multi.take(3):
     multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
     
   

from sklearn.linear_model import LinearRegression
mr=LinearRegression()

mr.predict(x_train_multi,y_train_multi)

pickle.dump(mr,open('app.pkl','wb'))
y_pred=mr.predict(y_train_multi)

#Feature scaling
#separating dependent and independent variables
x=data.iloc[0:,1:2].values
y=data.iloc[0:,2].values

from sklearn.preprocessing import MinMaxScaler
m=MinMaxScaler()
x=m.fit_transform(x)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)


#Model Building
#Splitting Training Data and Test Data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train
y_train
x_test
y_test
plt.scatter(x,y)
#plt.scatter(x[:,2],y)
#plt.scatter(x[:,0],y)

from sklearn.linear_model import LinearRegression
mr=LinearRegression()
mr.fit(x_train,y_train)
pickle.dump(mr,open('wind.pkl','wb'))
y_pred=mr.predict(x_test)
y_pred
y_test

#Model Evaluation
from sklearn.metrics import r2_score
r2_score(y_pred,y_test)
data.head()
data.describe()
data.info()


#Data Visualization
data['LV ActivePower (kW)'].hist(bins=10)
plt.savefig("../static/datavizLVactive.png")
data['Theoretical_Power_Curve (KWh)'].hist(bins=10)
plt.savefig("../static/datavizTHEO.png")
data['Wind Speed (m/s)'].hist(bins=10)
plt.savefig("../static/datavizwindspeed.png")
data['Wind Direction (°)'].hist(bins=10)
plt.savefig("../static/datavizwinddirec.png")

data['LV ActivePower (kW)'].hist(bins=10)
data['Theoretical_Power_Curve (KWh)'].hist(bins=10)
data['Wind Speed (m/s)'].hist(bins=10)
data['Wind Direction (°)'].hist(bins=10)
plt.savefig("../static/dataviz of all.png")


#boxplot
x=data.iloc[0:,2:3].values
y=data.iloc[0:,1:2].values

data.boxplot(column="Theoretical_Power_Curve (KWh)")
plt.savefig("../static/boxplotTHEOR.png")
data.boxplot(column="LV ActivePower (kW)")
plt.savefig("../static/boxplotLVPower.png")
data.boxplot(column="Wind Speed (m/s)")
plt.savefig("../static/boxplotWindSpeed.png")
data.boxplot(column="Wind Direction (°)")
plt.savefig("../static/boxplotWindDirection.png")


#time series graph for database
x=data.iloc[0:,0:1].values
y=data.iloc[0:,1:3].values

data.index = data['Date/Time'] # indexing the Datetime to get the time period on the x-axis.
ts_theoretical = data['Theoretical_Power_Curve (KWh)']
ts_real = data['LV ActivePower (kW)']
plt.figure(figsize=(25,10))
plt.plot(ts_theoretical, label='Power Theoretical (KWH)',color='green')
plt.plot(ts_real,label='Power Real (KWH)',color='red')
plt.title('Time Series')
plt.xlabel("Time(year-month-date)")
plt.ylabel("Power (KWH)")
plt.legend(loc='best')
plt.savefig("../static/timeseries1.png")

#time series graph for database 
x=data.iloc[0:,0:1].values
y=data.iloc[0:,2:].values

data.index = data['Date/Time'] # indexing the Datetime to get the time period on the x-axis.
ts_windspeed = data['Wind Speed (m/s)']
ts_winddirection = data['Wind Direction (°)']
ts_real = data['LV ActivePower (kW)']
plt.figure(figsize=(25,10))
plt.plot(ts_windspeed, label='windspeed (m/s)',color='green')
plt.plot(ts_winddirection,label='wind direction (°)',color='red')
plt.plot(ts_real,label='LV ActivePower (kW)',color='blue')
plt.title('Time Series')
plt.xlabel("Time(year-month-date)")
plt.ylabel("Power (KWH)")
plt.legend(loc='best')
plt.savefig("../static/timeseries2.png")

#pairplot
data['Date/Time'] = pd.to_datetime(data['Date/Time'],format='%d %m %Y %H:%M')
data['Hour'] = data['Date/Time'].dt.hour
data['Minute'] = data['Date/Time'].dt.minute
data['Day'] = data['Date/Time'].dt.day
data['Month'] = data['Date/Time'].dt.month
data['Year'] = data['Date/Time'].dt.year
data.head()

# Calculating difference between theoretical power Curve and LV Active Power
data['Energy_Difference(KW)'] = data['Theoretical_Power_Curve (KWh)']-data['LV ActivePower (kW)']
data['Energy_Difference(KW)'].head(5)
data = data.reindex(columns=['Minute','Hour','Day', 'Month','Year','Date/Time', 'LV ActivePower (kW)', 
       'Theoretical_Power_Curve (KWh)','Energy_Difference(KW)', 'Wind Direction (°)','Wind Speed (m/s)'])
data.head()
sns.pairplot(data)
plt.savefig("../static/pairplot.png")



#heatmap
sns.heatmap(data.corr())
plt.savefig("../static/heatmap.png")

#scatter plot
plt.scatter(data['LV ActivePower (kW)'],data['Theoretical_Power_Curve (KWh)'])
plt.title("Numerical Feature: Theoretical_Power_Curve (KWh) v/s LV ActivePower (kW)")
plt.xlabel("LV ActivePower (kW)")
plt.ylabel("Theoretical_Power_Curve (KWh)")
plt.savefig("../static/Scatterplot LV vs TH.png")

plt.scatter(data['Date/Time'],data['Theoretical_Power_Curve (KWh)'],color='g')
plt.title("Numerical Feature:  Theoretical_Power_Curve (KWh)  v/s Date/Time")
plt.xlabel("Date/Time")
plt.ylabel("Theoretical_Power_Curve (KWh)")
plt.savefig("../static/Scatterplot dt vs thpower.png")

plt.scatter(data['Date/Time'],data['LV ActivePower (kW)'],color='y')
plt.title("Numerical Feature: Date/Time vs LV ActivePower (kW)  ")
plt.xlabel("Date/Time ")
plt.ylabel("LV ActivePower (kW) ")
plt.savefig("../static/Scatterplot dt vs LV.png")

plt.scatter(data['Wind Speed (m/s)'],data['LV ActivePower (kW)'],color='r')
plt.title("Numerical Feature:Wind Speed (m/s) vs LV ActivePower (kW)  ")
plt.xlabel("Wind Speed (m/s) ")
plt.ylabel("LV ActivePower (kW)")
plt.savefig("../static/Scatterplot windspeed vs LV.png")

#joint plot
sns.jointplot(data['LV ActivePower (kW)'],data['Theoretical_Power_Curve (KWh)'])
plt.savefig("../static/joint LV vs THEO.png")

sns.jointplot(data['Theoretical_Power_Curve (KWh)'],data['Wind Direction (°)'])
plt.savefig("../static/joint THEO vs WindDirection.png")

#line plots

plt.plot(data['Date/Time'],data['Theoretical_Power_Curve (KWh)'])
plt.xlabel("Date/Time")
plt.ylabel("Theoretical_Power_Curve (KWh)")
plt.savefig("../static/lineplot dt vs theo.png")


plt.plot(data['Date/Time'],data['LV ActivePower (kW)'])
plt.xlabel("Date/Time")
plt.ylabel("LV ActivePower (kW)")
plt.savefig("../static/lineplot dt vs LV.png")


plt.plot(data['Wind Speed (m/s)'],data['Theoretical_Power_Curve (KWh)'])
plt.xlabel("wind speed")
plt.ylabel("Theoretical_Power_Curve (KWh)")
plt.savefig("../static/lineplot wind vs theo.png")


plt.plot(data['LV ActivePower (kW)'],data['Theoretical_Power_Curve (KWh)'])
plt.xlabel("LV ActivePower")
plt.ylabel("Theoretical power")
plt.savefig("../static/lineplot LV vs theo.png")










v