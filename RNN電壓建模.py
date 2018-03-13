#%%
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense, Dropout, LSTM, GRU
from keras.optimizers import Adam,RMSprop
#import seaborn as sns

random_num = 777
#%%
data0808 = pd.read_csv("/Volumes/Transcend/NCTUWORK/低電壓比賽/data/format_0808.csv", index_col="Date")
data0806 = pd.read_csv("/Volumes/Transcend/NCTUWORK/低電壓比賽/data/format_0806.csv", index_col="Date")
data_total = pd.concat([data0806,data0808])
#%% 要用哪個用戶？
Cus = "M00033"

data_total['per_value3'] = data_total.Value.shift(3)  #往後平移三格
d1 = data_total.dropna(axis=0,how='any')
print ("檢查是否仍有缺失值：",pd.isnull(d1).values.sum())
print ("各用戶的數據量：",d1.CustomerID.value_counts().head())

d2 = d1[d1["CustomerID"]==Cus]
#%%
hol = 3 # 1:只使用假日，2:只使用平日，3:都用
if hol==1:
    d2 = d2[d2["holiday"]==1]
    d2 = d2.drop(['holiday'],axis=1)
    print ("只使用假日")
elif hol==2:
    d2 = d2[d2["holiday"]==0]
    d2 = d2.drop(['holiday'],axis=1) 
    print ("只使用平日")
elif hol==3:
    d2=d2
    print ("假日與平日都用")
            
d2 = d2.drop(["CustomerID",'contract',"position",'user'],axis=1) #去除欄位
#print (pd.value_counts(d2["Week"]))   #看各星期之數量

# 處理dummues欄位：Week
week = pd.get_dummies(d2["Week"])
d2 = d2.drop("Week",axis=1)
d2 = pd.concat([d2,week],axis=1)

#%% 製作天氣的LAG
shift_days = 1 #延遲天數

d2['Temp'] = d2.Temp.shift(shift_days)
d2['Humidity'] = d2.Humidity.shift(shift_days)
d2['rain'] = d2.rain.shift(shift_days)
print (d2.head())
data_ID = d2.dropna(axis=0,how='any')

#%% 標準化
scaling = 1   # 1：True，2：False
need_scale = ["Value","per_value1","per_value168","per_value2","per_value24","Temp","Humidity","rain","per_value3"]

if scaling ==1:
    data_ID[need_scale] = scale(data_ID[need_scale])
elif scaling ==2:
    data_ID = d2.dropna(axis=0,how='any')
#%% 
X = data_ID.iloc[:,1:].values
Y = data_ID.iloc[:,:1].values

#
random_num = 777
split_ratio = 0.2
TIME_STEPS = 24
INPUT_SIZE = X.shape[1]
#BATCH_SIZE = int(len(X)*(1-split_ratio))
BATCH_SIZE = 256
BATCH_INDEX = 0
OUTPUT_SIZE = Y.shape[1]
CELL_SIZE = 128
LR = 0.001

def timewindow(file,time):
    X_train = np.array([[[0 for k in range(INPUT_SIZE)] for j in range(time)] for i in range(len(X)-time)],dtype=float)
    window = file
    
    for j in range(len(file)-time):
        X_train[j] = window[j:j+time]
    return(X_train)

X = timewindow(X, TIME_STEPS)
Y = Y[:(len(Y)-TIME_STEPS)]
print ("X.shape = ", X.shape)
print ("Y.shape = ", Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = split_ratio, random_state = random_num) 

#%%
# Model
model = Sequential()
model.add(Dense(INPUT_SIZE,
        input_shape=(TIME_STEPS,INPUT_SIZE),        
        activation='linear',
        ))
model.add(Dropout(0.3))

activation_func = 'relu'

# Simple RNN
model.add(SimpleRNN(
        batch_input_shape=(None,TIME_STEPS,INPUT_SIZE),
        output_dim=CELL_SIZE,  #output的size其實就是cell size
        activation = 'relu',  #預設是linear
        ))


## LSTM
#model.add(LSTM(
#        batch_input_shape=(None,TIME_STEPS,INPUT_SIZE),
#        output_dim=CELL_SIZE,  #output的size其實就是cell size
#        activation = activation_func,
##        dropout_W=0.1, # Fraction of the input units to drop for input gates.
##        dropout_U=0.1, #Fraction of the input units to drop for recurrent connections.
#        ))



## GRU
#model.add(GRU(
#        batch_input_shape=(None,TIME_STEPS,INPUT_SIZE),
#        output_dim=CELL_SIZE,  #output的size其實就是cell size
#        activation = 'sigmoid'
#        ))

model.add(Dropout(0.3))

#Output Layer
model.add(Dense(OUTPUT_SIZE))  #不需寫input的size為何，因為會自動抓到上一層的output size當此層的input

#Optimizer
adam=Adam(LR)
rms = RMSprop(LR)

opt = adam
model.compile(
        optimizer = opt,
        loss = 'mse',
        metrics = ['mape']
        )
#metrics的官方文檔
# https://github.com/fchollet/keras/blob/master/keras/metrics.py
#%%
## NN
#model = Sequential()
#model.add(Dense(
#        batch_input_shape=(None,TIME_STEPS,INPUT_SIZE),        
#        activation='relu',
#        ))
#model.add(Dropout(0.2))
#model.add(Dense(64, batch_input_shape=(None,TIME_STEPS,INPUT_SIZE),activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(32, activation='relu'))
#model.add(Dropout(0.2))
#model.add(Dense(OUTPUT_SIZE))
#
##Optimizer
#adam=Adam(LR)
#rms = RMSprop(LR)
#
#opt = adam
#model.compile(
#        optimizer = opt,
#        loss = 'mape',
#        metrics = ['mape']
#        )



#%%
from keras.utils import plot_model
import pydot_ng as pydot
pydot.find_graphviz()
plot_model(model,to_file='/Volumes/Transcend/NCTUWORK/低電壓比賽/figure/model/LSTM.png')

#%%
#c = []
#a = []
#print ("batch size:" ,BATCH_SIZE)
#for step in range(8001):
#    X_batch = X_train[BATCH_INDEX:BATCH_SIZE+BATCH_INDEX, :,:] #三維
#    Y_batch = Y_train[BATCH_INDEX:BATCH_SIZE+BATCH_INDEX, :]  #二維
#    cost = model.train_on_batch(X_batch, Y_batch)
#    
#    BATCH_INDEX +=BATCH_SIZE  #每跑一次，index就會加一個batch size
#    BATCH_INDEX = 0 if BATCH_INDEX>=X_train.shape[0] else BATCH_INDEX  #不讓batch index超過總體大小
#    
#    if step%50 ==0:
#        cost, accuracy = model.evaluate(X_test, Y_test, batch_size=Y_test.shape[0], verbose=False)
#        print ('step:',step,'test cost:',cost, 'test MAPE:',accuracy)
#        c.append(cost)
#        a.append(accuracy)
#    
#%%
his = []
his_val = []
#%%
H = model.fit(X_train, Y_train, epochs=2500, batch_size=BATCH_SIZE, verbose=2,validation_split=0.2) #validation_split:在每個epoch訓練時切出測試集，不參與訓練，並在每個epoch結束後做檢查
his.append(H.history["loss"]) #紀錄loss
his_val.append(H.history["val_loss"])


train_pre = model.predict(X_train)
test_pre = model.predict(X_test)

MAPE_train = np.mean(np.abs(((train_pre-Y_train)/Y_train) *100))
MAPE_test = np.mean(np.abs(((test_pre-Y_test)/Y_test) *100))

print ("MAPE of train:", MAPE_train)
print ("MAPE of test:", MAPE_test)

#%%
loss_his = [val for b in his for val in b]
loss_val = [val for b in his_val for val in b]

plt.plot(range(len(loss_his)), loss_his, label="train_mse")
plt.plot(range(len(loss_val)), loss_val, label="validation_mse")
plt.title("Curve of Loss Function MSE")
plt.legend(loc='best',fontsize=12)
#plt.xlim(0,50)
plt.ylabel("MSE")
plt.xlabel("Epochs")
#plt.xlim(0,450)
#plt.savefig('/Volumes/Transcend/NCTUWORK/低電壓比賽/figure/curve/loss_function_mse.png')
plt.show()

#%%
import seaborn as sns

n = 50
plt.plot(range(n),Y_train[:n], label='Y_train')
plt.plot(range(n),train_pre[:n], label='predict of train')
plt.title("%d train vs predict"%n)
plt.legend(loc='best')
plt.savefig('/Volumes/Transcend/NCTUWORK/低電壓比賽/figure/'+activation_func+'/'+Cus+'/%d predict_of_train.png'%n)
plt.show()

n2 = Y_train.shape[0]
plt.plot(range(n2),Y_train[:n2], label='Y_train')
plt.plot(range(n2),train_pre[:n2], label='predict of train')
plt.title("All train vs predict")
plt.legend(loc='best')
plt.savefig('/Volumes/Transcend/NCTUWORK/低電壓比賽/figure/'+activation_func+'/'+Cus+'/All predict_of_train.png')
plt.show()

n3 = 50
plt.plot(range(n3),Y_test[:n3], label='Y_test')
plt.plot(range(n3),test_pre[:n3], label='predict of test')
plt.title("LSTM\n%d test vs predict"%n3)
plt.legend(loc='best')
plt.savefig('/Volumes/Transcend/NCTUWORK/低電壓比賽/figure/'+activation_func+'/'+Cus+'/%d predict_of_test.png'%n3)
plt.show()

n4 = Y_test.shape[0]
plt.plot(range(n4),Y_test[:n4], label='Y_test')
plt.plot(range(n4),test_pre[:n4], label='predict of test')
plt.title("LSTM\nAll test vs predict")
plt.legend(loc='best')
plt.savefig('/Volumes/Transcend/NCTUWORK/低電壓比賽/figure/'+activation_func+'/'+Cus+'/All predict_of_test.png')
plt.show()
#%%
plt.plot(range(len(c)),c,c='r')
plt.title("Cost")
plt.show()
plt.plot(range(len(a)),a,c='g')
plt.ylim(0,30)
plt.title("MAPE")
plt.show()
#%%
model.save('/Volumes/Transcend/NCTUWORK/低電壓比賽/python/model/rnn_'+activation_func+'_'+'%d'%hol+'.h5')
#%%
model = load_model('/Volumes/Transcend/NCTUWORK/低電壓比賽/python/model/rnn_'+activation_func+'_'+'%d'%hol+'.h5')




