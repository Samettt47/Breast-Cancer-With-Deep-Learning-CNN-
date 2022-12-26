import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten , Dense  , Dropout , BatchNormalization
from tensorflow.keras.layers import Conv1D , MaxPool1D

from tensorflow.keras.optimizers import Adam

print(tf.__version__)

import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import datasets , metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

kanser = datasets.load_breast_cancer()
print(kanser.DESCR)

X = pd.DataFrame(data = kanser.data , columns = kanser.feature_names)
X.head()

#buradan kanseri numerikleştirdik ve böylece kanserin iyi huylu veya kötü huylu olduğunu anlayacağız 

y = kanser.target
print(y)

# iyi huylumu kötü huylumu oldugunu
# eğer 0 ise kötü 1 ise iyi huylu diyerek anlayabiliriz.

kanser.target_names # bu kod satırı bize kanserin malignant(kötü huylu) , bening(iyi huylu) oldugunu gosterir

#                       MODELİN EĞİTİMİ  

X.shape # 569 satırdan ve 30 sütundan oluşuyor.

# test boyutunu (yüzde 20 lik bi test olacak )
# stratify = katman demektir y verdik ki test ve eğitim verilerini simetrik olarak dağıtabilsin
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 0 , stratify= y)

print(X_train.shape )# test verisini aldıktan sonra artık deneme de ki boyutu aldık
print(X_test.shape) # test verisinin boyutu 

# CNN üç boyutlu bir veriyi kabul ettiği için veriyi yeniden şekillendirmek gerekiyor
# Standard Scaler ile bir diziye attık diyebiliriz
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

X_train = X_train.reshape(455,30,1)
X_test = X_test.reshape(114,30,1)

#Modelimizi eğitmek isteyeceğimiz epoch sayısını tanımlayacağız
# bu katmanda her girdi için 30 sutun 1 satır olacak
epochs = 50
model = Sequential()
model.add(Conv1D(filters = 32 , kernel_size = 2 , activation = 'relu' , input_shape = (30,1)))
model.add(BatchNormalization()) # burada bunu kullandık   eş zamanlı olarak öğrenime olanak sağlar
model.add(Dropout(0.2)) # eğitim sırasında aşırı öğrenmeyi engellemek için yazdık(bazı noronları unutmak için kullanılır bu noron yüzdesi de 0.2 dir)


model.add(Conv1D(filters = 64 , kernel_size = 2 , activation = 'relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten()) # çok boyutlu veriyi tek boyutlu yapar
model.add(Dense(64 , activation = 'relu'))
model.add(Dropout(0.5))

model.add(Dense(1,activation = 'sigmoid')) # her bir noronun bir önceki
                                            # katmandan girdi alır ( Dense )
                    
print(model.summary()) # modelin özetinin getirdik

#öğrenme hızını 0.00005 ayarrlıyoruz hız yavaş loss functionı ise binary (entropi) kullanıyoruz 
model.compile(optimizer = Adam(learning_rate = 0.00005) , loss = 'binary_crossentropy' , metrics = ['accuracy'])

#modelin derlenen modele uyarlanması 
history = model.fit(X_train , y_train , epochs = epochs , validation_data=(X_test , y_test) , verbose = 1)


def plot_learningCurve(history , epoch):
  epoch_range =range(1 , epoch + 1)
  plt.plot(epoch_range , history.history['accuracy'])
  plt.plot(epoch_range , history.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'val'] , loc = 'upper left')
  plt.show()

  plt.plot(epoch_range , history.history['loss'])
  plt.plot(epoch_range , history.history['val_loss'])
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'val'] , loc = 'upper left')
  plt.show()
  
plot_learningCurve(history, epochs)


























