# Usually CNN consists of convolution-subsampling pairs -> dense layer

# 1. How many pairs of conv-subs? -> try 1 ~ 3 (usually, complicated problems require more)
# 2. How many filters in conv? : usually increases (ex. 24 -> 48 -> 64 ...) -> try 8, 16 / 16, 32 / ...
# 3. How deep / large dense layer? : 1 layer 0, 32, 64, ... 2048 nodes / 2layer ...
# 4. How much dropouts?

# 5. Advanced : Include BatchNormalization?, replace '32C5' with '32C3-32C3', replace 'P2' with '32C5S2'
#               add batch normalization, add data augmentation

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# 1. Load
mnist = keras.datasets.mnist
(X_train_full, y_train_full), (test_X, test_Y) = mnist.load_data()

print(type(X_train_full))
print(X_train_full[0].shape)


# 2. CNN은 채널 필요
X_train_full, test_X = X_train_full.reshape(-1, 28, 28, 1), test_X.reshape(-1, 28, 28, 1)
print(X_train_full.shape, test_X.shape)

# 3. 보통 CNN은 zero-centering만 해준다. 
    # per-channel pixel mean(VGGNet)을 빼거나 그냥 전체 mean(AlexNet)을 빼주면 되는데, MNIST는 채널이 1개라서 상관 없다.
    # normalize 안함 : 이미지 데이터에서 feature의 범위가 달라서 모델에 악영향을 미치지 않기 떄문.
    # PCA 안함 : 상관성 분석 필요 없음
    # Whitening 안함 : whitening : feature의 상관성을 없애주고 분산을 1로 만드는 것. 필요없음

X_train_full, test_X = X_train_full/1.0, test_X/1.0 # float로 변경해줌
print(X_train_full.dtype)
exit()
avg = np.mean(X_train_full, axis = 0)
X_train_full -= avg
    # training data의 mean value를 다 빼주면서 zero-centering
test_X -= avg

# 4. Data Split
random_seed = 0
train_X, val_X, train_Y, val_Y = \
    train_test_split(X_train_full, y_train_full, random_state = random_seed, test_size = 0.1, stratify = y_train_full)

# 5. OneHot
train_Y, val_Y, test_Y = \
    to_categorical(train_Y, num_classes=10), to_categorical(val_Y, num_classes=10), to_categorical(test_Y, num_classes=10)


############## MODELS ##################

# 1. LeNET은 필터 갯수 6, 16을 사용하였는데, 요새 대부분의 코드는 32, 64 ,128 이런식으로 간다. 바꿔보자.

model =Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1))) 
model.add(MaxPooling2D())

model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(120, activation='relu'))

model.add(Dense(84, activation='relu'))

model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True) 

history = model.fit(train_X, train_Y, epochs = 100, validation_data=(val_X, val_Y),callbacks=[early_stopping_cb])

model.evaluate(test_X, test_Y)

# 0.9828 -> 0.9859 오답률 22% 감소

'''
Epoch 1/100
1688/1688 [==============================] - 17s 10ms/step - loss: 0.2414 - accuracy: 0.9424 - val_loss: 0.0793 - val_accuracy: 0.9782
Epoch 2/100
1688/1688 [==============================] - 16s 9ms/step - loss: 0.0858 - accuracy: 0.9750 - val_loss: 0.0840 - val_accuracy: 0.9778
Epoch 3/100
1688/1688 [==============================] - 16s 9ms/step - loss: 0.0700 - accuracy: 0.9791 - val_loss: 0.0571 - val_accuracy: 0.9840
Epoch 4/100
1688/1688 [==============================] - 16s 10ms/step - loss: 0.0615 - accuracy: 0.9822 - val_loss: 0.0515 - val_accuracy: 0.9848
Epoch 5/100
1688/1688 [==============================] - 16s 10ms/step - loss: 0.0531 - accuracy: 0.9838 - val_loss: 0.0488 - val_accuracy: 0.9877
Epoch 6/100
1688/1688 [==============================] - 16s 10ms/step - loss: 0.0443 - accuracy: 0.9870 - val_loss: 0.0488 - val_accuracy: 0.9870
Epoch 7/100
1688/1688 [==============================] - 17s 10ms/step - loss: 0.0421 - accuracy: 0.9886 - val_loss: 0.0632 - val_accuracy: 0.9855
Epoch 8/100
1688/1688 [==============================] - 17s 10ms/step - loss: 0.0360 - accuracy: 0.9893 - val_loss: 0.1089 - val_accuracy: 0.9778
Epoch 9/100
1688/1688 [==============================] - 16s 10ms/step - loss: 0.0357 - accuracy: 0.9901 - val_loss: 0.0488 - val_accuracy: 0.9873
Epoch 10/100
1688/1688 [==============================] - 17s 10ms/step - loss: 0.0343 - accuracy: 0.9906 - val_loss: 0.0531 - val_accuracy: 0.9880
Epoch 11/100
1688/1688 [==============================] - 16s 10ms/step - loss: 0.0343 - accuracy: 0.9912 - val_loss: 0.0542 - val_accuracy: 0.9863
Epoch 12/100
1688/1688 [==============================] - 16s 10ms/step - loss: 0.0282 - accuracy: 0.9922 - val_loss: 0.0468 - val_accuracy: 0.9897
Epoch 13/100
1688/1688 [==============================] - 16s 10ms/step - loss: 0.0308 - accuracy: 0.9912 - val_loss: 0.0897 - val_accuracy: 0.9847
Epoch 14/100
1688/1688 [==============================] - 16s 9ms/step - loss: 0.0276 - accuracy: 0.9925 - val_loss: 0.0675 - val_accuracy: 0.9880
Epoch 15/100
1688/1688 [==============================] - 16s 10ms/step - loss: 0.0276 - accuracy: 0.9938 - val_loss: 0.1024 - val_accuracy: 0.9858
Epoch 16/100
1688/1688 [==============================] - 16s 9ms/step - loss: 0.0295 - accuracy: 0.9928 - val_loss: 0.0725 - val_accuracy: 0.9880
Epoch 17/100
1688/1688 [==============================] - 16s 9ms/step - loss: 0.0261 - accuracy: 0.9938 - val_loss: 0.0521 - val_accuracy: 0.9893
Epoch 18/100
1688/1688 [==============================] - 16s 9ms/step - loss: 0.0213 - accuracy: 0.9944 - val_loss: 0.0788 - val_accuracy: 0.9875
Epoch 19/100
1688/1688 [==============================] - 16s 10ms/step - loss: 0.0238 - accuracy: 0.9944 - val_loss: 0.0883 - val_accuracy: 0.9860
Epoch 20/100
1688/1688 [==============================] - 16s 10ms/step - loss: 0.0279 - accuracy: 0.9934 - val_loss: 0.0740 - val_accuracy: 0.9897
Epoch 21/100
1688/1688 [==============================] - 16s 9ms/step - loss: 0.0191 - accuracy: 0.9954 - val_loss: 0.0573 - val_accuracy: 0.9905
Epoch 22/100
1688/1688 [==============================] - 16s 9ms/step - loss: 0.0258 - accuracy: 0.9946 - val_loss: 0.0876 - val_accuracy: 0.9852
313/313 [==============================] - 1s 3ms/step - loss: 0.0519 - accuracy: 0.9859
'''