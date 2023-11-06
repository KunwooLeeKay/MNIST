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

# 2. CNN은 채널 필요
X_train_full, test_X = X_train_full.reshape(-1, 28, 28, 1), test_X.reshape(-1, 28, 28, 1)
print(X_train_full.shape, test_X.shape)

# 3. 보통 CNN은 zero-centering만 해준다. 
    # per-channel pixel mean(VGGNet)을 빼거나 그냥 전체 mean(AlexNet)을 빼주면 되는데, MNIST는 채널이 1개라서 상관 없다.
    # normalize 안함 : 이미지 데이터에서 feature의 범위가 달라서 모델에 악영향을 미치지 않기 떄문.
    # PCA 안함 : 상관성 분석 필요 없음
    # Whitening 안함 : whitening : feature의 상관성을 없애주고 분산을 1로 만드는 것. 필요없음

X_train_full, test_X = X_train_full/1.0, test_X/1.0 # float로 변경해줌
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
# 2. conv 레이어를 하나 늘려보자. 32 -> 64 -> 128을 그냥하면 크기땜에 음수되서 padding = same까지 넣어준다. 

model =Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1))) 
model.add(MaxPooling2D())

model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(128, (5,5), activation='relu', padding = 'same'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(120, activation='relu'))

model.add(Dense(84, activation='relu'))

model.add(Dense(10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True) 

history = model.fit(train_X, train_Y, epochs = 100, validation_data=(val_X, val_Y),callbacks=[early_stopping_cb])

model.evaluate(test_X, test_Y)

# 0.9852 -> 0.9876
'''
Epoch 1/100
1688/1688 [==============================] - 25s 14ms/step - loss: 0.2428 - accuracy: 0.9409 - val_loss: 0.0802 - val_accuracy: 0.9777
Epoch 2/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0780 - accuracy: 0.9766 - val_loss: 0.1109 - val_accuracy: 0.9673
Epoch 3/100
1688/1688 [==============================] - 25s 15ms/step - loss: 0.0638 - accuracy: 0.9819 - val_loss: 0.0571 - val_accuracy: 0.9842
Epoch 4/100
1688/1688 [==============================] - 25s 15ms/step - loss: 0.0590 - accuracy: 0.9829 - val_loss: 0.0536 - val_accuracy: 0.9850
Epoch 5/100
1688/1688 [==============================] - 25s 15ms/step - loss: 0.0508 - accuracy: 0.9859 - val_loss: 0.0725 - val_accuracy: 0.9790
Epoch 6/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0451 - accuracy: 0.9872 - val_loss: 0.0652 - val_accuracy: 0.9842
Epoch 7/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0418 - accuracy: 0.9886 - val_loss: 0.0534 - val_accuracy: 0.9877
Epoch 8/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0359 - accuracy: 0.9900 - val_loss: 0.0696 - val_accuracy: 0.9822
Epoch 9/100
1688/1688 [==============================] - 24s 14ms/step - loss: 0.0350 - accuracy: 0.9902 - val_loss: 0.0533 - val_accuracy: 0.9903
Epoch 10/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0333 - accuracy: 0.9911 - val_loss: 0.0474 - val_accuracy: 0.9882
Epoch 11/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0307 - accuracy: 0.9921 - val_loss: 0.0530 - val_accuracy: 0.9883
Epoch 12/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0324 - accuracy: 0.9919 - val_loss: 0.0575 - val_accuracy: 0.9900
Epoch 13/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0277 - accuracy: 0.9932 - val_loss: 0.1127 - val_accuracy: 0.9800
Epoch 14/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0250 - accuracy: 0.9939 - val_loss: 0.0887 - val_accuracy: 0.9865
Epoch 15/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0270 - accuracy: 0.9933 - val_loss: 0.0654 - val_accuracy: 0.9860
Epoch 16/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0297 - accuracy: 0.9930 - val_loss: 0.1416 - val_accuracy: 0.9805
Epoch 17/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0313 - accuracy: 0.9929 - val_loss: 0.0773 - val_accuracy: 0.9897
Epoch 18/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0255 - accuracy: 0.9943 - val_loss: 0.1050 - val_accuracy: 0.9848
Epoch 19/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0196 - accuracy: 0.9949 - val_loss: 0.0581 - val_accuracy: 0.9913
Epoch 20/100
1688/1688 [==============================] - 23s 14ms/step - loss: 0.0349 - accuracy: 0.9930 - val_loss: 0.1166 - val_accuracy: 0.9755
313/313 [==============================] - 1s 4ms/step - loss: 0.0544 - accuracy: 0.9876
'''