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

# 앞선 trial에서도 확인했듯이, MNIST는 비교적 단순한 형태의 데이터기 때문에, vgg16처럼 깊은 CNN은 어울리지 않는다. 
# MNIST 데이터셋은 오래된 데이터셋으로, 이미지넷 대회가 나오기 전까지 컴퓨터 비전의 대표적인 데이터셋으로 사용되었다.
# MNIST가 핫할 때 이를 해결하는 최고의 알고리즘은 LeNet-5이다. 그러니 얘를 베이스로 업그레이드 해보자.


base_model =Sequential()
base_model.add(Conv2D(6, (3,3), activation='relu', input_shape=(28,28,1))) # 원본은 tanh인데 relu로 교체
base_model.add(MaxPooling2D())
    # 원랜 평균 풀링인데 요즘엔 평균 풀링에 파라미터가 없어서 그냥 최대풀링을 쓴다.(원본 LeNet은 파라미터가 있다.)
base_model.add(Conv2D(16, (5,5), activation='relu'))
base_model.add(MaxPooling2D())

base_model.add(Flatten())

base_model.add(Dense(120, activation='relu'))

base_model.add(Dense(84, activation='relu'))

base_model.add(Dense(10, activation = 'softmax'))

base_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True) 

history = base_model.fit(train_X, train_Y, epochs = 100, validation_data=(val_X, val_Y),callbacks=[early_stopping_cb])

base_model.evaluate(test_X, test_Y)

'''
Epoch 1/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.5041 - accuracy: 0.9071 - val_loss: 0.1220 - val_accuracy: 0.9642
Epoch 2/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.1025 - accuracy: 0.9705 - val_loss: 0.1075 - val_accuracy: 0.9687
Epoch 3/100
1688/1688 [==============================] - 7s 4ms/step - loss: 0.0757 - accuracy: 0.9774 - val_loss: 0.0894 - val_accuracy: 0.9777
Epoch 4/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.0630 - accuracy: 0.9811 - val_loss: 0.0879 - val_accuracy: 0.9780
Epoch 5/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.0543 - accuracy: 0.9835 - val_loss: 0.0828 - val_accuracy: 0.9775
Epoch 6/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.0486 - accuracy: 0.9854 - val_loss: 0.0728 - val_accuracy: 0.9785
Epoch 7/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.0411 - accuracy: 0.9879 - val_loss: 0.0894 - val_accuracy: 0.9778
Epoch 8/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.0398 - accuracy: 0.9882 - val_loss: 0.0594 - val_accuracy: 0.9837
Epoch 9/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.0346 - accuracy: 0.9893 - val_loss: 0.0664 - val_accuracy: 0.9840
Epoch 10/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.0284 - accuracy: 0.9913 - val_loss: 0.0927 - val_accuracy: 0.9813
Epoch 11/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.0305 - accuracy: 0.9911 - val_loss: 0.0737 - val_accuracy: 0.9842
Epoch 12/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.0269 - accuracy: 0.9921 - val_loss: 0.0917 - val_accuracy: 0.9810
Epoch 13/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.0277 - accuracy: 0.9919 - val_loss: 0.0986 - val_accuracy: 0.9793
Epoch 14/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.0281 - accuracy: 0.9920 - val_loss: 0.0770 - val_accuracy: 0.9855
Epoch 15/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.0234 - accuracy: 0.9933 - val_loss: 0.0774 - val_accuracy: 0.9845
Epoch 16/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.0249 - accuracy: 0.9924 - val_loss: 0.0649 - val_accuracy: 0.9865
Epoch 17/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.0244 - accuracy: 0.9934 - val_loss: 0.0807 - val_accuracy: 0.9827
Epoch 18/100
1688/1688 [==============================] - 8s 5ms/step - loss: 0.0233 - accuracy: 0.9938 - val_loss: 0.0898 - val_accuracy: 0.9855
313/313 [==============================] - 1s 2ms/step - loss: 0.0646 - accuracy: 0.9828
'''