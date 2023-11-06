# ADD EMSEMBLE
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import pickle
 
# 1. Load
mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test_original, Y_test_original) = mnist.load_data()
 
# 2. CNN은 채널 필요
X_train_full, X_test_original = X_train_full.reshape(-1, 28, 28, 1), X_test_original.reshape(-1, 28, 28, 1)
 
# 3. 보통 CNN은 zero-centering만 해준다.
    # per-channel pixel mean(VGGNet)을 빼거나 그냥 전체 mean(AlexNet)을 빼주면 되는데, MNIST는 채널이 1개라서 상관 없다.
    # normalize 안함 : 이미지 데이터에서 feature의 범위가 달라서 모델에 악영향을 미치지 않기 떄문.
    # PCA 안함 : 상관성 분석 필요 없음
    # Whitening 안함 : whitening : feature의 상관성을 없애주고 분산을 1로 만드는 것. 필요없음
 
X_train_full, X_test_original = X_train_full/1.0, X_test_original/1.0 # float로 변경해줌
avg = np.mean(X_train_full, axis = 0)
X_train_full -= avg
    # training data의 mean value를 다 빼주면서 zero-centering
X_test_original -= avg
 
 
# 6. Data Augmentation
 
datagen = ImageDataGenerator(
    rotation_range=15, # 대충 1을 한 15도 정도까지 기울여서 써도 알아보니까 15도로 설정
    width_shift_range=0.1,
    height_shift_range=0.1, # 너무 크게하면 7이랑 1이랑 너무 비슷할듯 하니까 작게 설정
    shear_range=0.0, # 삐뚜로 쓰는것도 넣으면 rotation이랑 겹쳐서 너무 왜곡될 수 있으니 안한다.
    zoom_range=0.1, # 크기 조절
)
 
nets = 10
model = [0]*nets
history = [0]*nets
test = [0]*nets

for i in range(nets):
    # 4. Data Split
    train_X, val_X, train_Y, val_Y = \
        train_test_split(X_train_full, y_train_full, random_state = i, test_size = 0.1, stratify = y_train_full)
 
    # 5. OneHot
    train_Y, val_Y, test_Y = \
        to_categorical(train_Y, num_classes=10), to_categorical(val_Y, num_classes=10), to_categorical(Y_test_original, num_classes=10)
 
    # 6. Data Augmentation
    datagen.fit(train_X)
 
    # 7. Model
    model[i] = Sequential()
    model[i].add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
    model[i].add(BatchNormalization())
    model[i].add(MaxPooling2D())
 
    model[i].add(Conv2D(32, (3,3), activation='relu'))
    model[i].add(BatchNormalization())
    model[i].add(MaxPooling2D())
 
    model[i].add(Conv2D(64, (5,5), activation='relu', padding = 'same'))
    model[i].add(BatchNormalization())
    model[i].add(MaxPooling2D())
 
    model[i].add(Conv2D(128, (5,5), activation='relu', padding = 'same'))
    model[i].add(BatchNormalization())
    model[i].add(MaxPooling2D())
 
    model[i].add(Flatten())
 
    model[i].add(Dense(120, activation='relu'))
 
    model[i].add(Dense(10, activation = 'softmax'))
 
    model[i].compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
 
    early_stopping_cb = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True)
    checkpoint_cb = keras.callbacks.ModelCheckpoint("MNIST_Final_"+str(i)+".h5", save_best_only = True)
 
    history[i] = model[i].fit(datagen.flow(train_X, train_Y), epochs = 100, validation_data=(val_X, val_Y),\
        callbacks=[early_stopping_cb, checkpoint_cb])
    
    test[i] = model[i].evaluate(X_test_original, test_Y)


   
results = np.zeros( (X_test_original.shape[0],10) )
 
for i in range(nets):
 
    print("CNN {0:d}: Train accuracy={1:.5f}, Validation accuracy={2:.5f}, Test accuracy={3:.5f}".\
        format(i+1,max(history[i].history['accuracy']),max(history[i].history['val_accuracy']),test[i][1]))
    # 예측값을 다 더한다음에 가장 큰 값의 인덱스를 뽑아내서 앙상블
    results = results + model[i].predict(X_test_original)
 
 
# Reverse to_categorical
actual = Y_test_original
prediction = np.argmax(results, axis = 1)
 

# Accuracy 구하기 + 틀린애들 뭔지 확인하기
correct = 0
wrong_indices = []
wrong_pred = []

for i in range(actual.shape[0]):
    if Y_test_original[i] == prediction[i]:
        correct += 1
    else:
        wrong_indices.append(i)
        wrong_pred.append(prediction[i])


accuracy = correct/10000

print("="*50)
print("Final Ensembled Score : Accuracy = {0:.5f}".format(accuracy))

from matplotlib import pyplot as plt
print(wrong_indices)
plt.figure(figsize=(15,10))
num = 0
X_test_original += avg # 가독성을 위해 zero-centering 제거

for i in wrong_indices:
    plt.subplot(5, 10, num+1)
    plt.imshow(X_test_original[i].reshape((28,28)),cmap=plt.cm.binary)
    plt.title("pred = "+ str(wrong_pred[num])+"\nans = "+str(Y_test_original[i]), y=1.0)
    plt.axis('off')
    num += 1

plt.subplots_adjust(wspace=0.3, hspace=-0.1)
plt.savefig("Mistaken.jpg")

with open("MNIST_ensemble_model.pickle", 'wb') as pickle_data:
    pickle.dump(model, pickle_data)
 
with open("MNIST_ensemble_history.pickle", 'wb') as pickle_data:
    pickle.dump(history, pickle_data)


'''
This Computer
CNN 1: Train accuracy=0.99291, Validation accuracy=0.99483, Test accuracy=0.99470
CNN 2: Train accuracy=0.99376, Validation accuracy=0.99400, Test accuracy=0.99230
CNN 3: Train accuracy=0.99261, Validation accuracy=0.99400, Test accuracy=0.99410
CNN 4: Train accuracy=0.99389, Validation accuracy=0.99400, Test accuracy=0.99390
CNN 5: Train accuracy=0.99378, Validation accuracy=0.99483, Test accuracy=0.99420
CNN 6: Train accuracy=0.99467, Validation accuracy=0.99433, Test accuracy=0.99370
CNN 7: Train accuracy=0.99478, Validation accuracy=0.99433, Test accuracy=0.99470
CNN 8: Train accuracy=0.99480, Validation accuracy=0.99450, Test accuracy=0.99450
CNN 9: Train accuracy=0.99456, Validation accuracy=0.99567, Test accuracy=0.99430
CNN 10: Train accuracy=0.99491, Validation accuracy=0.99417, Test accuracy=0.99380
==================================================
Final Ensembled Score : Accuracy = 0.99740
[359, 659, 1232, 1260, 1621, 1790, 1901, 2040, 2130, 2462, 2597, 2939, 3422, 3762, 4176, 4740, 4761, 4823, 5654, 5937, 5997, 6576, 6625, 8408, 9642, 9729]
'''


'''
회의실 컴퓨터
CNN 1: Train accuracy=0.99409, Validation accuracy=0.99467, Test accuracy=0.99440
CNN 2: Train accuracy=0.99474, Validation accuracy=0.99417, Test accuracy=0.99490
CNN 3: Train accuracy=0.99387, Validation accuracy=0.99383, Test accuracy=0.99470
CNN 4: Train accuracy=0.99556, Validation accuracy=0.99450, Test accuracy=0.99510
CNN 5: Train accuracy=0.99444, Validation accuracy=0.99600, Test accuracy=0.99460
CNN 6: Train accuracy=0.99317, Validation accuracy=0.99317, Test accuracy=0.99380
CNN 7: Train accuracy=0.99357, Validation accuracy=0.99367, Test accuracy=0.99430
CNN 8: Train accuracy=0.99333, Validation accuracy=0.99333, Test accuracy=0.99330
CNN 9: Train accuracy=0.99485, Validation accuracy=0.99483, Test accuracy=0.99420
CNN 10: Train accuracy=0.99269, Validation accuracy=0.99333, Test accuracy=0.99180
==================================================
Final Ensembled Score : Accuracy = 0.99700
[582, 659, 674, 1232, 1260, 1393, 1438, 1790, 1901, 2130, 2462, 2597, 2654, 3422, 3762, 4176, 4740, 4761, 5654, 5937, 5955, 5997, 6576, 6597, 6625, 8408, 9642, 9679, 9692, 9729]
'''