
from tensorflow import keras
import numpy as np

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
 

nets = 10

model = []
results = np.zeros( (X_test_original.shape[0],10) )

for i in range(nets):
    model.append(keras.models.load_model\
        (r"C:\Users\user\Desktop\MNIST_project\models\FINAL_MODEL\trial1\MNIST_Final_"+str(i)+".h5"))

for i in range(nets):
    model.append(keras.models.load_model\
        (r"C:\Users\user\Desktop\MNIST_project\models\FINAL_MODEL\trial2\MNIST_Final_"+str(i)+".h5"))
    
for i in range(2*nets):
    results = results + model[i].predict(X_test_original)


# Reverse to_categorical
actual = Y_test_original
prediction = np.argmax(results, axis = 1)

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

print("===Mistaken Classes===")

for i in range(len(wrong_indices)):
    print(Y_test_original[wrong_indices[i]], end = ', ')

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
