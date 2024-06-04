import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# 모델 불러오기
model_path = '/Users/daeyeonwon/Documents/git_plactice/cifar10_resnet50_model.h5'
model = tf.keras.models.load_model(model_path)

# CIFAR-10 클래스 이름
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 고양이 이미지 불러오기 및 전처리
img_path = '/Users/daeyeonwon/Documents/git_plactice/pexels-lina-1741205.jpg'  # 여기에 고양이 이미지 경로를 넣으세요.
img = image.load_img(img_path, target_size=(32, 32)) #기본 32,32
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 1000  # 이미지 정규화 #255.0

# 모델 예측
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])
predicted_class_name = class_names[predicted_class]

print(f'The model predicts that the image is a: {predicted_class_name}')
