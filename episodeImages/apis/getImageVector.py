from io import BytesIO

import requests
from flask import Flask, request, jsonify
from keras.applications.mobilenet import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
import numpy as np
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# 加载预训练的MobileNet模型
model = MobileNet(weights='imagenet', include_top=False)


# 处理图像并获取特征向量
def get_feature_vector(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    feature_vector = features.flatten()  # 将特征向量展平为一维数组
    return feature_vector.tolist()


# Flask路由和视图函数
@app.route('/get_feature_vector', methods=['POST'])
def get_feature_vector_api():
    print("api....")
    image_url = request.json.get('image_url')
    print(image_url)

    # # 从图像URL获取图像文件
    # response = requests.get(image_url)
    # image_file = BytesIO(response.content)
    #
    # # 保存上传的图像到服务器临时文件夹或其他位置，然后获取图像路径
    # image_path = request.json.get('image_path')  # 替换为图像的实际路径
    # with open(image_path, 'wb') as f:
    #     f.write(image_file.read())

    # 获取图像特征向量
    image_path = '../test/ysz.jpg'
    features = get_feature_vector(image_path)
    print(features)
    print(len(features))
    return jsonify({'features': features}), 200, {'Content-Type': 'application/json'}


if __name__ == '__main__':
    app.run()
