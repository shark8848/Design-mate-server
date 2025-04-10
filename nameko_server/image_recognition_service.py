import io,sys
import json,base64

from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
from nameko.rpc import rpc
import pickle
sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog

class ImageRecognitionService(object):
    # 定义微服务名称
    name = "image_recognition_service"

    imagenet_class_index = json.load(open('./static/imagenet_class_index.json'))
    model = models.densenet121(pretrained=True)
    model.eval()

    def transform_image(self,image_bytes):
        my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        return my_transforms(image).unsqueeze(0)

    def get_prediction(self,image_bytes):
        tensor = self.transform_image(image_bytes=image_bytes)
        outputs = self.model.forward(tensor)
        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
        return self.imagenet_class_index[predicted_idx]

    @rpc
    def image_recognition_service(self,img_bytes): # 直接传入文件base64 编码的字节bytes
            class_id, class_name = self.get_prediction(io.BytesIO(base64.b64decode(img_bytes)).read())
            #return jsonify({'class_id': class_id, 'class_name': class_name})
            return 'class_id:' + class_id +' class_name:' + class_name
