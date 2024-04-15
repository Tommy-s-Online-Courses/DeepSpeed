import requests
import base64
import json
from PIL import Image

# 打开图像

with open("test_images/img01.jpg", "rb") as f:
    img_bytes = f.read()
    
# 转换为 base64 编码

img_b64 = base64.b64encode(img_bytes).decode("utf-8")

# 请求服务
response = requests.post("http://0.0.0.0:8000/predict", json= {"img_base64": img_b64})

print(response.json())