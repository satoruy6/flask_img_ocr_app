import cv2
import numpy as np
from transformers import pipeline
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def hello_world():
    img_dir = "static/imgs/"
    if request.method == 'GET':
        img_path=None
        final_results=[]
    elif request.method == 'POST':
        #### POSTにより受け取った画像を読み込む
        stream = request.files['img'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)
        #### img.jpgを名前として「static/imgs/」に保存する
        img_path = img_dir + "img.jpg"
        cv2.imwrite(img_path, img)
        ### 画像を認識する
        classifier = pipeline(model="microsoft/beit-base-patch16-224-pt22k-ft22k")
        results = classifier(img_dir + "img.jpg")
        final_results=[]
        n_top = 3
        for result in results[:n_top]:
            final_result = str(round(result["score"]*100, 2)) + "%の確率で" + result["label"] + "です。"
            final_results.append(final_result)
    #### 保存した画像ファイルのpathをHTMLに渡す
    return render_template('index.html', img_path=img_path, final_results=final_results) 

if __name__ == "__main__":
    app.run(debug=False)