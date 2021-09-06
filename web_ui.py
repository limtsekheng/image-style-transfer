from flask import Flask
from model import StyleTransfer
from io import BytesIO
import base64
from PIL import Image
import json
from flask import request, render_template
import os


class Processor():
    def __init__(self, input_content_path, input_style_path):
        self.model= StyleTransfer()
        self.content_path = input_content_path
        self.style_path = input_style_path

    def transfer(self):
        img = Image.open(self.content_path)
        content, res = self.model.obtain_result(img, self.style_path)
        content.save('static/images/content.jpg')
        res.save('static/images/res.jpg')


app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return '''<form action="/" method="post">
              <p> Enter the input image file path: </p>

              <p><input name="image_path"></p>

              <p> Choose a style: </p>
              <select id="style" name = "style">
              <option value="Starry Night">StarryNight</option>
              <option value="Anime">Anime</option>
              </select>

              <p><button type="submit">Process</button></p>

              </form>'''


@app.route('/', methods=['POST'])
def display():
    if request.form['image_path']:

        # full_filename = 'static/images/tubingen.jpg'  # input 'static/images/tubingen.jpg' into the textbox
        full_filename = request.form['image_path']

        if request.form['style'] == 'Starry Night':
            style_filename = 'static/images/starry_night.jpg'
        else:
            style_filename = 'static/images/japan.jpg'


        content_filename = 'static/images/content.jpg'
        res_filename = 'static/images/res.jpg'
        
        processor = Processor(full_filename, style_filename)
        processor.transfer()
        return render_template("index.html", user_image = content_filename, result_image = res_filename)
    return '<h3>No input path entered.</h3>'





if __name__ == '__main__':
    app.run(host= '127.0.0.1', port=5001, debug=True)
