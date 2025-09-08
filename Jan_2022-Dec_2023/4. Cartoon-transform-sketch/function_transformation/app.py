import pandas as pd, numpy as np
from flask import Flask, render_template, request
import cv2, skimage, os
from transformed import *
from werkzeug.utils import secure_filename

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('home.html')

print(f"success, current directory = {os.getcwd()}")

@app.route('/' , methods = ['GET', 'POST'])
def main():

    if request.method == 'POST':
        img_file = request.files['file']
        transform_type = request.form['transformed_type']
        print(f"input image name : {img_file.filename} \t transformed_type: {transform_type}")
        img_name = os.path.join('static', 'input.png')
        img_file.save(os.path.join('static', 'input.png'))

        output_im = transformed(img_name, type = transform_type)
        oput_fname = os.path.join('static', 'output.png')
        cv2.imwrite(oput_fname, (output_im*255).astype(np.uint8))
        print(f"output image name : {oput_fname}")

        return render_template('home.html', 
                                uploaded_image = img_name, 
                                transformed_img = oput_fname
                            )

if __name__ == "__main__":
    app.run(port='8080', threaded=False, debug=True)