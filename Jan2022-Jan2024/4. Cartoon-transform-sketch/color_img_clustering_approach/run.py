import cv2, os
import numpy as np
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans
from flask import Flask, render_template, request

def cartoon_segment(img_path, nb_clusters, transtype):
    """
        Parameters:
            img_path (directory): directory to the loading image
            nb_clusters (int): number of cluster added to the image segmentation
    """
    nb_clusters = int(nb_clusters)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    ## Flatten the image_array
    X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
    kmeans = KMeans(n_clusters = nb_clusters)
    kmeans.fit(X)

    ## create labels and centroids
    label = kmeans.predict(X)
    temp_img = np.zeros_like(X)

    # replace each pixel by its center
    for k in range(nb_clusters):
        centroids_val =  np.uint8(kmeans.cluster_centers_[k])
        temp_img[label == k] = centroids_val 

    out_img = temp_img.reshape(img.shape[0], img.shape[1], img.shape[2])
    
    # sketch
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)
    out_img = cv2.adaptiveThreshold(out_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 15)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2RGB)

    if transtype == 'blue_form':
        out_img[:,:,1] = 0
        out_img[:,:,2] = 0
    elif transtype == 'red_form':
        out_img[:,:,0] = 0
        out_img[:,:,1] = 0
    elif transtype == 'green_form':
        out_img[:,:,0] = 0
        out_img[:,:,2] = 0
    else: 
        out_img[:,:,0] = 255
        out_img[:,:,1] = 0
        out_img[:,:,2] = 255

    return cv2.bitwise_or(out_img, temp_img.reshape(img.shape[0], img.shape[1], img.shape[2]))

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/' , methods = ['GET', 'POST'])
def main():

    if request.method == 'POST':
        img_file = request.files['file']
        nb_cluster = request.form['nb_cluster']
        transformed_type = request.form['transformed_type']

        img_name = os.path.join('static', 'input.png')
        img_file.save(os.path.join('static', 'input.png'))

        output_im = cartoon_segment(img_name, nb_cluster, transformed_type)
        oput_fname = os.path.join('static', 'output.png')
        cv2.imwrite(oput_fname, (output_im).astype(np.uint8))
        print(f"output image name : {oput_fname}")

        return render_template('home.html', 
                                uploaded_image = img_name, 
                                transformed_img = oput_fname
                            )

if __name__ == "__main__":
    app.run(port='8080', threaded=False, debug=True)