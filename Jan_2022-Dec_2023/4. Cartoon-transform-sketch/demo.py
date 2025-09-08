import warnings
import cv2
import easygui 
import numpy as np 
import imageio
from sklearn.cluster import KMeans
import sys
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image

warnings.filterwarnings("ignore")

def KMean_transform(img, nb_clusters=6):
    """
        Image segmentation is the classification of an image into different groups. 
        Many kinds of research have been done in the area of image segmentation 
        using clustering, here is KMeans.
        This will reduce the number of color-range from the original images and make
        your output look like a cartoon image.
    """
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

    return out_img

def smoothing_image(img, size=(960, 640)):
    """
        Smoothing image by using median-bluring into the gray-scale image
    """
    ReSized1 = cv2.resize(img, size)

    # converting an image to grayscale
    grayScaleImage= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ReSized2 = cv2.resize(grayScaleImage, size)

    # applying median blur to smoothen an image
    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
    ReSized3 = cv2.resize(smoothGrayScale, size)

    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 9, 9)

    ReSized4 = cv2.resize(getEdge, size)

    return ReSized1, getEdge, ReSized4

def to_cartoon(ImagePath, nb_clusters, size=(960, 640)):
    """
        Wrapping up all the techniques (gray-scale + Smoothing + KMeans-clustering)
    """
    # read the image
    originalmage = cv2.imread(ImagePath)
    originalmage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2RGB)
    #print(image)  # image is stored in form of numbers

    # confirm that image is chosen
    if originalmage is None:
        print("Can not find any image. Choose appropriate file")
        sys.exit()

    # retrieving the edges for cartoon effect by using thresholding technique
    ReSized1, getEdge, ReSized4 = smoothing_image(originalmage, size)

    # applying bilateral filter to remove noise & keep edge sharp as required
    colorImage = cv2.bilateralFilter(originalmage, 5, 300, 300)
    ReSized5 = cv2.resize(colorImage, size)
    
    # masking edged image with our "BEAUTIFY" image
    cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)
    ReSized6 = KMean_transform(cv2.resize(cartoonImage, size), nb_clusters)

    # Plotting the whole transition
    images=[ReSized1, ReSized6]

    fig, axes = plt.subplots(1, 2, figsize=(8,8), 
                             subplot_kw={'xticks':[], 'yticks':[]}, 
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    titles = ["Input", "Transformed"]
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(titles[i])

    save1 = Button(top,text = "Save your transformed cartoon image",
                 command=lambda: save(ReSized6, ImagePath),
                 padx = 30, pady = 5)
    save1.configure(background='#364156', 
                    foreground='white',
                    font=('calibri',10,'bold'))
    save1.pack(side = TOP, pady = 50)    
    plt.show()

def upload():
    ImagePath = easygui.fileopenbox()
    to_cartoon(ImagePath, n_clusters)
    
def save(ReSized6, ImagePath):

    newName = "image_to_cartoon"
    path1 = os.path.dirname(ImagePath)
    extension = os.path.splitext(ImagePath)[1]
    path = os.path.join(path1, newName+extension)
    cv2.imwrite(path, cv2.cvtColor(ReSized6, cv2.COLOR_RGB2BGR))
    I = "Image saved by name " + newName +" at "+ path
    tk.messagebox.showinfo(title = None, message = I)

if __name__ == "__main__":
    n_clusters = int(input("Input the number of clusters:"))

    top = tk.Tk()
    top.geometry('400x400')
    top.title('Cartoonify Your Image !')
    top.configure(background='lightblue')
    label=Label(top,
                background='#d6bbcb', 
                font = ('consolas', 20, 'bold'))

    upload=Button(top,text="Cartoonify an Image", 
                  command = upload,
                  padx = 10, pady = 5)
    upload.configure(background = '#e1cec6', 
                     foreground = 'black', 
                     font = ('consolas', 10, 'bold'))
    upload.pack(side = TOP, pady = 50)

    top.mainloop()