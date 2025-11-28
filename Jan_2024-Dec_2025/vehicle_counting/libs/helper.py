# helper.py
# Helper functions for video frame processing and utilities

import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import subprocess
from tqdm import tqdm
import IPython
from IPython.display import Video, display

def risize_frame(frame, scale_percent):
    """
        Resize an image by a percentage scale.
        Args:
            frame : numpy.ndarray
                Input image frame in BGR format (as read by OpenCV).
            scale_percent : float or int
                Percentage scaling factor applied to both width and height.
                For example, `50` reduces the image size to 50% of the original.
        Returns
            numpy.ndarray
                The resized image.

        Notes:
            This function uses `cv2.INTER_AREA` interpolation, which is optimal
            when reducing image resolution. If `scale_percent` is 100, the image
            size remains unchanged.
    """
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

def output_video(model, path, scale_percent, 
                 video_name='result.mp4', VIDEO_CODEC = "MP4V"):
    """
        Run object detection on a video, count vehicle movements, annotate frames,
        and generate an output video with visual overlays and summary statistics.

        Args:
            model : object
                A YOLO model instance supporting `.predict()` and containing
                `model.names` for class label lookup.
            path : str
                Path to the input video file to be processed.
            scale_percent : int or float
                Percentage used to downscale each video frame. Downscaling
                improves inference performance but may reduce line–pixel accuracy.
            video_name : str, optional
                Name of the final output video saved to disk (default "result.mp4").
            VIDEO_CODEC : str, optional
                FourCC video codec for OpenCV `VideoWriter`. Default is "MP4V".

        Returns:
            None
                The function writes video files to disk and displays an embedded
                video object via the notebook environment.

        Description:
            The function performs the following steps:

                1. Load the input video using OpenCV.
                2. Optionally rescale all frames based on `scale_percent`.
                3. Detect vehicles using the YOLO model (restricted to specific classes).
                4. Draw bounding boxes, class labels, confidence scores, and centroid dots.
                5. Draw a horizontal counting line used to determine `in` versus `out`
                vehicle movement based on centroid crossing.
                6. Maintain per-class and total counters for inbound/outbound vehicles.
                7. Save each processed frame to a temporary output video.
                8. Convert the temporary output to a browser-friendly H.264 video using ffmpeg.
                9. Display the embedded resulting video.

        Vehicle Counting Logic:
            - The horizontal counting line is at y = `cy_linha`.
            - A vehicle is counted when its bounding-box centroid lies within
            `(line ± offset)` pixels vertically.
            - Vehicles on the left side of the frame are counted as inbound,
            and vehicles on the right side are counted as outbound.

        Side Effects:
            - Creates temporary video files (`tmp_*.mp4`).
            - Overwrites existing output files if names collide.
            - Prints informational messages about frame size, fps, and scaling.
            - Uses tqdm for progress visualization.

        Notes: 
            - If `scale_percent != 100`, pixel-based geometry such as bounding
            boxes and counting lines will shift proportionally.
            - `Video()` at the end assumes execution inside a Jupyter/Colab-like
            notebook environment.
            - ffmpeg must be installed and available in the system path.
    """
    video = cv2.VideoCapture(path)

    #========== Configurations
    #Verbose during prediction
    verbose = False
    output_path = "rep_" + video_name
    tmp_output_path = "tmp_" + output_path
    
    # Scaling percentage of original frame
    scale_percent = 50

    # Objects to detect Yolo
    class_IDS = [2, 3, 5, 7] 
    # Auxiliary variables
    veiculos_contador_in = dict.fromkeys(class_IDS, 0)
    veiculos_contador_out = dict.fromkeys(class_IDS, 0)
    frames_list = []
    cy_linha = int(1500 * scale_percent/100 )
    cx_sentido = int(2000 * scale_percent/100) 
    offset = int(8 * scale_percent/100 )
    contador_in = 0
    contador_out = 0
    print(f'[INFO] - Verbose during Prediction: {verbose}')


    # Original informations of video
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video.get(cv2.CAP_PROP_FPS)
    print('[INFO] - Original Dim: ', (width, height), 'with fps:', fps, 'n_frames:', 
          int(video.get(cv2.CAP_PROP_FRAME_COUNT)))

    # Scaling Video for better performance 
    if scale_percent != 100:
        print('[INFO] - Scaling change may cause errors in pixels lines ')
        width = int(width * scale_percent / 100)
        height = int(height * scale_percent / 100)
        print('[INFO] - Dim Scaled: ', (width, height))
        
    ### Video output ####
    output_video = cv2.VideoWriter(tmp_output_path, 
                                cv2.VideoWriter_fourcc(*VIDEO_CODEC), 
                                fps, (width, height))
    
    # Executing Recognition 
    for i in tqdm(range(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT)), 1)):    
        # reading frame from video
        _, frame = video.read()
        
        #Applying resizing of read frame
        frame  = risize_frame(frame, scale_percent)
        
        if verbose:
            print('Dimension Scaled(frame): ', (frame.shape[1], frame.shape[0]))

        # Getting predictions
        y_hat = model.predict(frame, conf = 0.7, classes = class_IDS, device = 0, verbose = False)
        
        # Getting the bounding boxes, confidence and classes of the recognize objects in the current frame.
        boxes   = y_hat[0].boxes.xyxy.cpu().numpy()
        conf    = y_hat[0].boxes.conf.cpu().numpy()
        classes = y_hat[0].boxes.cls.cpu().numpy() 
        
        # Storing the above information in a dataframe
        positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.data, 
                                       columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])
        
        #geting names from classes
        dict_classes = model.model.names

        #Translating the numeric class labels to text
        labels = [dict_classes[i] for i in classes]
        
        # Drawing transition line for in\out vehicles counting (x1, y1) to (x2, y2)
        cv2.line(frame, (0, cy_linha), (int(4500 * scale_percent/100 ), cy_linha), (255,255,0),8)
        
        # For each vehicles, draw the bounding-box and counting each one the pass thought the transition line (in\out)
        for ix, row in enumerate(positions_frame.iterrows()):
            # Getting the coordinates of each vehicle (row)
            xmin, ymin, xmax, ymax, confidence, category,  = row[1].astype('int')
            
            # Calculating the center of the bounding-box
            center_x, center_y = int(((xmax+xmin))/2), int((ymax+ ymin)/2)
            
            # drawing center and bounding-box of vehicle in the given frame 
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,0,0), 5) # box
            cv2.circle(frame, (center_x,center_y), 5,(255,0,0),-1) # center of box
            
            #Drawing above the bounding-box the name of class recognized.
            cv2.putText(img=frame, text=labels[ix]+' - '+str(np.round(conf[ix],2)),
                        org= (xmin,ymin-10), 
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=2)
            
            # Checking if the center of recognized vehicle is in the area given by the transition line + offset and transition line - offset 
            if (center_y < (cy_linha + offset)) and (center_y > (cy_linha - offset)):
                if  (center_x >= 0) and (center_x <=cx_sentido):
                    contador_in +=1
                    veiculos_contador_in[category] += 1
                else:
                    contador_out += 1
                    veiculos_contador_out[category] += 1
        
        #updating the counting type of vehicle 
        contador_in_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_in.items()]
        contador_out_plt = [f'{dict_classes[k]}: {i}' for k, i in veiculos_contador_out.items()]
        # cy_linha (750) là tọa độ của trục y còn cx_sentido là tọa độ của x
        
        #drawing the number of vehicles in\out
        cv2.putText(img=frame, text='N. vehicles In', 
                    org= (30,30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=1, color=(255, 0, 0),thickness=1)
        
        cv2.putText(img=frame, text='N. vehicles Out', 
                    org= (int(2800 * scale_percent/100 ),30), 
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=1)
        
        #drawing the counting of type of vehicles in the corners of frame 
        xt = 40
        for txt in range(len(contador_in_plt)):
            xt +=30
            cv2.putText(img=frame, text=contador_in_plt[txt], 
                        org= (30,xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                        fontScale=1, color=(255, 0, 0),thickness=1)
            
            cv2.putText(img=frame, text=contador_out_plt[txt],
                        org= (int(2800 * scale_percent/100 ),xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1, color=(0, 255, 0),thickness=1)
        
        #drawing the number of vehicles in\out
        cv2.putText(img=frame, text=f'In:{contador_in}', 
                    org= (int(1820 * scale_percent/100 ),cy_linha+60),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=2)
        
        cv2.putText(img=frame, text=f'Out:{contador_out}', 
                    org= (int(1800 * scale_percent/100 ),cy_linha-40),
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=2)

        if verbose:
            print(contador_in, contador_out)
        #Saving frames in a list 
        frames_list.append(frame)
        #saving transformed frames in a output video format
        output_video.write(frame)

    #Releasing the video    
    output_video.release()

    ####  pos processing
    # Fixing video output codec to run in the notebook\browser
    if os.path.exists(output_path):
        os.remove(output_path)
        
    subprocess.run(
        ["ffmpeg",  "-i", tmp_output_path,"-crf","18","-preset","veryfast","-hide_banner","-loglevel","error","-vcodec","libx264",output_path])
    os.remove(tmp_output_path)

    #output video result
    frac = 0.7 
    Video(data='rep_result.mp4', embed=True, height=int(720 * frac), width=int(1280 * frac))