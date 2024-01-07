import os
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import seaborn as sb
import cv2
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Loading Our final trained Knn model 
model = open("knn_coconut_min_max.pkl", "rb")
knn = joblib.load(model)

# create title
st.title("Coconut Quality Classification App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    temp_image_path = 'temp_image.jpg'
    image.save(temp_image_path)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    rataR = []
    rataG = []
    rataB = []
    rataH = []
    rataS = []
    rataV = []
    ratagray = []
    stand = []

    luas = []
    keliling = []

    contrast = []
    dissimilarity = []
    homogeneity = []
    energy = []
    correlation = []
    
    gbr_read = cv2.imread(temp_image_path)
    gbr_rgb = cv2.cvtColor(gbr_read, cv2.COLOR_BGR2RGB)
    (R, G, B) = cv2.split(gbr_rgb)
    meanR = np.mean(R)
    rataR.append(meanR)
    meanG = np.mean(G)
    rataG.append(meanG)
    meanB = np.mean(B)
    rataB.append(meanB)
    
    gbr_hsv = cv2.cvtColor(gbr_read, cv2.COLOR_BGR2HSV)
    H = gbr_hsv[:, :, 0]
    S = gbr_hsv[:, :, 1]
    V = gbr_hsv[:, :, 2]
    meanH = np.mean(H)
    rataH.append(meanH)
    meanS = np.mean(S)
    rataS.append(meanS)
    meanV = np.mean(V)
    rataV.append(meanV)

    gbr_gray = cv2.cvtColor(gbr_read, cv2.COLOR_BGR2GRAY)
    rata = np.mean(gbr_gray)
    ratagray.append(rata)
    standar = np.std(gbr_gray)
    stand.append(standar)  
    
    _, thresh = cv2.threshold(gbr_gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter_val = cv2.arcLength(contour, True)
    else:
        area = 0
        perimeter_val = 0
    
    luas.append(area)
    keliling.append(perimeter_val)

    # Ekstraksi Fitur Tekstur (GLCM)
    glcm = graycomatrix(gbr_gray, [1], [0], 256, symmetric=True, normed=True)
    contrast_val = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity_val = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity_val = graycoprops(glcm, 'homogeneity')[0, 0]
    energy_val = graycoprops(glcm, 'energy')[0, 0]
    correlation_val = graycoprops(glcm, 'correlation')[0, 0]
    
    contrast.append(contrast_val)
    dissimilarity.append(dissimilarity_val)
    homogeneity.append(homogeneity_val)
    energy.append(energy_val)
    correlation.append(correlation_val)

    data1 = pd.DataFrame(rataR, columns=['Mean-R'])
    data2 = pd.DataFrame(rataG, columns=['Mean-G'])
    data3 = pd.DataFrame(rataB, columns=['Mean-B'])
    data4 = pd.DataFrame(rataH, columns=['Mean-H'])
    data5 = pd.DataFrame(rataV, columns=['Mean-V'])
    data6 = pd.DataFrame(rataS, columns=['Mean-s'])
    data7 = pd.DataFrame(ratagray, columns=['Mean-Gray'])
    data8 = pd.DataFrame(stand, columns=['Standar-Deviasi'])
    data9 = pd.DataFrame(luas, columns=['Luas'])
    data10 = pd.DataFrame(keliling, columns=['Keliling'])
    data11 = pd.DataFrame(contrast, columns=['Contrast'])
    data12 = pd.DataFrame(dissimilarity, columns=['Dissimilarity'])
    data13 = pd.DataFrame(homogeneity, columns=['Homogeneity'])
    data14 = pd.DataFrame(energy, columns=['Energy'])
    data15 = pd.DataFrame(correlation, columns=['Correlation'])

    listdata = [data1, data2, data3, data4, data5, data6, data7, data8, data9, 
                data10, data11, data12, data13, data14, data15]
    x = pd.concat(listdata, axis=1, ignore_index=True)
    
    st.write(x) 
    y_prediksi = knn.predict(x)
    
    if y_prediksi == 0:
        st.write("<p style='text-align: center;'><center>Kelapa Standar</center></p>", unsafe_allow_html=True)
    elif y_prediksi == 1:
        st.write("<p style='text-align: center;'><center>Kelapa Tidak Standar</center></p>", unsafe_allow_html=True)
    else:
        st.write("<p style='text-align: center;'><center>error</center></p>", unsafe_allow_html=True)