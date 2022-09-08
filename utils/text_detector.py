import cv2
import os
import keras_ocr
import numpy as np
import pytesseract
import easyocr
import matplotlib.pyplot as plt
from utils.settings import Settings

sett = Settings()


def easyocr_model_load():
    """
    docstr : load easyocr model
    return : easy ocr model
    """
    text_reader = easyocr.Reader(sett.ocr_lang_list) #Initialzing the ocr
    return text_reader

def easyocr_model_works(text_reader,images,visualization=False):
    
    """
    docstr : img to text function
    input : easyocr model
    return : None
    """
    texts=list()
    for i in range(len(images)):
        results = text_reader.recognize(images[i] ) # reader.recognize sadece recognize, text detection yok
        for (bbox, text, prob) in results:
            texts.append(text)
        if visualization:
            plt.imshow(images[i])
            plt.title("{} Image".format(str(i)));
            plt.show()
    return texts

def pytesseract_model_works(images,visualization=False):
    """
    docstr : pytesseract img to str model
    return : None
    """
    
    tesseract_preds = []
    for img in images:
        tesseract_preds.append(pytesseract.image_to_string(img))

    for i in range(len(images)):
        print(tesseract_preds[i])

        if visualization:
            plt.imshow(images[i])
            plt.title("{} Image".format(str(i)));
            plt.show()