
from utils.craft_segmenter import (create_dir,rectify_poly,export_detected_region,
                            export_detected_regions,export_extra_results,load_models,read_image)

from craft_text_detector import (load_craftnet_model,load_refinenet_model
                                ,get_prediction,empty_cuda_cache)

from utils.text_detector import (easyocr_model_load,easyocr_model_works,
                        pytesseract_model_works)

import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import os
import csv



def allocate_ocr_models():
    refine_net,craft_net = load_models() # craft text detector models
    text_reader=easyocr_model_load() # text recog modelsimage = 'test_.jpeg' # can array, image or PIL !

    return refine_net, craft_net, text_reader


def ocr_prediction(image,craft_net,refine_net):
    prediction_result = get_prediction(
    image=image,
    craft_net=craft_net,
    refine_net=refine_net
)

    boxes = (prediction_result['boxes'])
    heatmaps = (prediction_result['heatmaps'])

    cropped_images =exported_file_paths = export_detected_regions(
        image=image,
        regions=prediction_result["boxes"],
        rectify=True
    )

    return boxes, heatmaps,cropped_images

def ocr_text_reader(text_reader,cropped_images):
    texts=easyocr_model_works(text_reader, cropped_images)
    return texts


def write_csv(words):


    with open("words.csv", mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows([words])