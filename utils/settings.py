import platform
from enum import Enum
import os
import torch


class Settings:
    def __init__(self):
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

       
        self.display_width = 1920
        self.display_height = 1080

        self.backend_url = ""

        self.max_box_to_draw = 17
        self.min_confidence = 0.3

        self.ocr_lang_list=['tr','en']
        self.ocr_device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ocr_xconfig = "-c page_separator=''"

      
        self.color = (255, 200, 90)
        self.text_x_align = 10
        self.inference_time_y = 30
        self.fps_y = 60
        self.analysis_time_y = 90
        self.font_scale = 0.7
        self.thickness = 2
        self.rect_thickness = 3
        self.frame_count = 0
        self.hide_conf = True

        
