import platform
from enum import Enum
import os
import torch


class Settings:
    def __init__(self):
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        self.profibus_host = ""
        self.profibus_port = ""
        self.plc_ip = "192.168.0.1"
        self.plc_rack = 0
        self.plc_slot = 1
        self.stm32_port = ""
        self.stm32_baudrate = 9600

        self.display_width = 1920
        self.display_height = 1080

        self.backend_url = ""

        self.model_name = "qcai"
        # self.label = os.path.join(self.root, "ai", self.model_name, 'label.txt')
        self.model = os.path.join(self.root, "ai", self.model_name, 'yolov5n.pt')
        self.model_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_box_to_draw = 17
        self.min_confidence = 0.3

        self.ocr_lang_list=['tr','en']
        self.ocr_device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.ocr_xconfig = "-c page_separator=''"

        self.pred_resize_w = 640
        self.pred_resize_h = 480
        self.vis_resize_w = 640
        self.vis_resize_h = 480

        self.use_trt_models = False

        self.camera_type = CAMERA_TYPE.USB
        self.video_source = 0

        self.camera_type_1 = CAMERA_TYPE.USB
        self.video_source_1 = 'C:/Users/berat/Desktop/2.mp4'

        self.camera_type_2 = CAMERA_TYPE.USB
        self.video_source_2 = 'C:/Users/berat/Desktop/1.mp4'

        self.illumination_type = ILLUMINATION_TYPE.ONE_SHOT

        self.color_move_on = (255, 200, 90)
        self.text_x_align = 10
        self.inference_time_y = 30
        self.fps_y = 60
        self.analysis_time_y = 90
        self.font_scale = 0.7
        self.thickness = 2
        self.rect_thickness = 3
        self.frame_count = 0
        self.hide_conf = True

        self.ui_components_background_color = '#FFFFFF',
        self.show_frame_delay = 10

        self.save_img_count_mod = 20
        self.passwd = 'Moveon.1261.1'

        if platform.system() == 'Windows':
            self.run_platform = DEVICE.COMPUTER
        elif platform.uname().node == "moveonboxer81701":
            self.run_platform = DEVICE.BOXER
        elif platform.uname().node == "moveonjn1":
            self.run_platform = DEVICE.NANO
        elif platform.uname().node == "akdeniz":
            self.run_platform = DEVICE.NANO
        elif platform.uname().node == "mef-FX503VD":
            self.run_platform = DEVICE.NANO

    def inc_frame_count(self):
        self.frame_count = self.frame_count + 1


class DEVICE:
    NANO = 0
    BOXER = 1
    COMPUTER = 2


class ILLUMINATION_TYPE:
    ONE_SHOT = 0
    PHOTOMETRIC_STEREO = 1


class STM32_CONTROL_TYPE(Enum):
    LED1_ON = 0
    LED1_OFF = 1
    LED2_ON = 2
    LED2_OFF = 3
    LED3_ON = 4
    LED3_OFF = 5
    LED4_ON = 6
    LED4_OFF = 7
    LEDALL_ON = 8
    LEDALL_OFF = 9

    def name(self):
        if self is STM32_CONTROL_TYPE.LED1_ON:
            return "l11"
        if self is STM32_CONTROL_TYPE.LED1_OFF:
            return "l10"

        if self is STM32_CONTROL_TYPE.LED2_ON:
            return "l21"
        if self is STM32_CONTROL_TYPE.LED2_OFF:
            return "l20"

        if self is STM32_CONTROL_TYPE.LED3_ON:
            return "l31"
        if self is STM32_CONTROL_TYPE.LED3_OFF:
            return "l30"

        if self is STM32_CONTROL_TYPE.LED4_ON:
            return "l41"
        if self is STM32_CONTROL_TYPE.LED4_OFF:
            return "l40"

        if self is STM32_CONTROL_TYPE.LEDALL_ON:
            return "la1"
        if self is STM32_CONTROL_TYPE.LEDALL_OFF:
            return "la0"


class CAMERA_TYPE:
    PI = 0
    USB = 1
    BASLER = 2


class CONTROL_TYPE(Enum):
    IDLE = 0
    CAPTURE = 1
    QCAI_DETECT = 2
    DATA_COLLECT = 3
    CLOSE = 4
    PLC_TRIG = 5

    def name(self):
        if self is CONTROL_TYPE.IDLE:
            return "IDLE"

        if self is CONTROL_TYPE.CAPTURE:
            return "CAPTURE"

        if self is CONTROL_TYPE.QCAI_DETECT:
            return "QCAI_DETECT"

        if self is CONTROL_TYPE.DATA_COLLECT:
            return "DATA_COLLECT"

        if self is CONTROL_TYPE.PLC_TRIG:
            return "VIDEO CAPTURE"

    def len(self):
        return 7


class MESSAGE_TYPES(Enum):
    INITIATING = 0
    FRAME_GRABBING = 1
    IDLE = 2
    CAPTURE = 3
    QCAI_DETECT = 4
    DATA_COLLECT = 5
    UNABLE_VIDEO_SOURCE = 6
    CLOSE = 7

    def name(self):
        if self is MESSAGE_TYPES.INITIATING:
            return "INITIATING"

        if self is MESSAGE_TYPES.FRAME_GRABBING:
            return "FRAME GRABBING"

        if self is MESSAGE_TYPES.IDLE:
            return "IDLE"

        if self is MESSAGE_TYPES.CAPTURE:
            return "CAPTURE"

        if self is MESSAGE_TYPES.QCAI_DETECT:
            return "QCAI DETECT"

        if self is MESSAGE_TYPES.DATA_COLLECT:
            return "DATA COLLECTION"

        if self is MESSAGE_TYPES.UNABLE_VIDEO_SOURCE:
            return "Unable to open video source"

        if self is MESSAGE_TYPES.CLOSE:
            return "CLOSE"
