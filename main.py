

from utils import main_functions,craft_segmenter,settings,text_detector
from utils.main_functions import *

if __name__ == "__main__":

    image="kanban.png"
    image = read_image(image)
    
    refine_net, craft_net, text_reader = allocate_ocr_models()
    boxes,heatmaps ,cropped_images = ocr_prediction(image,craft_net,refine_net)
    texts = ocr_text_reader(text_reader,cropped_images)
    write_csv(texts)
    img_show,heatmap,link_score_heatmap = export_extra_results(image=image,regions=boxes,heatmaps=heatmaps)
    
    cv2.imshow("OCR Result of the Kanban Cards",img_show)
    cv2.waitKey(0)


empty_cuda_cache()

