import matplotlib.pyplot as plt
import numpy as np
import cv2
import copy
import os
from craft_text_detector import (read_image,load_craftnet_model,load_refinenet_model,
                                get_prediction,empty_cuda_cache)


def load_models():
    refine_net = load_refinenet_model(cuda=True)
    craft_net = load_craftnet_model(cuda=True)
    return refine_net,craft_net


def create_dir(_dir):
    """
    Creates given directory if it is not present.
    """
    if not os.path.exists(_dir):
        os.makedirs(_dir)


def read_image(image):
    if type(image) == str:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(image) == bytes:
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(image) == np.ndarray:
        if len(image.shape) == 2:  # grayscale
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            img = image
        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBAscale
            img = image[:, :, :3]

    return img


def export_detected_region(image, poly, rectify=True):
    """
    docstring : this func crop the detected images
    input : -> image : full image
    """
    if rectify:
        # rectify poly region
        result_rgb = rectify_poly(image, poly)
    else:
        result_rgb = crop_poly(image, poly)

    # export corpped region
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    return result_bgr


def export_detected_regions(
    image,
    regions,
    rectify: bool = False,
):
    """
    image: path to the image to be processed or numpy array or PIL image
    regions: list of bboxes or polys
    """

    # read/convert image
    image = read_image(image)

    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)

    # create crops dir
    #crops_dir = os.path.join(output_dir, file_name + "_crops")
    #create_dir(crops_dir)

    # init exported file paths
    #exported_file_paths = []
    
    cropped_images = []
    # export regions
    for ind, region in enumerate(regions):
        # get export path
        #file_path = os.path.join(crops_dir, "crop_" + str(ind) + ".png")
        # export region
        img = export_detected_region(image, poly=region, rectify=rectify)
        cropped_images.append(img)
        # note exported file path
        #exported_file_paths.append(file_path)

    return cropped_images


def export_extra_results(
    image,
    regions,
    heatmaps,
    texts=None,
):
    """initilaze text detection result one by one
    image: path to the image to be processed or numpy array or PIL image
    boxes (array): array of result file

    """
    # read/convert image
    image = read_image(image)
    # result directory
    #res_file = os.path.join(output_dir, file_name + "_text_detection.txt")
    #res_img_file = os.path.join(output_dir, file_name + "_text_detection.png")
    #text_heatmap_file = os.path.join(output_dir, file_name + "_text_score_heatmap.png")
    #link_heatmap_file = os.path.join(output_dir, file_name + "_link_score_heatmap.png")""

    # export heatmaps
    heatmap=(heatmaps["text_score_heatmap"])
    link_score_heatmap=(heatmaps["link_score_heatmap"])

    for i, region in enumerate(regions):
            region = np.array(region).astype(np.int32).reshape((-1))

            region = region.reshape(-1, 2)
            cv2.polylines(
                image,
                [region.reshape((-1, 1, 2))],
                True,
                color=(0, 0, 255),
                thickness=2,
            )

            if texts is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                cv2.putText(
                    image,
                    "{}".format(texts[i]),
                    (region[0][0] + 1, region[0][1] + 1),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness=1,
                )
                cv2.putText(
                    image,
                    "{}".format(texts[i]),
                    tuple(region[0]),
                    font,
                    font_scale,
                    (0, 255, 255),
                    thickness=1,
                )

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR),heatmap,link_score_heatmap


def rectify_poly(img, poly):
    # Use Affine transform
    n = int(len(poly) / 2) - 1
    width = 0
    height = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k + 1], poly[-k - 2], poly[-k - 1]])
        width += int(
            (np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])) / 2
        )
        height += np.linalg.norm(box[1] - box[2])
    width = int(width)
    height = int(height / n)

    output_img = np.zeros((height, width, 3), dtype=np.uint8)
    width_step = 0
    for k in range(n):
        box = np.float32([poly[k], poly[k + 1], poly[-k - 2], poly[-k - 1]])
        w = int((np.linalg.norm(box[0] - box[1]) + np.linalg.norm(box[2] - box[3])) / 2)

        # Top triangle
        pts1 = box[:3]
        pts2 = np.float32(
            [[width_step, 0], [width_step + w - 1, 0], [width_step + w - 1, height - 1]]
        )
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(
            img, M, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)
        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        output_img[warped_mask == 1] = warped_img[warped_mask == 1]

        # Bottom triangle
        pts1 = np.vstack((box[0], box[2:]))
        pts2 = np.float32(
            [
                [width_step, 0],
                [width_step + w - 1, height - 1],
                [width_step, height - 1],
            ]
        )
        M = cv2.getAffineTransform(pts1, pts2)
        warped_img = cv2.warpAffine(
            img, M, (width, height), borderMode=cv2.BORDER_REPLICATE
        )
        warped_mask = np.zeros((height, width, 3), dtype=np.uint8)
        warped_mask = cv2.fillConvexPoly(warped_mask, np.int32(pts2), (1, 1, 1))
        cv2.line(
            warped_mask, (width_step, 0), (width_step + w - 1, height - 1), (0, 0, 0), 1
        )
        output_img[warped_mask == 1] = warped_img[warped_mask == 1]

        width_step += w
    return output_img