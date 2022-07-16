import cv2
import streamlit as st

from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, letterbox
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


ROOT = Path.cwd()
imgsz = 640
conf_thres = 0.25
iou_thres = 0.45
max_det = 1000
classes = None
agnostic_nms = False
line_thickness = 2
device = select_device('cpu')
half = False


def get_model(weights=ROOT / 'yolov3.pt',  # model.pt path(s)
              device=None,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
              view_img=False,  # show results
              save_txt=False,  # save results to *.txt
              save_conf=False,  # save confidences in --save-txt labels
              save_crop=False,  # save cropped prediction boxes
              nosave=False,  # do not save images/videos
              augment=False,  # augmented inference
              visualize=False,  # visualize features
              update=False,  # update all models
              project=ROOT / 'runs/detect',  # save results to project/name
              name='exp',  # save results to project/name
              exist_ok=False,  # existing project/name ok, do not increment
              hide_labels=False,  # hide labels
              hide_conf=False,  # hide confidences
              dnn=False,  # use OpenCV DNN for ONNX inference
            ):

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    # device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, pt = model.stride, model.pt

    # Half
    half1 = half & pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half1 else model.model.float()

    model.eval()

    return model


@torch.no_grad()
def get_drawing_array(image_array):
    """
    input:
          image_array: image array RGB size 512 x 512 from webcam

    output:
          drawing_array: image RGBA size 512 x 512 only contain bounding box and text,
                              channel A value = 255 if the pixel contains drawing properties (lines, text)
                              else channel A value = 0
    """
    drawing_array = np.zeros([512,512,4], dtype=np.uint8)
    img = letterbox(image_array, new_shape=imgsz)[0]

    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # (0 - 255) to (0.0 - 1.0)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)
    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

    # Process detections
    det = pred[0]

    annotator = Annotator(drawing_array, line_width=line_thickness, example=str(names))

    if det is not None and len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image_array.shape).round()

        # Write results
        for *xyxy, conf, cls in det:
            label = '%s %.2f' % (names[int(cls)], conf)
            annotator.box_label(xyxy, label, color=colors(int(cls), True))

    drawing_array[:,:,3] = (drawing_array.max(axis = 2) > 0 ).astype(int) * 255

    return drawing_array


if __name__ == '__main__':

    model = get_model()
    names = model.names

    st.title("Webcam - YOLOv3 Test")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print(type(frame), frame.shape, frame)

        frame_detected = get_drawing_array(frame)
        # print(type(frame_detected), frame_detected.shape, frame_detected)

        frame = cv2.resize(frame, (512, 512))
        merged = frame + frame_detected[:, :, :3]
        FRAME_WINDOW.image(merged)

    else:
        st.write('Stopped')
