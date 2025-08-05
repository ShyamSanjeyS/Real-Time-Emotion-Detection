from pathlib import Path
import numpy as np
import argparse
import time
import os
import torch.backends.cudnn as cudnn
import torch
import cv2

from emotion import detect_emotion, init
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    scale_coords, set_logging, create_folder
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


def detect(opt):
    source, view_img, imgsz, nosave, show_conf, save_path, show_fps = opt.source, not opt.hide_img, opt.img_size, opt.no_save, not opt.hide_conf, opt.output_path, opt.show_fps
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    create_folder(save_path)
    set_logging()
    device = select_device(opt.device)
    init(device)
    half = device.type != 'cpu'

    model = attempt_load("weights/yolov7-tiny.pt", map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()

    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = ((0,52,255),(121,3,195),(176,34,118),(87,217,255),(69,199,79),(233,219,155),(203,139,77),(214,246,255))

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()
    img_saved = False  # Save one frame image

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                images = []

                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
                    images.append(im0.astype(np.uint8)[int(y1):int(y2), int(x1):int(x2)])
                
                if images:
                    emotions = detect_emotion(images, show_conf)

                i = 0
                for *xyxy, conf, cls in reversed(det):
                    if view_img or not nosave:
                        label = emotions[i][0]
                        colour = colors[emotions[i][1]]
                        i += 1
                        plot_one_box(xyxy, im0, label=label, color=colour, line_thickness=opt.line_thickness)

            # Save single image frame
            if not img_saved:
                frame_path = os.path.join(save_path if os.path.isdir(save_path) else os.path.dirname(save_path), "frame.jpg")
                cv2.imwrite(frame_path, im0)
                cv2.imshow("Captured Image", cv2.resize(im0, (im0.shape[1] // 2, im0.shape[0] // 2)))
                img_saved = True

            if view_img:
                display_img = cv2.resize(im0, (im0.shape[1]*2, im0.shape[0]*2))
                cv2.imshow("Emotion Detection", display_img)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    print("\n[INFO] Exiting webcam view.")
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    cv2.destroyAllWindows()
                    return

            if not nosave:
                ext = save_path.split(".")[-1]
                if ext in ["mp4", "avi"]:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
                elif ext.lower() in ["jpg", "jpeg", "png", "bmp", "tiff"]:
                    cv2.imwrite(save_path, im0)
                else:
                    output_path = os.path.join(save_path, os.path.split(path)[1])
                    create_folder(output_path)
                    cv2.imwrite(output_path, im0)

            if show_fps:
                print(f"FPS: {1/(time.time()-t0):.2f}" + " "*5, end="\r")
                t0 = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source (0 for webcam)')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold')
    parser.add_argument('--device', default='', help='cuda device or cpu')
    parser.add_argument('--hide-img', action='store_true', help='hide image results')
    save = parser.add_mutually_exclusive_group()
    save.add_argument('--output-path', default="output.mp4", help='save path')
    save.add_argument('--no-save', action='store_true', help='do not save images/videos')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--show-fps', default=False, action='store_true', help='print fps')
    opt = parser.parse_args()
    check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        detect(opt=opt)
