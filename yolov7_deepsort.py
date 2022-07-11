import os
import cv2
import time
import argparse
import torch
import warnings
import numpy as np

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results

from PIL import Image


class VideoTracker(object):
    def __init__(self, args, video_path):
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector()
        self.deepsort = build_tracker(use_cuda=True)
        self.class_names = self.detector.class_names

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

 
    def run(self):
        results = []
        idx_frame = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            ref, ori_im = self.vdo.retrieve()

            if ref is True:
                im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
                #----- do detection
                frame = Image.fromarray(np.uint8(im))
                bbox_xywh, cls_conf, cls_ids = self.detector.new_detect(frame)
                if cls_conf is not None:
                    #-----copy
                    list_fin = []
                    for i in bbox_xywh:
                        temp = []
                        temp.append(i[0])
                        temp.append(i[1])
                        temp.append(i[2]*1.)
                        temp.append(i[3]*1.)
                        list_fin.append(temp)
                    new_bbox = np.array(list_fin).astype(np.float32)

                    #-----#-----mask processing filter the useless part
                    mask = [0,1,2,3,5,7]# keep specific classes the indexes are corresponded to coco_classes
                    mask_filter = []
                    for i in cls_ids:
                        if i in mask:
                            mask_filter.append(1)
                        else:
                            mask_filter.append(0)
                    new_cls_conf = []
                    new_new_bbox = []
                    new_cls_ids = []
                    for i in range(len(mask_filter)):
                        if mask_filter[i]==1:
                            new_cls_conf.append(cls_conf[i])
                            new_new_bbox.append(new_bbox[i])
                            new_cls_ids.append(cls_ids[i])
                    new_bbox =  np.array(new_new_bbox).astype(np.float32)
                    cls_conf =  np.array(new_cls_conf).astype(np.float32)
                    cls_ids  =  np.array(new_cls_ids).astype(np.float32) 
                    #-----#-----

                    # do tracking
                    outputs = self.deepsort.update(new_bbox, cls_conf, im)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_tlwh = []
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        ori_im = draw_boxes(ori_im, bbox_xyxy, identities)
                        for bb_xyxy in bbox_xyxy:
                            bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
                        results.append((idx_frame - 1, bbox_tlwh, identities))

                    end = time.time()

                    if self.args.display:
                        cv2.imshow("test", ori_im)
                        cv2.waitKey(1)

                    if self.args.save_path:
                        self.writer.write(ori_im)

                    # save results
                    write_results(self.save_results_path, results, 'mot')

                    # logging
                    self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                                    .format(end - start, 1 / (end - start), new_bbox.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with VideoTracker( args, video_path='./001.avi') as vdo_trk:
        vdo_trk.run()
