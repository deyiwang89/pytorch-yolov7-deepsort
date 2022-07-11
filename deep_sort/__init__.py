from .deep_sort import DeepSort


__all__ = ['DeepSort', 'build_tracker']


def build_tracker(use_cuda):
    return DeepSort('./deep_sort/deep/checkpoint/ckpt.t7',# namesfile=cfg.DEEPSORT.CLASS_NAMES,
                max_dist=0.2, min_confidence=0.1, 
                nms_max_overlap=0.5, max_iou_distance=0.7, 
                max_age=70, n_init=3, nn_budget=100, use_cuda=True)
    

# def build_tracker(cfg, use_cuda):
#     return DeepSort(cfg.DEEPSORT.REID_CKPT,# namesfile=cfg.DEEPSORT.CLASS_NAMES,
#                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, 
#                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, 
#                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=use_cuda)
    








