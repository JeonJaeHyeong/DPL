import torch
import torchvision.transforms as TT

from pysgg.config import cfg
from pysgg.data import transforms as T
from pysgg.structures.image_list import to_image_list
from pysgg.structures.bounding_box import BoxList
from pysgg.modeling.roi_heads.box_head.inference import make_roi_box_post_processor


def im_detect_bbox_aug(model, images, device):
    # Collect detections computed under different transformations
    boxlists_ts = []
    for _ in range(len(images)):
        boxlists_ts.append([])

    def add_preds_t(boxlists_t):
        for i, boxlist_t in enumerate(boxlists_t):
            if len(boxlists_ts[i]) == 0:
                # The first one is identity transform, no need to resize the boxlist
                boxlists_ts[i].append(boxlist_t)
            else:
                # Resize the boxlist as the first one
                boxlists_ts[i].append(boxlist_t.resize(boxlists_ts[i][0].size))

    # Compute detections for the original image (identity transform)
    boxlists_i = im_detect_bbox(
        model, images, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, device
    )
    add_preds_t(boxlists_i)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        boxlists_hf = im_detect_bbox_hflip(
            model, images, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST, device
        )
        add_preds_t(boxlists_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES:
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        boxlists_scl = im_detect_bbox_scale(
            model, images, scale, max_size, device
        )
        add_preds_t(boxlists_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            boxlists_scl_hf = im_detect_bbox_scale(
                model, images, scale, max_size, device, hflip=True
            )
            add_preds_t(boxlists_scl_hf)

    # Merge boxlists detected by different bbox aug params
    boxlists = []
    for i, boxlist_ts in enumerate(boxlists_ts):
        bbox = torch.cat([boxlist_t.bbox for boxlist_t in boxlist_ts])
        scores = torch.cat([boxlist_t.get_field('scores') for boxlist_t in boxlist_ts])
        boxlist = BoxList(bbox, boxlist_ts[0].size, boxlist_ts[0].mode)
        boxlist.add_field('scores', scores)
        boxlists.append(boxlist)

    if cfg.GLOBAL_SETTING.DATASET_CHOICE == 'VG':
        num_classes = cfg.MODEL.ROI_BOX_HEAD.VG_NUM_CLASSES
    elif cfg.GLOBAL_SETTING.DATASET_CHOICE == 'GQA_200':
        num_classes = cfg.MODEL.ROI_BOX_HEAD.GQA_200_NUM_CLASSES
    else:
        num_classes = None
        exit('wrong Dataset choice')
    # Apply NMS and limit the final detections
    results = []
    post_processor = make_roi_box_post_processor(cfg)
    for boxlist in boxlists:
        results.append(post_processor.filter_results(boxlist, num_classes))

    return results


def im_detect_bbox(model, images, target_scale, target_max_size, device):
    """
    Performs bbox detection on the original image.
    """
    transform = TT.Compose([
        T.Resize(target_scale, target_max_size),
        TT.ToTensor(),
        T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255
        )
    ])
    images = [transform(image) for image in images]
    images = to_image_list(images, cfg.DATALOADER.SIZE_DIVISIBILITY)
    return model(images.to(device))


def im_detect_bbox_hflip(model, images, target_scale, target_max_size, device):
    """
    Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    transform = TT.Compose([
        T.Resize(target_scale, target_max_size),
        TT.RandomHorizontalFlip(1.0),
        TT.ToTensor(),
        T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255
        )
    ])
    images = [transform(image) for image in images]
    images = to_image_list(images, cfg.DATALOADER.SIZE_DIVISIBILITY)
    boxlists = model(images.to(device))

    # Invert the detections computed on the flipped image
    boxlists_inv = [boxlist.transpose(0) for boxlist in boxlists]
    return boxlists_inv


def im_detect_bbox_scale(model, images, target_scale, target_max_size, device, hflip=False):
    """
    Computes bbox detections at the given scale.
    Returns predictions in the scaled image space.
    """
    if hflip:
        boxlists_scl = im_detect_bbox_hflip(model, images, target_scale, target_max_size, device)
    else:
        boxlists_scl = im_detect_bbox(model, images, target_scale, target_max_size, device)
    return boxlists_scl
