import os.path as osp
import numpy as np
import cv2

from utils.heatmap import compute_heatmap
from utils.batch import prep_im_for_batch, im_list_to_batch
from utils.batch import crop_image, chromatic_transform
from config import cfg


def get_image_batch(db, scale_inds=None):
    num = len(db)
    processed_ims = []
    im_crops = []
    im_scales = []

    for i in xrange(num):
        im = cv2.imread(db[i]['image'])
        # Data Augmentation: Randomly flip the image.
        if db[i]['flipped']:
            im = im[:, ::-1, :]

        # Data Augmentation: Randomly crop the image.
        im, im_crop = crop_image(im, None, cfg.TRAIN.CROP_SCALE)
        im_crops.append(im_crop)

        # Data Augmentation: Chromatic transformation.
        if cfg.TRAIN.USE_CHROMATIC_CHANGE:
            im, _ = chromatic_transform(im)

        # Data Augmentation: Randomly scale the image.
        im, im_scale = prep_im_for_batch(
                im, cfg.PIXEL_MEANS, cfg.TRAIN.MAX_SIZE, cfg.TRAIN.MIN_SIZE,
                random=True)
        im_scales.append(im_scale)

        processed_ims.append(im)

    # Create a blob to hold the input images.
    max_shape = (cfg.TRAIN.MAX_SIZE, cfg.TRAIN.MAX_SIZE, 3)
    im_batch = im_list_to_batch(processed_ims, max_shape)

    return im_batch, im_crops, im_scales


def get_video_batch(db):
    length = cfg.TRAIN.MAX_LENGTH
    num = len(db)
    video_batch = np.zeros((length * num, 224, 224, 3), dtype=np.float32)
    valid_batch = np.zeros((length, num), dtype=np.bool)

    for i in xrange(num):
        video_dir = db[i]['video']
        assert osp.exists(video_dir)

        im_crop = None
        d_hsl = None

        for t in xrange(length):
            im_path = osp.join(video_dir, 'img{:05d}.png'.format(t+1))
            if not osp.exists(im_path):
                continue

            im = cv2.imread(im_path)

            # Data Augmentation: Randomly flip the video.
            if db[i]['flipped']:
                im = im[:, ::-1, :]

            # Data Augmentation: Randomly crop the image.
            im, im_crop = crop_image(im, im_crop, cfg.TRAIN.CROP_SCALE)

            # Data Augmentation: Chromatic transformation.
            if cfg.TRAIN.USE_CHROMATIC_CHANGE:
                im, d_hsl = chromatic_transform(im, d_hsl)

            im = im.astype(np.float32, copy=False)
            im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR)
            im -= cfg.PIXEL_MEANS
            video_batch[t * num + i] = im
            valid_batch[t, i] = True

    return video_batch, valid_batch


def get_heatmap_batch(db, im_crops, im_scales, max_shape):
    """ """
    assert len(db) == len(im_scales)

    num = len(db)
    heatmap_batch = np.zeros((num, max_shape[0], max_shape[1]),
                             dtype=np.float32)
    mask_batch = np.zeros(heatmap_batch.shape, dtype=np.float32)

    for i in xrange(num):
        im_crop = im_crops[i]
        im_scale = im_scales[i]
        points = db[i]['points']
        shape = db[i]['image_shape']
        heatmap = compute_heatmap(points, shape, im_crop, im_scale,
                                  cfg.KERNEL_SIZE_RATIO)
        heatmap_batch[i, :heatmap.shape[0], :heatmap.shape[1]] = heatmap
        mask_batch[i, :heatmap.shape[0], :heatmap.shape[1]] = 1.0

    return heatmap_batch, mask_batch


def get_point_batch(db, im_crops, im_scales):
    assert len(db) == len(im_scales)

    num = len(db)
    point_list = []

    for i in xrange(num):
        im_crop = im_crops[i]
        im_scale = im_scales[i]
        offset = (-im_crop[1], -im_crop[0])
        points = db[i]['points']
        points = resize_points(points, offset, im_scale)
        point_list.append(points[np.newaxis, :])

    point_batch = np.vstack(point_list)

    return point_batch


def resize_points(points, offset, scale):
    xs = points[:, 0]
    ys = points[:, 1]
    new_xs = (xs + offset[0]) * scale
    new_ys = (ys + offset[1]) * scale
    new_points = np.hstack([new_xs[:, np.newaxis], new_ys[:, np.newaxis]])
    return new_points


def flip_points(points, im_shape):
    xs = points[:, 0]
    ys = points[:, 1]
    new_xs = im_shape[0] - xs - 1
    new_points = np.hstack([new_xs[:, np.newaxis], ys[:, np.newaxis]])
    return new_points


def get_minibatch(db, debug=False):
    """Given sampled objects, construct a minibatch sampled from it."""
    # Get the input image batch, formatted for caffe.
    im_batch, im_crops, im_scales = get_image_batch(db)
    heatmap_batch, mask_batch = get_heatmap_batch(
            db, im_crops, im_scales, im_batch.shape[1:])
    point_batch = get_point_batch(db, im_crops, im_scales)

    # Build the minibatch.
    batch = {
            'image': im_batch,
            'point': point_batch,
            'heatmap': heatmap_batch,
            'mask': mask_batch
            }

    if cfg.VIDEO_ENCODER:
        video_batch, valid_batch = get_video_batch(db)
        batch['video'] = video_batch
        batch['valid'] = valid_batch

    return batch
