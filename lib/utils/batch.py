"""Batch helper functions."""

import os
import os.path as osp
import numpy as np
import numpy.random as npr
import cv2


def prep_im_for_batch(im, pixel_means, max_size=None, min_size=None,
                      random=False):
    """Mean subtract and scale an image for use in a batch."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    min_im_size = np.min(im_shape[0:2])
    max_im_size = np.max(im_shape[0:2])

    # Prevent the biggest axis from being more than MAX_SIZE.
    if max_size is None:
        max_scale = 1.0
    else:
        max_scale = float(max_size) / float(max_im_size)

    # Prevent the smallest axis from being less than MIN_SIZE.
    if min_size is None:
        min_scale = 1.0
    else:
        min_scale = float(min_size) / float(min_im_size)

    # Choose the image scaling factor.
    assert max_scale >= min_scale
    if random:
        im_scale = np.exp(npr.uniform(np.log(min_scale), np.log(max_scale)))
    else:
        im_scale = max(min_scale, min(max_scale, 1.0))

    # Resize the image.
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale


def im_list_to_batch(ims, max_shape=None):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    n_ims = len(ims)
    max_im_shape = np.array([im.shape for im in ims]).max(axis=0)

    if max_shape is None:
        max_shape = max_im_shape
    else:
        assert (max_im_shape[0] <= max_shape[0]) and \
                (max_im_shape[1] <= max_shape[1]), \
                'max_shape: {} max_im_shape: {}'.format(max_shape, max_im_shape)

    batch = np.zeros((n_ims, max_shape[0], max_shape[1], max_shape[2]),
                     dtype=np.float32)

    for i in xrange(n_ims):
        im = ims[i]
        batch[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return batch


def video_read(video_dir, max_length=None):
    assert osp.exists(video_dir)

    if max_length:
        length = max_length
    else:
        length = len(os.listdir(video_dir))

    video = []

    for t in xrange(length):
        im_path = osp.join(video_dir, 'img{:05d}.png'.format(t+1))

        if not osp.exists(im_path):
            continue

        im = cv2.imread(im_path)
        video.append(im)

    return video


def crop_image(im, box=None, min_scale=None, random=False):
    if box is None:
        assert min_scale <= 1
        w = im.shape[0]
        h = im.shape[1]
        w_scale = npr.uniform(min_scale, 1.0)
        h_scale = npr.uniform(min_scale, 1.0)
        w_offset = npr.randint(0, w * (1.0 - w_scale) + 1)
        h_offset = npr.randint(0, h * (1.0 - h_scale) + 1)
        new_w = int(w * w_scale)
        new_h = int(h * h_scale)
        box = [w_offset, h_offset, new_w, new_h]

    new_im = im[box[0]:box[0]+box[2], box[1]:box[1]+box[3]]

    return new_im, box


def chromatic_transform(im, d_hsl=None):
    """
    Given an image array, add the hue, saturation and luminosity to the image
    """
    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1.
    if d_hsl is None:
        d_h = (np.random.rand(1) - 0.5) * 0.1 * 180
        d_l = (np.random.rand(1) - 0.5) * 0.2 * 256
        d_s = (np.random.rand(1) - 0.5) * 0.2 * 256
        d_hsl = (d_h, d_s, d_l)
    else:
        d_h, d_s, d_l = d_hsl

    # Convert the BGR to HLS.
    hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)

    # Add the values to the image H, L, S.
    new_h = (h + d_h) % 180
    new_l = np.clip(l + d_l, 0, 255)
    new_s = np.clip(s + d_s, 0, 255)

    # Convert the HLS to BGR.
    new_hls = cv2.merge((new_h, new_l, new_s)).astype('uint8')
    new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2BGR)

    return new_im, d_hsl
