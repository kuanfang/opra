import numpy as np
import cv2


def prepare_db(dataset):
    """
    Prepare the db for training.
    """
    all_db = dataset.db
    for db in all_db:
        # 1) Set the path to the clipped video and the target image.
        image_name = db['image']
        video_name = db['video']
        db['image'] = dataset.image_path_at(image_name)
        db['video'] = dataset.video_path_at(video_name)

        # 2) Resize the annotations to align with the target image.
        im = cv2.imread(db['image'])
        _image_shape = db['image_shape']
        if (im.shape[0] != _image_shape[0]) or (im.shape[1] != _image_shape[1]):
            db['points'] = resize_points(db['points'], _image_shape, im.shape)
        db['image_shape'] = (im.shape[0], im.shape[1])


def resize_points(points, src_shape, dst_shape):
    xs = points[:, 0]
    ys = points[:, 1]
    new_xs = xs * dst_shape[0] / src_shape[0]
    new_ys = ys * dst_shape[1] / src_shape[1]
    new_points = np.hstack([new_xs[:, np.newaxis], new_ys[:, np.newaxis]])

    return new_points
