import os.path
import numpy as np
import PIL
from sets import Set


class Dataset(object):

    def __init__(self, data_dir, num_points=10):
        self._data_dir = data_dir
        self._num_pointss = num_points
        self._db = None
        self._video_names = Set()
        self._image_names = Set()

    @ property
    def data_dir(self):
        return self._data_dir

    @ property
    def num_pointss(self):
        return self._num_pointss

    @property
    def db(self):
        if self._db is not None:
            return self._db

        self._db = self.load_db()

        return self._db

    @ property
    def video_names(self):
        return self._video_names

    @ property
    def image_names(self):
        return self._image_names

    def load_db(self):
        """ """
        db = self._load_db()

        print('Totally {} annotations in the dataset.'.format(len(db)))

        return db

    def _load_db(self):
        all_db = []

        filename = os.path.abspath(
                os.path.join(self.data_dir, 'annotations.txt'))

        with open(filename, 'r') as fin:
            for line in fin:
                items = line.split(' ')

                points = []
                for i in xrange(self.num_pointss):
                    x = float(items[4 + 2*i])
                    y = float(items[4 + 2*i + 1])
                    points.append([x, y])

                points = np.array(points, dtype=np.float32)

                entry = {
                        'channel': items[0],
                        'playlist': items[1],
                        'video': items[2],
                        'start_time': items[3],
                        'duration': items[4],
                        'image': items[5],
                        'image_shape': (float(items[6]), float(items[7])),
                        'points': np.array(points, dtype=np.float32),
                        }

                video_name = os.path.join(
                        entry['channel'], entry['playlist'], entry['video'],
                        '%s_%s.mp4' % (entry['start_time'], entry['duration']))
                image_name = os.path.join(
                        entry['channel'], entry['playlist'], entry['video'],
                        entry['image'])

                all_db.append(entry)

                # Save video names and image names.
                self._video_names.add(video_name)
                self._image_names.add(image_name)

        # Save video names and image names
        self._video_names = list(self._video_names)
        self._image_names = list(self._image_names)

        return all_db

    def get_widths(self):
        self.db
        return [PIL.Image.open(self.image_path_at(name)).size[0]
                for name in self._image_names]

    def append_flipped_images(self):
        n_db = len(self.db)
        for i in xrange(n_db):
            db_i = self.db[i]
            points = db_i['points'].copy()
            points[:, 0] = db_i['image_shape'][0] - points[:, 0] - 1
            entry = {
                    'video': db_i['video'],
                    'image': db_i['image'],
                    'image_shape': db_i['image_shape'],
                    'points': points,
                    'flipped': True
                    }
            self._db.append(entry)
        self._video_names = self._video_names * 2
        self._image_names = self._image_names * 2

    def image_path_at(self, filename):
        image_path = os.path.join(self.data_dir, 'images', filename)

        assert os.path.exists(image_path), (
                'Path does not exist: %s' % (image_path))

        return image_path

    def video_path_at(self, filename):
        video_name = os.path.splitext(os.path.basename(filename))[0]
        video_path = os.path.join(self.data_dir, 'videos', video_name)

        assert os.path.exists(video_path), (
                'Path does not exist: %s' % (video_path))

        return video_path
