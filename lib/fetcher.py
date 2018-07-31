import numpy as np
from multiprocessing import Process, Queue

from minibatch import get_minibatch
from config import cfg


QUEUE_MAXSIZE = 16
N_PROCS = 8
VERBOSE = False


def _fetch(mgr, db, debug):
    # Get the indices of the minibatch
    inds = mgr.get_next_minibatch_inds()
    sampled_db = [db[i] for i in inds]

    # Get the minibatch
    batch_data = get_minibatch(sampled_db, debug)

    return batch_data


class IndexManager(object):
    """Managing the minibatch indices."""

    def __init__(self, n_iters, batch_size, id=0):
        self.n_iters = n_iters
        self.batch_size = batch_size
        self.perm = None
        self.cur = 0
        # fix the random seed for reproducibility
        np.random.seed(cfg.RNG_SEED + id)

    def shuffle_inds(self):
        """Randomly permute the training roidb."""
        self.perm = np.random.permutation(np.arange(self.n_iters))
        self.cur = 0

    def get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        if self.cur + self.batch_size > self.n_iters:
            self.shuffle_inds()

        inds = self.perm[self.cur:self.cur + self.batch_size]
        self.cur += self.batch_size
        return inds


class PrefetchProcess(Process):
    """Experimental class for prefetching batches in a separate process."""

    def __init__(self, queue, fetch, id=None):
        super(PrefetchProcess, self).__init__()
        self.queue = queue
        self.fetch = fetch
        self.id = id

    def run(self):
        if self.id is None:
            print('PrefetchProcess started.')
        else:
            print('PrefetchProcess {:d} started.'.format(self.id))
        while True:
            batch_data = self.fetch()
            self.queue.put(batch_data, block=True)


class DataFetcher(object):
    """RoI data fetcher used for training."""

    def __init__(self, db, debug=False):
        if cfg.TRAIN.USE_PREFETCH:
            self.data_queue = Queue(QUEUE_MAXSIZE)
            self.prefetch_procs = []
            for i in xrange(N_PROCS):
                mgr = IndexManager(len(db), cfg.TRAIN.BATCH_SIZE, i)
                mgr.shuffle_inds()

                def fetch():
                    return _fetch(mgr, db, debug)

                proc = PrefetchProcess(self.data_queue, fetch, i)
                proc.start()
                self.prefetch_procs.append(proc)

            # Terminate the child process when the parent exists.
            def cleanup():
                for i, proc in enumerate(self.prefetch_procs):
                    print('Terminating PrefetchProcess {:d}...'.format(i))
                    proc.terminate()
                    proc.join()

            import atexit

            atexit.register(cleanup)

        else:
            mgr = IndexManager(len(db), cfg.TRAIN.BATCH_SIZE)
            mgr.shuffle_inds()
            self.fetch = lambda: _fetch(mgr, db, debug)

    def get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self.data_queue.
        """
        if cfg.TRAIN.USE_PREFETCH:
            batch_data = self.data_queue.get()
        else:
            batch_data = self.fetch()
        return batch_data

    def terminate(self):
        """ """
        if cfg.TRAIN.USE_PREFETCH:
            for i, proc in enumerate(self.prefetch_procs):
                print('Terminating PrefetchProcess {:d}...'.format(i))
                proc.terminate()
                proc.join()
