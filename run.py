from tensorpack import *
from tensorpack.utils import logger
from model import Model
from dataset import MyDataFlow, sunrgbd_object, type2class
import multiprocessing
from evaluator import Evaluator, eval_det
import six
import numpy as np
from utils import get_gt_cls

BATCH_SIZE = 2


def pad_along_axis(array, target_length, axis=0):
    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return array

    npad = [(0, 0) for _ in range(axis_nb)]
    npad[axis] = (0, pad_size)
    b = np.pad(array, pad_width=npad, mode='edge')
    return b


class BatchData2Biggest(BatchData):
    def __iter__(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """
        holder = []
        for data in self.ds:
            holder.append(data)
            if len(holder) == self.batch_size:
                yield BatchData2Biggest._aggregate_batch(holder, self.use_list)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield BatchData2Biggest._aggregate_batch(holder, self.use_list)

    @staticmethod
    def _batch_numpy(data_list):
        data = data_list[0]
        if isinstance(data, six.integer_types):
            dtype = 'int32'
        elif type(data) == bool:
            dtype = 'bool'
        elif type(data) == float:
            dtype = 'float32'
        elif isinstance(data, (six.binary_type, six.text_type)):
            dtype = 'str'
        else:
            try:
                dtype = data.dtype
            except AttributeError:
                raise TypeError("Unsupported type to batch: {}".format(type(data)))
        try:
            return np.asarray(data_list, dtype=dtype)
        except Exception:  # noqa)
            try:
                largest_dim = max([d.shape[0] for d in data_list])
                data_list = [pad_along_axis(d, largest_dim) for d in data_list]
                return np.asarray(data_list, dtype=dtype)
            except Exception:
                try:
                    # open an ipython shell if possible
                    import IPython as IP
                    IP.embed()  # noqa
                except ImportError:
                    pass

    @staticmethod
    def _aggregate_batch(data_holder, use_list=False):
        first_dp = data_holder[0]
        if isinstance(first_dp, (list, tuple)):
            result = []
            for k in range(len(first_dp)):
                data_list = [x[k] for x in data_holder]
                if use_list:
                    result.append(data_list)
                else:
                    result.append(BatchData2Biggest._batch_numpy(data_list))
        elif isinstance(first_dp, dict):
            result = {}
            for key in first_dp.keys():
                data_list = [x[key] for x in data_holder]
                if use_list:
                    result[key] = data_list
                else:
                    result[key] = BatchData2Biggest._batch_numpy(data_list)
        return result


if __name__ == '__main__':
    logger.auto_set_dir()

    # this is the official train/val split
    train_set = MyDataFlow('/data/mysunrgbd', 'training', training=True, idx_list=list(range(5051, 10336)), cache_dir='cache_train')
    # TestDataSpeed(train_set).start()

    gt_cls = {}
    gt_all = {}
    for classname in type2class:
        gt_cls[classname] = get_gt_cls('/home/neil/frustum-pointnets/sunrgbd/sunrgbd_detection/gt_boxes', classname)
        for img_id in gt_cls[classname]:
            if img_id not in gt_all:
                gt_all[img_id] = []
            for box in gt_cls[classname][img_id]:
                gt_all[img_id].append((classname, box))
                    # print(classname, box)

    lr_schedule = [(80, 1e-4), (120, 1e-5)]
    # lr_schedule = [(i, 5e-5) for i in range(260)]
    # get the config which contains everything necessary in a training
    config = AutoResumeTrainConfig(
        always_resume=False,
        model=Model(),
        # The input source for training. FeedInput is slow, this is just for demo purpose.
        # In practice it's best to use QueueInput or others. See tutorials for details.
        data=QueueInput(BatchData2Biggest(
            PrefetchDataZMQ(train_set, multiprocessing.cpu_count() // 2, multiprocessing.cpu_count() // 2), BATCH_SIZE)),
        # starting_epoch=60,
        callbacks=[
            ModelSaver(),  # save the model after every epoch
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            SimpleMovingAverage(['obj_accuracy', 'sem_accuracy', 'total_cost'], 100),  # change the frequency for loss
            # compute mAP on val set
            PeriodicTrigger(Evaluator(MyDataFlow('/data/mysunrgbd', 'training', training=False, idx_list=list(range(1, 5051)), cache_dir='cache_val'), gt_all),
                            every_k_epochs=5),
            # MaxSaver('val_accuracy'),  # save the model with highest accuracy
        ],
        monitors=train.DEFAULT_MONITORS() + [ScalarPrinter(enable_step=True, enable_epoch=False)],
        max_epoch=260,
    )
    launch_train_with_config(config, SimpleTrainer())
