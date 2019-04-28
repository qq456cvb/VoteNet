from tensorpack import *
import tensorpack.utils
from model import Model

BATCH_SIZE = 8


if __name__ == '__main__':
    if __name__ == '__main__':
        tensorpack.utils.logger.auto_set_dir()

        train_set = MyDataset('data/modelnet40_ply_hdf5_2048/train_files.txt', transform=train_transform,
                              target_transform=None)
        test_set = MyDataset('data/modelnet40_ply_hdf5_2048/test_files.txt', transform=test_transform,
                             target_transform=None)

        # dataset = BatchData(PrefetchData(train_set, 4, 4), BATCH_SIZE)

        lr_schedule = [(80, 1e-4), (1e-5, 120)]
        # lr_schedule = [(i, 5e-5) for i in range(260)]
        # get the config which contains everything necessary in a training
        config = AutoResumeTrainConfig(
            always_resume=False,
            model=Model(),
            # The input source for training. FeedInput is slow, this is just for demo purpose.
            # In practice it's best to use QueueInput or others. See tutorials for details.
            data=QueueInput(BatchData(PrefetchData(train_set, 10, 10), BATCH_SIZE)),
            # starting_epoch=60,
            callbacks=[
                ModelSaver(),  # save the model after every epoch
                ScheduledHyperParamSetter('learning_rate', lr_schedule),
                InferenceRunner(BatchData(PrefetchData(test_set, 10, 10), BATCH_SIZE),
                                [ScalarStats(['accuracy', 'total_cost'])]),
                # MaxSaver('val_accuracy'),  # save the model with highest accuracy
            ],
            # steps_per_epoch=100,
            max_epoch=260,
        )
        launch_train_with_config(config, AsyncMultiGPUTrainer(gpus=[0, 1]))

