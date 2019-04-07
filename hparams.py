

class HParams:

    def __init__(self):
        self.l_size = 25
        self.learning_rate = 0.1
        self.decay_rate = 0.6
        self.decay_steps = 20
        self.steps = 200
        self.batch_size = 16
        self.class_nums = 16
        self.tfrecord_dir = '/data/data/crnn_tfrecords'
        self.key_fpath = '/data/CRNN_Name_v2.0/key/keys.txt'

