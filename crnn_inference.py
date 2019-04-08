import os
import numpy as np
import tensorflow as tf
from crnn_model import crnn_model
from global_configuration import config
import define_input_fn
import hparams
from data_prepare import char_dict, load_tf_data
import cv2

# from log_utils import log_util
#
# logger = log_util.init_logger()
import logging

logging.basicConfig(level=logging.DEBUG)


def crnn_net(is_training, feature, label, batch_size, l_size):
    seq_len = l_size
    if is_training:
        shadownet = crnn_model.ShadowNet(phase='Train', hidden_nums=256, layers_nums=2, seq_length=seq_len,
                                         num_classes=config.cfg.TRAIN.CLASSES_NUMS, rnn_cell_type='lstm')

        imgs = tf.image.resize_images(feature, (32, l_size * 4), method=0)
        input_imgs = tf.cast(x=imgs, dtype=tf.float32)

        with tf.variable_scope('shadow', reuse=False):
            net_out, tensor_dict = shadownet.build_shadownet(inputdata=input_imgs)

        cost = tf.reduce_mean(tf.nn.ctc_loss(labels=label, inputs=net_out,
                                             sequence_length=seq_len * np.ones(batch_size)))

        # lstm l2
        lstm_tv = tf.trainable_variables(scope='LSTMLayers')
        r_lambda = 0.001
        regularization_cost = r_lambda * tf.reduce_sum([tf.nn.l2_loss(v) for v in lstm_tv])
        cost = cost + regularization_cost

        model_params = tf.trainable_variables()
        tower_grad = tf.gradients(cost, model_params)

        return cost, zip(tower_grad, model_params), net_out, tensor_dict, seq_len
    else:
        shadownet = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=seq_len,
                                         num_classes=config.cfg.TRAIN.CLASSES_NUMS, rnn_cell_type='lstm')

        imgs = tf.image.resize_images(feature, (32, l_size * 4), method=0)
        input_imgs = tf.cast(x=imgs, dtype=tf.float32)

        with tf.variable_scope('shadow', reuse=False):
            net_out, tensor_dict = shadownet.build_shadownet(inputdata=input_imgs)

        cost = None

        model_params = None
        tower_grad = None

        return cost, None, net_out, tensor_dict, seq_len


def my_model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL)
    loss, gradvars, preds, tensor_dict, seq_len = crnn_net(
        is_training, features, labels, params.batch_size, params.l_size)

    if (mode == tf.estimator.ModeKeys.TRAIN):
        global_step = tf.train.get_global_step()
        # starter_learning_rate = params.learning_rate
        # learning_rate = tf.train.exponential_decay(starter_learning_rate,
        #                                            global_step,
        #                                            params.decay_steps,
        #                                            params.decay_rate)
        # TODO: optimizer
        optimizer = tf.train.AdadeltaOptimizer().minimize(loss, global_step=global_step)
        # log define
        tensors_to_log = {'global_step': global_step, 'loss': loss}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)
        train_hooks = [logging_hook]
        predictions = None

    elif mode == tf.estimator.ModeKeys.EVAL:
        predictions = None
        optimizer = None
        train_hooks = None
    elif mode == tf.estimator.ModeKeys.PREDICT:
        # bs = features.get_shape
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(preds,
                                                          seq_len * np.ones(1),  # TODO
                                                          merge_repeated=False)
        # decoded, log_prob = tf.nn.ctc_greedy_decoder(preds,
        #                                              seq_len * np.ones(1),  # TODO
        #                                              merge_repeated=False)
        # predictions = tf.sparse_to_dense(tf.to_int32(decoded[0].indices),
        #                                  tf.to_int32(decoded[0].dense_shape),
        #                                  tf.to_int32(decoded[0].values),
        #                                  name="output")
        predictions = tf.sparse.to_dense(sp_input=decoded[0], name='output')
        loss = None
        optimizer = None
        train_hooks = None
    else:
        predictions = None
        loss = None
        optimizer = None
        train_hooks = None

    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=predictions,
                                      loss=loss,
                                      train_op=optimizer,
                                      training_hooks=train_hooks
                                      )


def main():
    # Session configuration.
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        intra_op_parallelism_threads=0,
        gpu_options=tf.GPUOptions(force_gpu_compatible=False))
    # sess_config.gpu_options.per_process_gpu_memory_fraction=0.4
    run_config = tf.estimator.RunConfig(session_config=sess_config,
                                        save_checkpoints_steps=100,
                                        keep_checkpoint_max=3,
                                        model_dir='./run_output')

    # model_fn = my_model_fn(num_gpus=0)
    _hparams = hparams.HParams()

    # variable_strategy = 'CPU'
    # num_gpus = 0
    estimator = tf.estimator.Estimator(
        model_fn=my_model_fn,
        config=run_config,
        params=_hparams, )

    data_dir = _hparams.tfrecord_dir
    BATCH_SIZE = _hparams.batch_size  # 16
    # EPOCHS = 5
    STEPS = _hparams.steps  # 2000

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: define_input_fn.my_input_fn(data_dir=data_dir,
                                                                                    subset='train',
                                                                                    batch_size=BATCH_SIZE),
                                        max_steps=STEPS)

    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: define_input_fn.my_input_fn(data_dir=data_dir,
                                                                                  subset='val',
                                                                                  batch_size=BATCH_SIZE),
                                      steps=1,
                                      start_delay_secs=1)

    tf.estimator.train_and_evaluate(estimator,
                                    train_spec,
                                    eval_spec)

    print('ckp_path: ', estimator.latest_checkpoint())

    test_img_dir = './real_data/name_crop_result'
    img_fn_list = os.listdir(test_img_dir)
    for img_fn in img_fn_list:
        test_img_path = os.path.join(test_img_dir, img_fn)
        # test_img_path = './real_data/name_crop_result/1_55998336_IDFront_name_0.jpg'
        print(test_img_path)
        img_cv2 = cv2.imread(test_img_path)
        # predictions = estimator.predict(input_fn=lambda: define_input_fn.input_fn_for_inference(img_cv2),
        #                                 yield_single_examples=True)

        test_images = [cv2.resize(img_cv2, (100, 32))]
        test_images = np.asarray(test_images)
        test_labels = np.asarray([''])
        test_input_fn = tf.estimator.inputs.numpy_input_fn(x=test_images,
                                                           y=test_labels,
                                                           batch_size=1,
                                                           num_epochs=1,
                                                           shuffle=False,
                                                           queue_capacity=1,
                                                           num_threads=1)
        predictions = estimator.predict(input_fn=test_input_fn,
                                        yield_single_examples=True)

        # pred_result = load_tf_data.sparse_tensor_to_str(next(predictions))
        pred_res_num = next(predictions)
        print('pred_res_num: ', pred_res_num)
        int_to_char = load_tf_data.char_dict.int_to_char
        pred_res_str = ''.join([int_to_char[int] for int in pred_res_num])
        print('prediction: ', pred_res_str)


if __name__ == '__main__':
    main()
