import os
import setproctitle
import time
import cv2
import goturn_net
import numpy as np
import tensorflow as tf
import uuid
from helper.config import POLICY
from helper.helper import show_images
from logger.logger import setup_logger
from loader.loader_vid import loader_vid
from example_generator import example_generator, check_center

setproctitle.setproctitle('TRAIN_TRACKER_IMAGENET_VID')
logger = setup_logger(logfile=None)

NUM_EPOCHS = POLICY['NUM_EPOCHS']
BATCH_SIZE = POLICY['BATCH_SIZE']
WIDTH = POLICY['WIDTH']
HEIGHT = POLICY['HEIGHT']
pretraind_model = POLICY['pretrained_model']
kGeneratedExamplesPerImage = POLICY['kGeneratedExamplesPerImage']


run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth = True


def train_video(videos):
    """TODO: Docstring for train_video.
    """
    video_num = np.random.randint(0, len(videos))
    video = videos[video_num]

    annotations = video.annotations

    if len(annotations) < 2:
        logger.info('Error - video {} has only {} annotations', video.video_path, len(annotations))

        while len(annotations) < 2:
            video_num = np.random.randint(0, len(videos))
            video = videos[video_num]
            annotations = video.annotations

    ann_index = np.random.randint(0, len(annotations) - 1)
    frame_num_prev, image_prev, bbox_prev = video.load_annotation(ann_index)

    frame_num_curr, image_curr, bbox_curr = video.load_annotation(ann_index + 1)

    return image_prev, image_curr, bbox_prev, bbox_curr
    # tracker_trainer.train(image_prev, image_curr, bbox_prev, bbox_curr)


def data_reader(train_vid_videos):
    objExampleGen = example_generator(float(POLICY['lamda_shift']), float(POLICY['lamda_scale']),
                                      float(POLICY['min_scale']), float(POLICY['max_scale']), logger)

    images = []
    targets = []
    bbox_gt_scaleds = []

    for idx in xrange(POLICY['BATCH_SIZE']):
        center_bool = False
        while not center_bool:
            img_prev, img_curr, bbox_prev, bbox_curr = train_video(train_vid_videos)
            objExampleGen.reset(bbox_prev, bbox_curr, img_prev, img_curr)
            image, target, bbox_gt_scaled = objExampleGen.make_true_example()
            center_bool = check_center(bbox_gt_scaled)
            if center_bool:
                images.append(image)
                targets.append(target)
                bbox_gt_scaleds.append(bbox_gt_scaled)
                images, targets, bbox_gt_scaleds = objExampleGen.make_training_examples(kGeneratedExamplesPerImage,
                                                                                        images, targets, bbox_gt_scaleds)

    # debug
    # show_images(images, targets, bbox_gt_scaleds)

    for idx, (img, tag, box) in enumerate(zip(images, targets, bbox_gt_scaleds)):
        images[idx] = cv2.resize(img, (HEIGHT, WIDTH), interpolation=cv2.INTER_CUBIC)
        targets[idx] = cv2.resize(tag, (HEIGHT, WIDTH), interpolation=cv2.INTER_CUBIC)
        bbox_gt_scaleds[idx] = np.array([box.x1, box.y1, box.x2, box.y2], dtype=np.float32)

    images = np.reshape(np.array(images), (len(images), 227, 227, 3))
    targets = np.reshape(np.array(targets), (len(targets), 227, 227, 3))
    bbox_gt_scaled = np.reshape(np.array(bbox_gt_scaleds), (len(bbox_gt_scaleds), 4))

    return [images, targets, bbox_gt_scaled]
if __name__ == "__main__":
    # Load vid training images and annotations
    vid_folder = os.path.join(POLICY['vid2015'], 'images')
    vid_annotations_folder = os.path.join(POLICY['vid2015'], 'gt')
    objLoaderVID = loader_vid(vid_folder, vid_annotations_folder, logger)
    objLoaderVID.loaderVID()
    train_vid_videos = objLoaderVID.get_videos()

    vid_images = 0
    for vid_idx in xrange(len(train_vid_videos)):
        video = train_vid_videos[vid_idx]
        annos = video.annotations
        vid_images += len(annos)
    total_image_size = vid_images
    logger.info('total training VID images size is: ' + str(vid_images))

    # debug
    # cur_batch = data_reader(train_vid_videos)

    # network initialization
    tracknet = goturn_net.TRACKNET(BATCH_SIZE)
    tracknet.build()

    # learning policy
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.piecewise_constant(global_step,
                                                [tf.cast(v, tf.int32) for v in POLICY['step_values']],
                                                POLICY['learning_rates'])
    train_step = tf.train.AdamOptimizer(learning_rate,
                                       POLICY['momentum'],
                                       POLICY['momentum2']).minimize(tracknet.loss_wdecay, global_step=global_step)

    # summary
    merged_summary = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter('./train_summary', sess.graph)

    # variable initialization
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    sess.run(init)
    sess.run(init_local)

    # checkpoint
    ckpt_dir = "./checkpoints"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    start = 0

    if ckpt and ckpt.model_checkpoint_path:
        start = int(ckpt.model_checkpoint_path.split("-")[1])
        logger.info("start by iteration: %d" % (start))
        saver = tf.train.Saver()
        saver.restore(sess, ckpt.model_checkpoint_path)
        logger.info("model is restored using " + str(ckpt))
    elif pretraind_model:
        restore = {}
        from tensorflow.contrib.framework.python.framework.checkpoint_utils import list_variables
        slim = tf.contrib.slim
        for scope in list_variables(pretraind_model):
            if 'conv' in scope[0]:
                variables_to_restore = slim.get_variables(scope=scope[0])
                if variables_to_restore:
                    restore[scope[0]] = variables_to_restore[0]                                # variables_to_restore is list : [op]
        saver = tf.train.Saver(restore)
        saver.restore(sess, pretraind_model)
        logger.info("model is restored conv only using " + str(pretraind_model))

    assign_op = global_step.assign(start)
    sess.run(assign_op)
    model_saver = tf.train.Saver(max_to_keep=30)

    # train
    try:
        for i in range(start, int((total_image_size) / BATCH_SIZE * NUM_EPOCHS)):
            if i % int((total_image_size) / BATCH_SIZE) == 0:
                logger.info("start epoch[%d]" % (int(float(i) / (total_image_size) * BATCH_SIZE)))
                if i > start:
                    save_ckpt = "checkpoint.ckpt"
                    last_save_itr = i
                    model_saver.save(sess, "checkpoints/" + save_ckpt, global_step=i + 1)

            start_time = time.time()
            # dataloader test
            cur_batch = data_reader(train_vid_videos)
            logger.debug('data_reader: time elapsed: %.3f' % (time.time() - start_time))

            start_time = time.time()

            feed_val, error_box_index = tracknet._batch(cur_batch[2], POLICY)

            # debug
            if not len(error_box_index) == 0:
                for error_index in error_box_index:
                    randomString = uuid.uuid4()
                    cv2.imwrite('./error_generatedData/step_{:>3}_search_'.format(start) + str(randomString) + '.jpg', cur_batch[0][error_index])
                    cv2.imwrite('./error_generatedData/step_{:>3}_target_'.format(start) + str(randomString) + '.jpg', cur_batch[1][error_index])

            [_, loss] = sess.run([train_step, tracknet.loss_wdecay], feed_dict={tracknet.image: cur_batch[0],
                                                                               tracknet.target: cur_batch[1],
                                                                               tracknet.bbox: cur_batch[2],
                                                                               tracknet.confs: feed_val['confs'],
                                                                               tracknet.coord: feed_val['coord'],
                                                                               tracknet.upleft: feed_val['upleft'],
                                                                               tracknet.botright: feed_val['botright'],
                                                                               tracknet.areas: feed_val['areas']})
            logger.debug('Train: time elapsed: %.3fs, average_loss: %f' % (time.time() - start_time, loss))

            if i % 10 == 0 and i > start:
                summary = sess.run(merged_summary, feed_dict={tracknet.image: cur_batch[0],
                                                              tracknet.target: cur_batch[1],
                                                              tracknet.bbox: cur_batch[2],
                                                              tracknet.confs: feed_val['confs'],
                                                              tracknet.coord: feed_val['coord'],
                                                              tracknet.upleft: feed_val['upleft'],
                                                              tracknet.botright: feed_val['botright'],
                                                              tracknet.areas: feed_val['areas']})
                train_writer.add_summary(summary, i)
    except KeyboardInterrupt:
        print("get keyboard interrupt")
        if (i - start > 1000):
            model_saver = tf.train.Saver()
            save_ckpt = "checkpoint.ckpt"
            model_saver.save(sess, "checkpoints/" + save_ckpt, global_step=i + 1)