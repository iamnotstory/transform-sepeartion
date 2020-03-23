import os, sys
import logging
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import traceback
import shutil
import numpy as np
import tensorflow as tf
import config as cfg
from utils.tools import *
from speech_data import SpeechReader
from model.transformer import DF_Model

tf.logging.set_verbosity(tf.logging.ERROR)

if __name__ == "__main__":
    gpu_list = cfg.gpu_list
    num_gpu = len(gpu_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
    set_log(cfg.job_dir)
    shutil.copy('config.py', cfg.job_dir)
    tf.set_random_seed(cfg.seed)
    train_dir = cfg.train_dir
    dev_dir = cfg.dev_dir
    # for pretraining, first set max_sent_len to shorter_sent_len
    if cfg.pretrain_shorter_epoch > 0:
        init_max_len = cfg.shorter_sent_len
    else:
        init_max_len = cfg.longer_sent_len
    train_reader = SpeechReader(cfg, train_dir, max_sent_len=init_max_len,
                                min_sent_len=cfg.min_sent_len, num_gpu=num_gpu, job_type="train")
    dev_reader = SpeechReader(cfg, dev_dir, batch_size=cfg.dev_batch_size, max_sent_len=init_max_len,
                              num_gpu=num_gpu, min_sent_len=cfg.min_sent_len, job_type="train")
    try:
        with tf.Graph().as_default():
            loss_checker = checker(cfg)
            sess_config = tf.ConfigProto()
            sess_config.allow_soft_placement = True
            sess_config.gpu_options.allow_growth = True
            sess = tf.Session(config=sess_config)
            initializer = tf.random_normal_initializer(mean=cfg.init_mean, stddev=cfg.init_stddev)
            with tf.variable_scope("SpeechSeparation", initializer=None):
                model = DF_Model(sess, cfg, num_gpu, initializer)
                sess.run(tf.global_variables_initializer())
                #model.preload(sess)
                print('heer:')
                if cfg.resume:
                    model.restore_model()
                for i_epoch in range(cfg.max_epoch):
                    logging.info("Start Epoch {}/{}".format(i_epoch + 1, cfg.max_epoch))
                    logging.info("learning_rate: {}".format(loss_checker.learning_rate))
                    while not loss_checker.should_stop():
                        batch_data = train_reader.next_batch()
                        if batch_data is None:
                            model.reset()
                            if i_epoch == cfg.pretrain_shorter_epoch - 1:
                                logging.info("Pretraining Stage Finished")
                                train_reader.max_sent_len = cfg.longer_sent_len
                            train_reader.reset()
                            logging.info("Epoch {} finished!!!".format(i_epoch + 1))
                            break
                        else:
                            #loss_checker.learning_rate = 0.001
                            i_global = model.run_batch(batch_data, loss_checker.learning_rate)
                            if i_global % cfg.dev_period == 0 and i_epoch >= 0:
                                avg_loss = model.valid(dev_reader)
                                loss_improved, best_loss = loss_checker.update(sess, avg_loss)
                                if loss_improved:
                                    logging.info("New best loss {}".format(best_loss))
                                    save_path = os.path.join(model.best_loss_dir, "model.ckpt")
                                    model.best_loss_saver.save(sess, save_path)
                                # avoid early stopping in stage of pretraining
                                if i_epoch < cfg.pretrain_shorter_epoch:
                                    loss_checker.reset_step()
                                dev_reader.reset()
                    if loss_checker.should_stop():
                        logging.info("Early stopped")
                        break
    except Exception as e:
        logging.error("training exception: %s" % e)
        traceback.print_exc()
    finally:
        train_reader._producer.stop()
        dev_reader._producer.stop()
