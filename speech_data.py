import os, sys
import logging
import traceback
import random
import time
import threading
import numpy as np
import tensorflow as tf
import config as cfg
try:
    from Queue import Queue
except ImportError:
    from queue import Queue
from utils.signalprocess import *
from utils.tools import *
from tensorflow.python.client import device_lib

# tf.logging.set_verbosity(tf.logging.DEBUG)
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = "0"

class Producer(threading.Thread):
    def __init__(self, reader):
        threading.Thread.__init__(self)
        self.reader = reader
        self.exitcode = 0
        self.stop_flag = False

    def run(self):
        try:
            min_queue_size = self.reader.config.min_queue_size
            while not self.stop_flag:
                idx = self.reader._next_load_idx
                if idx >= len(self.reader.data_list):
                    self.reader._batch_queue.put([])
                    break
                if self.reader._batch_queue.qsize() < min_queue_size:
                    batch_list = self.reader.load_samples()
                    for batch in batch_list:
                        self.reader._batch_queue.put(batch)
                else:
                    time.sleep(1)
        except Exception as e:
            logging.warning("producer exception: %s" % e)
            self.exitcode = 1
            traceback.print_exc()

    def stop(self):
        self.stop_flag = True

class SpeechReader(object):
    def __init__(self, config, data_dir, batch_size=None, max_sent_len=-1,
                 min_sent_len=10, num_gpu=1, job_type="train"):
        self.num_gpu = num_gpu
        self.max_sent_len = max_sent_len
        self.min_sent_len = min_sent_len
        self.batch_size = batch_size
        if self.batch_size is None:
            self.batch_size = config.batch_size
        self.config = config
        self.data_dir = data_dir
        self.job_type = job_type
        self.data_list = self.read_data_list()
        self._batch_queue = Queue()
        self.reset()

    def read_data_list(self):
        file_path = os.path.join(self.data_dir, self.config.target_lst_file)
        with open(file_path, "r") as f:
            data_list = [line.strip().split()[0] for line in f.readlines()]
        #data_list = data_list[::20]
        return data_list

    def shuffle_data_list(self):
        random.shuffle(self.data_list)

    def reset(self):
        self.sample_buffer = []
        self._next_load_idx = 0
        if self.job_type == "train":
            self.shuffle_data_list()
        self._producer = Producer(self)
        self._producer.start()

    def load_one_mixture(self, file_name):
        target_channel = self.config.target_channel
        target_dir = self.config.target_dir
        refer_channels = self.config.refer_channels
        num_spkrs = self.config.num_spkrs
        num_refers = len(refer_channels)
        # must have refer channels to get IPD information and DC on them
        assert num_refers > 0
        sample_list = []
        samp_rate = self.config.samp_rate
        frame_size = self.config.frame_size
        shift = self.config.shift
        # read from target channel
        mixture_file = os.path.join(self.data_dir, "mix", file_name)
        total_sig = read_wav(mixture_file, samp_rate=samp_rate)
        mixture_sig =total_sig[0]
        mixture_stft = stft(mixture_sig, size=frame_size, shift=shift, fading=True, ceil=True)
        mixture_magn, mixture_phase = np.abs(mixture_stft), np.angle(mixture_stft)
        threshold = 10 **(self.config.threshold/20.0)*np.max(mixture_magn)
        silence_mask = (mixture_magn > threshold)
        seq_len, feat_len = mixture_magn.shape 
        # to mask the silence TF-bin
        # read refer channels to get spatial feats of IPD for input in a concat style
        spatial_feats = []
        for ch in refer_channels:
            #file_path = os.path.join(self.data_dir, "ch%d"%ch, file_name)
            #refer_sig = read_wav(file_path, samp_rate=samp_rate)
            refer_sig = total_sig[ch]
            refer_stft = stft(refer_sig, size=frame_size, shift=shift, fading=True, ceil=True)
            refer_phase = np.angle(refer_stft)
            cos_diff = np.cos(mixture_phase - refer_phase)
            sin_diff = np.sin(mixture_phase - refer_phase)
            spatial_feats.append(np.concatenate([cos_diff, sin_diff], axis=1))
        spatial_feats = np.concatenate(spatial_feats, axis=1)
        # read source target and get the classification label
        if self.config.job_anechoic:
            if self.job_type == 'train':
                data_dir = self.config.train_anechoic_dir
            else:
                data_dir = self.config.dev_anechoic_dir
        else:
            data_dir = self.data_dir
        source_target = []
        for spkr_i in range(num_spkrs):
            source_path = os.path.join(data_dir,  "s%d"%(spkr_i + 1), file_name)
            source_sig = read_wav(source_path, samp_rate=samp_rate)[0]
            source_stft = stft(source_sig, size=frame_size, shift=shift, fading=True, ceil=True)
            source_magn, source_phase = np.abs(source_stft), np.angle(source_stft)
            if self.config.phase_sensitive:
                source_target.append(source_magn * np.cos(source_phase - mixture_phase))
            else:
                source_target.append(source_magn)
        # shape is [T, F, N]
        source_target = np.stack(source_target, axis=2)
        argmax_idx = np.reshape(np.argmax(source_target, axis = 2), -1)
        seq_len, feat_len = mixture_magn.shape
        source_label = np.eye(num_spkrs)[argmax_idx]
        source_label = np.reshape(source_label, [seq_len, feat_len, num_spkrs])
        i = 0
        # shape is [T, F]
        while self.max_sent_len > 0 and i + self.max_sent_len <= seq_len:
            one_sample = (spatial_feats[i:i+self.max_sent_len],
                          mixture_magn[i:i+self.max_sent_len],
                          source_target[i:i+self.max_sent_len], self.max_sent_len, silence_mask[i:i+self.max_sent_len], source_label[i:i+self.max_sent_len])
            sample_list.append(one_sample)
            i += (1 - self.config.overlap_rate) * self.max_sent_len
        if seq_len - i >= self.min_sent_len and self.job_type != "train":
            one_sample = (spatial_feats[i:], mixture_magn[i:], source_target[i:], seq_len - i, silence_mask[i:], source_label[i:])
            sample_list.append(one_sample)
        return sample_list

    def patch_batch_data(self):
        batch_size = self.batch_size
        group_size = batch_size * self.num_gpu
        feat_dim = self.config.feat_dim
        num_groups = len(self.sample_buffer) // group_size
        if num_groups == 0:
            return []
        group_list = []
        num_spkrs = self.config.num_spkrs
        num_refers = len(self.config.refer_channels)
        choose_samples = [self.sample_buffer[i:i+group_size]
                          for i in range(0, group_size * num_groups, group_size)]
        self.sample_buffer = self.sample_buffer[group_size * num_groups:]
        for one_group in choose_samples:
            group_IPD = []
            group_MAGN = []
            group_source = []
            group_seq_len = []
            group_silence_mask = []
            group_source_label = []
            for i in range(0, group_size, batch_size):
                one_batch = one_group[i:i+batch_size]
                max_len = int(max(map(lambda x: x[3], one_batch)))
                batch_IPD = np.zeros((batch_size, max_len, feat_dim*2*num_refers), dtype=np.float32)
                batch_MAGN = np.zeros((batch_size, max_len, feat_dim), dtype=np.float32)
                batch_source = np.zeros((batch_size, max_len, feat_dim, num_spkrs), dtype=np.float32)
                batch_seq_len = np.zeros(batch_size, dtype=np.int32)
                batch_silence_mask = np.zeros((batch_size, max_len, feat_dim), dtype = np.bool)
                batch_source_label = np.zeros((batch_size, max_len, feat_dim, num_spkrs), dtype = np.float32)

                for j in range(batch_size):
                    this_len = one_batch[j][3]
                    batch_seq_len[j] = this_len
                    batch_IPD[j, 0:this_len, :] = one_batch[j][0]
                    batch_MAGN[j, 0:this_len, :] = one_batch[j][1]
                    batch_source[j, 0:this_len, :, :] = one_batch[j][2]
                    batch_silence_mask[j,0:this_len,:] = one_batch[j][4]
                    batch_source_label[j, 0:this_len,:] = one_batch[j][5]

                group_silence_mask.append(batch_silence_mask)
                group_IPD.append(batch_IPD)
                group_source.append(batch_source)
                group_MAGN.append(batch_MAGN)
                group_seq_len.append(batch_seq_len)
                group_source_label.append(batch_source_label)
            group_list.append((group_IPD, group_MAGN, group_source, group_seq_len,group_silence_mask,group_source_label))
        return group_list

    def load_samples(self):
        load_file_num = self.config.load_file_num
        idx = self._next_load_idx
        for file_name in self.data_list[idx: idx + load_file_num]:
            self.sample_buffer.extend(self.load_one_mixture(file_name))
        self._next_load_idx += load_file_num
        if self.job_type == "train":
            random.shuffle(self.sample_buffer)
        group_list = self.patch_batch_data()
        return group_list

    def next_batch(self):
        while self._producer.exitcode == 0:
            try:
                batch_data = self._batch_queue.get(block=False)
                if len(batch_data) == 0:
                    return None
                else:
                    return batch_data
            except Exception as e:
                time.sleep(3)

def test():
    data_dir = cfg.train_dir
    # set_log("tmp")
    # gpu_device_name = tf.test.gpu_device_name()
    # logging.info(gpu_device_name)
    # print(gpu_device_name)
    # logging.info(tf.test.is_gpu_available())
    # print(tf.test.is_gpu_available())
    # local_device_protos = device_lib.list_local_devices()
    # for x in local_device_protos:
    #     if x.device_type == "GPU":
    #         print(x)
    #         logging.info(x)
    start_time = time.time()
    reader = SpeechReader(cfg, data_dir, batch_size=32, max_sent_len=400, min_sent_len=10,
                          num_gpu=2, job_type="test")
    batch_data = reader.next_batch()
    spatial_feat, mix_feat, src_feat, seq_len,silence_mask ,source_label = batch_data
    print(source_label[0])
    print(source_label[0].shape)
    
    #print(spatial_feat)
    #logging.info("spatial_feat.shape: {}, {}".format(mix_feat[0].shape, mix_feat[0].dtype))
    #logging.info("silence_mask.shape: {}, {}".format(spatial_feat[0].shape, spatial_feat[0].dtype))
    #logging.info("src_label.shape: {}, {}".format(src_feat[0].shape, src_feat[0].dtype))
    #logging.info("seq_len.shape: {}, {}".format(seq_len[0].shape, seq_len[0].dtype))
    #for i in range(99):
    #    batch_data = reader.next_batch()
    #duration = time.time() - start_time
    #logging.info("read 100 batches consume {:.2f} seconds".format(duration))
    reader._producer.stop()

if __name__ == "__main__":
    test()
