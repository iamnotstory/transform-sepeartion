import os
import sys
import logging
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import traceback
import numpy as np
import config as cfg
import tensorflow as tf
from utils.tools import *
from model.transformer import DF_Model
from mir_eval import bss_eval_sources
from sklearn.cluster import KMeans

tf.logging.set_verbosity(tf.logging.ERROR)

def prepare_feat(data_dir, file_name):
    target_channel = cfg.target_channel
    target_dir = cfg.target_dir
    refer_channels = cfg.refer_channels
    num_spkrs = cfg.num_spkrs
    num_refers = len(refer_channels)
    assert num_refers > 0
    # read from target channel
    mixture_file = os.path.join(data_dir, "mix", file_name)
    total_sig = read_wav(mixture_file, samp_rate=cfg.samp_rate)
    mixture_sig = total_sig[0]
    mixture_stft = stft(mixture_sig, cfg.frame_size, cfg.shift, fading=True, ceil=True)
    mixture_magn, mixture_phase = np.abs(mixture_stft), np.angle(mixture_stft)
    threshold = 10 ** (cfg.threshold/20.0) * np.max(mixture_magn)
    silence_mask = (mixture_magn > threshold)
    # read from refer channels
    spatial_feats = []
    for ch in refer_channels:
        #refer_file = os.path.join(data_dir, "ch%d"%ch, file_name)
        #refer_sig = read_wav(refer_file, samp_rate=cfg.samp_rate)
        refer_sig = total_sig[ch]
        refer_stft = stft(refer_sig, cfg.frame_size, cfg.shift, fading=True, ceil=True)
        refer_phase = np.angle(refer_stft)
        cos_diff = np.cos(mixture_phase - refer_phase)
        sin_diff = np.sin(mixture_phase - refer_phase)
        spatial_feats.append(np.concatenate([cos_diff, sin_diff], axis=1))
    spatial_feats = np.concatenate(spatial_feats, axis=1)
    # read source target
    source_target = []
    source_sigs = []
    for spkr_i in range(num_spkrs):
        source_path = os.path.join(data_dir, "s%d"%(spkr_i + 1), file_name)
        source_sig = read_wav(source_path, samp_rate=cfg.samp_rate)[0]
        source_sigs.append(source_sig)
        source_stft = stft(source_sig, cfg.frame_size, cfg.shift, fading=True, ceil=True)
        source_target.append(np.abs(source_stft))
    source_target = np.stack(source_target, axis=2)
    source_sigs = np.stack(source_sigs, axis=0)
    argmax_idx = np.reshape(np.argmax(source_target, axis = 2),-1)
    seq_len, feat_len = mixture_magn.shape
    source_label = np.eye(num_spkrs)[argmax_idx]
    source_label = np.reshape(source_label, [seq_len, feat_len, num_spkrs])
    group_data = ([[spatial_feats]], [[mixture_magn]], [[source_target]], [[seq_len]],[[silence_mask]],[[source_label]])

    return group_data, mixture_magn, mixture_phase, mixture_sig, source_sigs

if __name__ == "__main__":
    num_gpu = 1
    job_dir = os.path.join(cfg.job_dir, cfg.test_name)
    est_dir = os.path.join(job_dir, "estimate")
    num_spkrs = cfg.num_spkrs
    for i in range(num_spkrs):
        est_src_dir = os.path.join(est_dir, "s%d"%(i+1))
        create_folders(est_src_dir)
    test_dir = cfg.test_dir
    file_list = os.path.join(test_dir, "target.lst")
    with open(file_list, "r") as f:
        mixture_lst = [line.strip().split()[0] for line in f.readlines()]
    snr_file = os.path.join(est_dir, 'snr.ans')
    sf = open(snr_file, 'a+')
    bss_file = os.path.join(est_dir, 'bss.ans')
    bf = open(bss_file, 'a+')
    pesq_file = os.path.join(est_dir, 'pesq.ans')
    pf = open(pesq_file, 'a+')
    try:
        with tf.Graph().as_default():
            sess_config = tf.ConfigProto()
            sess_config.allow_soft_placement = True
            sess_config.gpu_options.allow_growth = True
            sess = tf.Session(config=sess_config)
            with tf.variable_scope("SpeechSeparation"):
                model = DF_Model(sess, cfg, num_gpu)
                model.restore_model()
                num = 0
                for file_idx, file_name in enumerate(mixture_lst):
                    group_data, mix_magn, mix_phase, mix_sig, src_sigs = prepare_feat(
                        test_dir, file_name)
                    # get embedding
                    #recon_magn = model.get_pred(group_data)
                    #sig_len = len(mix_sig)
                    #print(np.array(recon_magn).shape)
                    #recon_magn = recon_magn[0][0]
                    
                    embed = model.get_pred(group_data)
                    embed = embed[0][0]
                    sig_len = len(mix_sig)
                    seq_len, feat_dim, embed_dim = embed.shape
                    flat_embed = np.reshape(embed, [-1, embed_dim])
                    kmeans = KMeans(n_clusters = num_spkrs, random_state = 0).fit(flat_embed)
                    pred_labels = np.eye(num_spkrs)[kmeans.labels_]
                    pred_labels = np.reshape(pred_labels, [seq_len, feat_dim, num_spkrs])
                    recon_magn = pred_labels *np.expand_dims(mix_magn, axis = 2)
                    recon_sigs = []
                    for spkr_i in range(num_spkrs):
                        recon_sig = get_recon_sig(recon_magn[:,:,spkr_i], mix_phase,
                                cfg.frame_size, cfg.shift, fading=True)
                        recon_sigs.append(recon_sig[0:sig_len])
                    recon_sigs = np.stack(recon_sigs, axis=0)
                    anchor_sigs = np.tile(mix_sig, [num_spkrs, 1])
                    # first cal sdr & popt to decide the permutation
                    sdr, sir, sar, popt = bss_eval_sources(src_sigs, recon_sigs)
                    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_sigs, anchor_sigs)
                    bss_line = np.concatenate([sdr, sir, sar, popt, sdr0, sir0, sar0, popt0], axis=0)
                    for item in bss_line:
                        bf.write("%.4f "%item)
                    bf.write("\n")
                    # cal sisnr and pesq
                    sisnr_lst = []
                    sisnr_base_lst = []
                    #pesq_lst = []
                    #pesq_base_lst = []
                    mix_path = os.path.join(test_dir, "ch%d"%cfg.target_channel, file_name)
                    for spkr_i in range(num_spkrs):
                        src_path = os.path.join(test_dir, cfg.target_dir,
                            "s%d"%(spkr_i + 1), file_name)
                        dst_path = os.path.join(est_dir, "s%d"%(spkr_i+1), file_name)
                        audiowrite(dst_path, recon_sigs[popt[spkr_i]], samp_rate=cfg.samp_rate)
                        sisnr = get_SISNR(src_sigs[spkr_i], recon_sigs[popt[spkr_i]])
                        sisnr_base = get_SISNR(src_sigs[spkr_i], mix_sig)
                        sisnr_lst.append(sisnr)
                        sisnr_base_lst.append(sisnr_base)
                        #pesq = float(getPESQ(src_path, dst_path, cfg.samp_rate))
                        #pesq_base = float(getPESQ(src_path, mix_path, cfg.samp_rate))
                        #pesq_lst.append(pesq)
                        #pesq_base_lst.append(pesq_base)
                    snr_line = np.concatenate([sisnr_lst, sisnr_base_lst], axis=0)
                    #pesq_line = np.concatenate([pesq_lst, pesq_base_lst], axis=0)
                    for item in snr_line:
                        sf.write("%.4f "%item)
                    sf.write('\n')
                    #for item in pesq_line:
                    #    pf.write("%.4f "%item)
                    pf.write('\n')
                    print("Sentence {}".format(file_name))
                    print("sdr: {:.4f}, sdri: {:.4f}, sisnr: {:.4f}, sisnr_inc: {:.4f} "
                          "pesq: {:.4f}".format(np.mean(sdr),
                          np.mean(sdr - sdr0), np.mean(sisnr_lst),
                          np.mean(sisnr_lst) - np.mean(sisnr_base_lst),
                          np.mean(sisnr_lst)))
                    if (file_idx + 1) % 5 == 0:
                        sf.flush()
                        bf.flush()
                        pf.flush()
        sf.close()
        bf.close()
        pf.close()
    except Exception as e:
        logging.error("evaluating exception: %s"%e)
        traceback.print_exc()
