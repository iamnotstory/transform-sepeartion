# config file for training or other process
# check the config file before your process
# and import the file to get pararmeters
# the project is for enhancing the target speech with directional features

# data config
total_channels = 7
target_channel = 7
refer_channels = [1]  # set of channels would be referred
                     # range of [1, total_channels] excluding target_channel
spatial_style = "concat"  # ['concat', 'average']
# microphone array params, for computing directional feats
# shape of ring, target channel 'ch7' locates at center
c = 340  # speed of sound
radius = 0.1
relative_angles = [(ch - 1) * 60 - 180 for ch in refer_channels]
relative_distances = [radius for ch in refer_channels]

target_dir = "target"
beamed_dir = "DSB"
target_lst_file = "target.lst"
num_spkrs = 2
num_refers = 1
samp_rate = 8000
frame_duration = 0.032
frame_size = int(samp_rate * frame_duration)
feat_dim = frame_size // 2 + 1
shift_duration = 0.008
shift = int(samp_rate * shift_duration)
overlap_rate = 0
batch_size = 32
dev_batch_size = 32
min_queue_size = 20
load_file_num = 20
min_sent_len = 10
shorter_sent_len = 100
longer_sent_len = 400
embedding_dim = 20
# train & dev dataset
#train_dir = "/gpfs/share/home/1801213802/data//data/train"
#dev_dir = "/gpfs/share/home/1801213802/data//data/dev"
global_cmvn_file = 'list/ipd_magn_net_cmvn_ipd'
train_dir = "/gpfs/share/home/1801213802/data/mixreverb/2speakers_reverb/wav8k/min/tr"
dev_dir = "/gpfs/share/home/1801213802/data/mixreverb/2speakers_reverb/wav8k/min/cv"
train_anechoic_dir = "/gpfs/share/home/1801213802/data/mixreverb/2speakers_anechoic/wav8k/min/tr"
dev_anechoic_dir = "/gpfs/share/home/1801213802/data/mixreverb/2speakers_anechoic/wav8k/min/cv"
# job config
job_type = "train"
lamda = 100.0
#job_dir = "job/no_pretrain_2layer_nodrop_ipd_anechoic"
#job_dir = "job/no_preptrain_multidcl1_cmvn_npsm_lambda50"
#job_dir = "job/original_nreluqkv_1tr_test_50epoch"
job_dir = "job/gw_tet_small_test_add_relu_layer_3"
#job_dir = "job/original_nreluqkv"
#job_dir = "job/no_preptrain_multidc_cmvn"
#job_dir = "job/no_pretrain_2layer_nodrop_multidc"
#job_dir = "job/no_pretrain_multidcl1_lambda1000"

ipd_dir = "trained/IPD/model.ckpt"
magn_dir = "trained/magn/model.ckpt"
#job_dir = "job/DFonAngle_ReferAll_concatDF_4x600lstm_dropout0.5_psm_without_steeringvertor"
gpu_list = [0]

# model config
log_feat = True
rnn_type = "lstm"
bidirectional = True
num_layers = 2
hidden_size = 600  # unidirectional
dropout_keep_fw = 0.7
dropout_keep_rc = 1.0
phase_sensitive = True
threshold = -40
fcl_layers = 0
fcl_hidden_size = 600
# training param
seed = 123
resume = False
init_mean = 0.0
init_stddev = 0.02
max_grad_norm = 200
learning_rate = 1e-3
max_epoch = 300
pretrain_shorter_epoch = 0
log_period = 10
save_period = 500
dev_period = 3000
early_stop_count = 3
decay_lr_count = 1
decay_lr = 0.8
min_learning_rate = 1e-6

# test config
#test_dir = "/gpfs/share/home/1801213802/data//data/test"
test_dir = '/gpfs/share/home/1801213802/data/mixreverb/2speakers_reverb/wav8k/min/tt'
#test_anechoic_dir = '/gpfs/share/home/1801213802/data/mixreverb/2speakers_anechoic/wav8k/min/tt'
#test_name = "test_h1000"
#test_dir = "/gpfs/share/home/1801213802/data/mixreverb/2speakers_reverb/wav8k/min/cv"
#test_name = 'test_h1000_load1'
test_name = 'GW_load_option_1_old'
# load option
load_option = 1
load_path = ""
stable = False
silence = False
ipdonly = False
DC = True
DC_new = True
job_anechoic = False

#transformer config
transformer_total_units = 387
transformer_heads = 3
transformer_num_layers = 3
