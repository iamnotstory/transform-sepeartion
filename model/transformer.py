import os, sys
proj_path = os.path.abspath('..')
sys.path.append(proj_path)
import logging
import time
import numpy as np
import tensorflow as tf
from utils.tools import *
import itertools
import math
class DF_Model(object):
    def __init__(self, sess, config, num_gpu, initializer=None):
        self.session = sess
        self.config = config
        self.num_gpu = num_gpu
        self.epoch_counter = 0
        self.initializer = initializer
        self.eps = 1e-20
        if hasattr(config, "global_cmvn_file") and config.global_cmvn_file != '':
            self.cmvn = self.read_cmvn(config.global_cmvn_file)
        else:
            self.cmvn = None
        self.global_step = tf.get_variable("global_step", shape=[], trainable=False,
                                           initializer=tf.constant_initializer(0),
                                           dtype=tf.int32)
        # define placeholder
        self.lr = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        self.create_placeholders()
        self.training = tf.placeholder(tf.bool, shape=[])
        # init graph
        self.optimize()
        self.reset()
        self.create_saver()
        # create job_env
        self.job_dir = config.job_dir
        create_folders(self.job_dir)
        self.best_loss_dir = os.path.join(config.job_dir, "best_loss")
        create_folders(self.best_loss_dir)
        train_event_dir = os.path.join(config.job_dir, "train_event")
        dev_event_dir = os.path.join(config.job_dir, "dev_event")
        create_folders(train_event_dir)
        create_folders(dev_event_dir)
        self.train_writer = tf.summary.FileWriter(train_event_dir, sess.graph)
        self.dev_writer = tf.summary.FileWriter(dev_event_dir)
    
    def create_saver(self):
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep = 5)
        self.best_loss_saver = tf.train.Saver(tf.global_variables(), max_to_keep = 1)
        a = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #IPD_key = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'SpeechSeparation/Model/IPD_DC')
        #print(IPD_key)
        #self.ipdsaver = tf.train.Saver(IPD_key)
        #if self.config.ipdonly == False:
        #DC_key = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'SpeechSeparation/Model/DC')
        #self.magnsaver = tf.train.Saver(DC_key)

    def preload(self, sess):
        self.ipdsaver.restore(sess,self.config.ipd_dir)
        if self.config.ipdonly == False:
            self.magnsaver.restore(sess,self.config.magn_dir)

    def read_cmvn(self, file_path):
        cmvn = np.loadtxt(file_path).astype(np.float32)
        cmvn[:, 1] = np.sqrt(cmvn[:, 1])  # get standard deviation
        return cmvn

    def create_placeholders(self):
        self._IPD_feat = []
        self._MAGN_feat = []
        self._source_feat = []
        self._seq_len = []
        self._silence_mask = []
        self._source_label = []
        self._gw_matrix = []
        self.training = tf.placeholder(tf.bool, shape = [])
        feat_dim = self.config.feat_dim
        num_spkrs = self.config.num_spkrs
        num_refers = self.config.num_refers
        for i in range(self.num_gpu):
            self._IPD_feat.append(tf.placeholder(tf.float32, shape=[None, None, feat_dim * 2 * num_refers]))
            self._MAGN_feat.append(tf.placeholder(tf.float32, shape=[None, None, feat_dim]))
            self._source_feat.append(tf.placeholder(tf.float32, shape=[None, None, feat_dim, num_spkrs]))
            self._silence_mask.append(tf.placeholder(tf.float32, shape = [None, None, feat_dim]))
            self._seq_len.append(tf.placeholder(tf.int32, shape=[None]))
            self._source_label.append(tf.placeholder(tf.float32, shape = [None, None, feat_dim, num_spkrs]))
            self._gw_matrix.append(tf.placeholder(tf.float32, shape = [None, None, None]))
        print("Last Dim of df_feat: {}".format(self._IPD_feat[0].get_shape().as_list()[-1]))
    def cmvn_process(self, IPD, magn):
        if self.cmvn is not None:
            input_feat = tf.concat([IPD, magn], axis = 2)
            input_feat = (input_feat - self.cmvn[:, 0]) / self.cmvn[:,1]
            IPD_cos, IPD_sin, magn = tf.split(input_feat, 3, axis = 2)
            IPD = tf.concat([IPD_cos, IPD_sin], axis = 2)
        return IPD, magn
        
    def reset(self):
        self.batch_counter = 0
        self.total_loss = 0.0
        self.latest_batch_counter = 0
        self.latest_loss = 0.0
        self.epoch_counter += 1

    def gw_encoding(self, gw_matrix):
        #cfg = self.config
        #res = []
        #for i in range(length):
        #    tmp = []
        #    for t in range(length):
        #        tmp.append(-np.abs(i - t) ** 2)
        #    res.append(tmp)
        #res = np.array(res)
        

        yeta = tf.get_variable(name='eta', shape = [1], dtype = tf.float32)
        res = tf.exp(gw_matrix / tf.square(yeta))

        #gw_ind = tf.tile(tf.expand_dims(tf.range(time_step),0), [batch_size, 1])
        #gw_enc = tf.convert_to_tensor(res, tf.float32)
        #gw_ind_ = tf.tile(tf.expand_dims(0,tf.range(time_step)),[batch_size, 1])
        #outputs = tf.nn.embedding_lookup(gw_enc, gw_ind)
        #outputs = tf.nn.embedding_lookup(outputs, gw_ind_)
        return res
    def multihead_attention(self, feat_input):
        cfg = self.config
        total_units = cfg.transformer_total_units
        num_heads = cfg.transformer_heads
        num_units = total_units / num_heads
        attention_out = []
        time_step = tf.shape(feat_input)[1]
        batch_size = tf.shape(feat_input)[0]
        for i in range(num_heads):
            with tf.variable_scope('head_%d'%(i+1)):
                Q = tf.layers.dense(feat_input, num_units, activation = None, name = 'Q', use_bias = None)
                K = tf.layers.dense(feat_input, num_units, activation = None, name = 'K', use_bias = None)
                V = tf.layers.dense(feat_input, num_units, activation = None, name = 'V', use_bias = None)
                gw = self.gw_encoding(self._gw_matrix[0])
                cl = tf.multiply(tf.matmul(Q, tf.transpose(K, [0,2,1])), 1.0 / math.sqrt(float(num_units)))
                sl = tf.multiply(gw, cl)
                sl = tf.abs(sl)
                sl = tf.nn.softmax(sl)
                attention_probs = sl
                attention_out_one = tf.matmul(sl, V)
                attention_out.append(attention_out_one)
                #attention_probs = tf.nn.softmax(tf.multiply(tf.matmul(Q,K, transpose_b = True), 1.0/ math.sqrt(float(num_units))))
                
                #attention_probs_f = tf.transpose(tf.nn.softmax(tf.multiply(tf.matmul(tf.transpose(Q, [0,2,1]), K),1.0/math.sqrt(float(num_units)))), [0,2,1])
                #attention_out.append(tf.matmul(tf.matmul(attention_probs, V), attention_probs_f))
        self.attention_probs = attention_probs
        attention_out = tf.concat(attention_out, 2)
        attention_out = tf.layers.dense(attention_out, total_units, activation = tf.nn.relu)
        attention_out = tf.contrib.layers.layer_norm(attention_out + feat_input)
        return attention_out
    
    def transformer(self, feat_input):
        cfg = self.config
        total_units = cfg.transformer_total_units
        num_layers = cfg.transformer_num_layers
        trm_input = feat_input
        for i in range(num_layers):
            with tf.variable_scope('transformer_%d'%(i+1)):
                trm_input = self.multihead_attention(trm_input)
                trm_input_tmp = tf.layers.dense(trm_input, total_units, activation = tf.nn.relu, name = 'ff', use_bias = None)
                trm_input = tf.contrib.layers.layer_norm(trm_input + trm_input_tmp)
        #trm_out = tf.contrib.layers.layer_norm(feat_input + trm_input)
        trm_out = trm_input
        return trm_out
    
    def positional_encoding(self,batch_size, time_step, feat_dim):
        with tf.variable_scope('pe'):
            position_ind = tf.tile(tf.expand_dims(tf.range(time_step),0), [batch_size, 1])
            position_enc = np.array([[pos / np.power(10000, (i - i%2) / feat_dim) for i in range(feat_dim)] for pos in range(3000)])
            position_enc[:,0::2] = np.sin(position_enc[:,0::2])
            position_enc[:,1::2] = np.cos(position_enc[:,1::2])
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)

            outputs = tf.nn.embedding_lookup(position_enc, position_ind)
        return tf.to_float(outputs)

    def transformer_embedding(self, feat_input):
        cfg = self.config
        feat_dim = cfg.feat_dim
        E = cfg.embedding_dim
        project_size = feat_dim * E
        time_step = tf.shape(feat_input)[1]
        batch_size = tf.shape(feat_input)[0]
        #pe = self.positional_encoding(batch_size, time_step, feat_dim ) 
        #feat_input = tf.concat([feat_input, pe], axis = 2)
        
        with tf.variable_scope('embedding'):
            feat_input = self.transformer(feat_input)
            s_project = tf.layers.dense(feat_input, project_size, activation = tf.nn.tanh, name = 'emb', use_bias = True)
            embedding = tf.reshape(s_project, [batch_size, time_step, feat_dim, E])
            norm_embedding = tf.nn.l2_normalize(embedding, 3)
        return norm_embedding
    
    def IPD_filter(self, IPD_input, seq_len):
        rnn_type = self.config.rnn_type
        hidden_size = self.config.hidden_size
        feat_dim = self.config.feat_dim
        N = self.config.num_spkrs
        E = self.config.embedding_dim
        project_size = feat_dim * E
        num_layers = self.config.num_layers
        bidirectional = self.config.bidirectional
        dropout_prob_fw = tf.cond(self.training, lambda:self.config.dropout_keep_fw, lambda:1.0)
        dropout_prob_rc = tf.cond(self.training, lambda:self.config.dropout_keep_rc, lambda:1.0)
        batch_size = tf.shape(IPD_input)[0]
        time_step = tf.shape(IPD_input)[1]
        rnn_cell = tf.contrib.rnn.LSTMCell
        rnn_input = IPD_input
        with tf.variable_scope("IPD_DC"):
            for i in range(num_layers):
                with tf.variable_scope("{}_{}".format(rnn_type,i+1)):
                    fw_cell = rnn_cell(hidden_size)
                    bw_cell = rnn_cell(hidden_size)
                    #if i != num_layers -1:
                    #    fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob = dropout_prob_fw,
                    #            state_keep_prob = dropout_prob_rc)
                    #    bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob = dropout_prob_fw,
                    #            state_keep_prob = dropout_prob_rc)
                    initial_fw = fw_cell.zero_state(batch_size, dtype = tf.float32)
                    initial_bw = bw_cell.zero_state(batch_size, dtype = tf.float32)
                    output, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, rnn_input, 
                            sequence_length = tf.to_int32(seq_len), initial_state_fw = initial_fw,
                            initial_state_bw = initial_bw, time_major = False, dtype = tf.float32)
                    output = tf.concat(output,axis = 2)
                    rnn_input = output
            s_project = tf.layers.dense(output, project_size, activation = tf.nn.tanh, name = "FC", use_bias = True)
            embedding = tf.reshape(s_project, [batch_size, time_step, feat_dim, E])
            norm_embedding = tf.nn.l2_normalize(embedding,3)

        return norm_embedding

    
    def DC_loss(self, embedding, source_label, mask):
        #print(embedding.shape, source_label.shape, mask.shape)
        
        V = tf.einsum('btfe,btf->btfe', embedding, mask)
        Y = tf.einsum('btfn,btf->btfn', source_label, mask)
        V_ = []
        Y_ = []
        index = tf.range(100,300)
        for i in range(self.config.batch_size):
            V_.append(tf.nn.embedding_lookup(V[i], index))
            Y_.append(tf.nn.embedding_lookup(Y[i], index))
        V = tf.convert_to_tensor(V_)
        Y = tf.convert_to_tensor(Y_)
        l2_VTV = tf.reduce_sum(tf.einsum('btfe,btfp->bep',V,V)**2/2, axis = [1,2])
        l2_YTY = tf.reduce_sum(tf.einsum('btfn,btfp->bnp',Y,Y)**2/2, axis = [1,2])
        l2_VTY = tf.reduce_sum(tf.einsum('btfe,btfn->ben',V,Y)**2/2, axis = [1,2])
        loss = l2_VTV + l2_YTY - 2 * l2_VTY
        return loss
    def dc_loss(self, embedding , source_label, mask):
        V = tf.einsum('btfe,btf->btfe', embedding, mask)
        Y = tf.einsum('btfn,btf->btfn', source_label, mask)
        V = tf.reshape(V, ( -1, 400 * 129, 20))
        Y = tf.reshape(Y, ( -1, 400 * 129, 2))
        #VTV = tf.matmul(V, V ,transpose_b = True)
        #YTY = tf.matmul(Y, Y, transpose_b = True)
        #loss = tf.reduce_sum((VTV - YTY) ** 2, axis = [1,2,3])
        loss = tf.norm(tf.matmul(V,V, transpose_b = True) - tf.matmul(Y,Y,transpose_b = True) , axis = (1,2))
        #loss = loss ** 2
        return loss

    def tower_cost_DC(self, IPD, magn, silence_mask, source_label, seq_len):
        time_step = tf.shape(IPD)[1]
        feat_dim = self.config.feat_dim
        input_feat = IPD
        #magn = tf.log(1 + magn)    
        IPD, magn = self.cmvn_process(IPD, magn)
        magn = tf.log(1 + magn)
        input_feat = tf.concat([IPD, magn], axis = 2)
        seq_mask = self.get_seq_mask(time_step, seq_len)
        count_bins = tf.cast(seq_len/2 * feat_dim, tf.float32)
        silence_mask = tf.cast(silence_mask, tf.float32)
        masks = tf.einsum('bt,btf->btf', seq_mask, silence_mask)
        #if self.config.DC_new == False:
        embedding = self.transformer_embedding(input_feat)
        #self.attention_out = self.transformer_embedding(input_feat) 
        loss = self.DC_loss(embedding, source_label, masks)
        #print(self.dc_loss(embedding, source_label, masks), self.DC_loss(embedding, source_label, masks))
        #self.loss2 = self.DC_loss(embedding, source_label, masks) / count_bins
        #self.loss1 = loss/count_bins
        loss = loss/count_bins
        loss = tf.reduce_mean(loss)
        return loss, embedding


    def get_seq_mask(self, max_len, seq_len):
        r = tf.range(max_len, dtype=tf.int32)
        func = lambda x: tf.cast(tf.less(r, x), tf.float32)
        seq_mask = tf.map_fn(func, seq_len, dtype=tf.float32)
        return seq_mask

    def tower_cost(self, IPD_feat, magn_feat,silence_mask, source_feat, seq_len):
        max_len = tf.shape(magn_feat)[1]
        seq_mask = self.get_seq_mask(max_len, seq_len)

        masks = self.total_filter(IPD_feat, magn_feat,  seq_len,silence_mask)
        loss,pred = self.pie_mse_loss(magn_feat,  source_feat,masks,  seq_mask,silence_mask)
        loss = tf.reduce_mean(loss)
        return loss, pred

    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.lr)
        tower_grads = []
        tower_cost = []
        tower_pred = []
        for i in range(self.num_gpu):
            worker = "/gpu:%d"%i
            device_setter = tf.train.replica_device_setter(
                worker_device=worker, ps_device='/cpu:0', ps_tasks=1)
            with tf.variable_scope("Model", reuse=(i>0)):
                with tf.device(device_setter):
                    with tf.name_scope("tower_%d"%i):
                        if self.config.DC == False:
                            cost, pred = self.tower_cost(self._IPD_feat[i],
                            self._MAGN_feat[i],self._silence_mask[i], self._source_feat[i], self._seq_len[i])
                        else:
                            
                            cost, pred = self.tower_cost_DC(self._IPD_feat[i], self._MAGN_feat[i], self._silence_mask[i], self._source_label[i], self._seq_len[i])
                        #print(cost)
                        key = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'SpeechSeparation/Model')
                        print(key)
                        if self.config.stable == True:
                            grads = optimizer.compute_gradients(cost,var_list = key)
                        else:
                            grads = optimizer.compute_gradients(cost)
                        tower_grads.append(grads)
                        tower_cost.append(cost)
                        tower_pred.append(pred)
        #grads = average_gradients(tower_grads, self.config.max_grad_norm)
        grads = tower_grads[0]
        
        self.apply_gradients_op = optimizer.apply_gradients(grads, global_step=self.global_step)
        self.avg_cost = tf.reduce_mean(tower_cost)
        self.tower_pred = tower_pred
        tf.summary.scalar('avg_cost', self.avg_cost)
        self.merged = tf.summary.merge_all()

    def run_batch(self, group_data, learning_rate):
        feed_dict = {self.training: True, self.lr: learning_rate}
        step_size = 0
        for i in range(self.num_gpu):
            feed_dict[self._IPD_feat[i]] = group_data[0][i]
            feed_dict[self._MAGN_feat[i]] = group_data[1][i]
            if self.config.DC == False:
                feed_dict[self._source_feat[i]] = group_data[2][i]
            else:
                feed_dict[self._source_label[i]] = group_data[5][i]
            feed_dict[self._seq_len[i]] = group_data[3][i]
            feed_dict[self._silence_mask[i]] = group_data[4][i]
            feed_dict[self._gw_matrix[i]] = group_data[6][i]
            step_size += len(group_data[3][i])
        start_time = time.time()
        _, i_global, i_merge, loss = self.session.run(
            [self.apply_gradients_op, self.global_step, self.merged, self.avg_cost],
            feed_dict=feed_dict)
        self.total_loss += loss
        self.latest_loss += loss
        self.batch_counter += 1
        self.latest_batch_counter += 1
        duration = time.time() - start_time
        if i_global % self.config.log_period == 0:
            logging.info("Epoch {:d}, Average Train MSE: {:.6f}={:.6f}/{:d}, "
                         "Latest MSE: {:.6f}, Speed: {:.2f} sentence/sec".format(
                         self.epoch_counter, self.total_loss / self.batch_counter,
                         self.total_loss, self.batch_counter,
                         self.latest_loss / self.latest_batch_counter,
                         step_size / duration))
            print(self.latest_loss / self.latest_batch_counter)
            self.latest_loss = 0.0
            self.latest_batch_counter = 0
        if i_global % self.config.save_period == 0:
            self.save_model(i_global)
        return i_global

    def get_pred(self, group_data):
        feed_dict = {self.training: False}
        for i in range(self.num_gpu):
            feed_dict[self._IPD_feat[i]] = group_data[0][i]
            feed_dict[self._MAGN_feat[i]] = group_data[1][i]
            if self.config.DC == False:
                feed_dict[self._source_feat[i]] = group_data[2][i]
            else:
                feed_dict[self._source_label[i]] = group_data[5][i]
            feed_dict[self._seq_len[i]] = group_data[3][i]
            feed_dict[self._silence_mask[i]] = group_data[4][i]
            feed_dict[self._gw_matrix[i]] = group_data[6][i]
        pred = self.session.run(self.tower_pred, feed_dict=feed_dict)
        return pred
    def get_attention(self, group_data):
        feed_dict = {self.training: False}
        for i in range(self.num_gpu):
            feed_dict[self._IPD_feat[i]] = group_data[0][i]
            feed_dict[self._MAGN_feat[i]] = group_data[1][i]
            if self.config.DC == False:
                feed_dict[self._source_feat[i]] = group_data[2][i]
            else:
                feed_dict[self._source_label[i]] = group_data[5][i]
            feed_dict[self._seq_len[i]] = group_data[3][i]
            feed_dict[self._silence_mask[i]] = group_data[4][i]
        pred = self.session.run(self.attention_out, feed_dict=feed_dict)
        return pred

    def valid(self, reader):
        total_loss, batch_counter = 0.0, 0
        num_sent = 0
        logging.info("Start to dev")
        start_time = time.time()
        while True:
            batch_data = reader.next_batch()
            if batch_data is None:
                break
            else:
                feed_dict = {self.training: False}
                for i in range(self.num_gpu):
                    feed_dict[self._IPD_feat[i]] = batch_data[0][i]
                    feed_dict[self._MAGN_feat[i]] = batch_data[1][i]
                    if self.config.DC == False:
                        feed_dict[self._source_feat[i]] = batch_data[2][i]
                    else:
                        feed_dict[self._source_label[i]] = batch_data[5][i]
                    feed_dict[self._seq_len[i]] = batch_data[3][i]
                    feed_dict[self._silence_mask[i]] = batch_data[4][i]
                    feed_dict[self._gw_matrix[i]] = batch_data[6][i]
                    num_sent += len(batch_data[3][i])
                loss = self.session.run(self.avg_cost, feed_dict=feed_dict)
                total_loss += loss
                batch_counter += 1
                if batch_counter % 10 == 0:
                    logging.info("Dev Sentence {:d}, AVG Dev MSE: {:.6f}={:.6f}/{:d}, "
                                 "Speed: {:.2f} sentence/sec".format(
                                 num_sent, total_loss / batch_counter, total_loss,
                                 batch_counter, num_sent / (time.time() - start_time)))
        duration = time.time() - start_time
        avg_loss = total_loss / batch_counter
        dev_summary = create_valid_summary(avg_loss)
        i_global = self.session.run(self.global_step)
        self.dev_writer.add_summary(dev_summary, i_global)
        logging.info("Finish dev {:d} sentences in {:.2f} seconds, "
                     "AVG MSE: {:.6f}".format(num_sent, duration, avg_loss))
        return avg_loss

    def save_model(self, i_global):
        model_path = os.path.join(self.job_dir, "model.ckpt")
        self.saver.save(self.session, model_path, global_step=i_global)
        logging.info("Saved model, global_step={}".format(i_global))

    def restore_model(self):
        load_option = self.config.load_option
        if load_option == 0:
            load_path = tf.train.latest_checkpoint(self.job_dir)
        elif load_option == 1:
            load_path = tf.train.latest_checkpoint(self.best_loss_dir)
        else:
            load_path = self.config.load_path
        try:
            self.saver.restore(self.session, load_path)
            logging.info("Loaded model from path {}".format(load_path))
        except Exception as e:
            logging.error("Failed to load model from {}".format(load_path))
            raise e


