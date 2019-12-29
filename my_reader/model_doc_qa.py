# -*- coding:utf8 -*-


import tensorflow as tf


from Zeras.model_baseboard import ModelBaseboard

from model_modules import do_encoding, do_matching, do_featuring
from model_modules import do_meshing_scores

from zoo_layers import dense


class ModelDocQA(ModelBaseboard):
    """
    """
    def __init__(self, settings):
        """
        """
        super(ModelDocQA, self).__init__(settings)
        #
        self.pb_outputs_name = ["vs_gpu/scores/FloorDiv",
                                "vs_gpu/scores/FloorDiv_1",
                                "vs_gpu/scores/sub_2",
                                "vs_gpu/scores/Reshape_5" ]
        self.pb_input_names = {"questions": "input_q:0",
                               "passages": "input_p:0" }
        self.pb_output_names = {}
        self.pb_output_names["span_probs"] = "vs_gpu/scores/Reshape_5:0"
        self.pb_output_names["idx_passage"] = "vs_gpu/scores/FloorDiv:0"
        self.pb_output_names["idx_start"] = "vs_gpu/scores/FloorDiv_1:0"
        self.pb_output_names["idx_end"] = "vs_gpu/scores/sub_2:0"
        #
        self.debug_tensor_names = []
        #
        
    def build_placeholder(self):
        """
        """        
        q = tf.placeholder(tf.int32, [None, None], name="input_q")          # [B, TQ]
        p = tf.placeholder(tf.int32, [None, None, None], name="input_p")    # [B, NP, TP]
        #
        passage_idx = tf.placeholder(tf.int32, [None], name="passage_idx")  # [B,]
        start_label = tf.placeholder(tf.int32, [None], name="start_posi")   #
        end_label = tf.placeholder(tf.int32, [None], name="end_posi")
        #
        
        #
        input_tensors = {}
        input_tensors["batch_questions"] = q
        input_tensors["batch_passages"] = p
        #
        label_tensors = {}
        label_tensors["batch_passage_idx"] = passage_idx
        label_tensors["batch_start_label"] = start_label
        label_tensors["batch_end_label"] = end_label
        #
        print(input_tensors)
        #
        return input_tensors, label_tensors
    
    def build_inference(self, input_tensors):
        """
        """
        sett = self.settings
        questions = input_tensors["batch_questions"]
        passages = input_tensors["batch_passages"]
        
        # keep_prob
        keep_prob = tf.get_variable("keep_prob", shape=[], dtype=tf.float32, trainable=False)
        #
        print(keep_prob)
        #
        
        # reshape passages
        shape_passages = tf.shape(passages)
        batch_size = shape_passages[0]
        num_psg_per_question =  shape_passages[1]
        psg_length = shape_passages[2]
        #
        passages_reshaped = tf.reshape(passages, [-1, psg_length])  # [B*N, TP]
        #
        
        # embedding
        trunc_norm_init = tf.truncated_normal_initializer(stddev=sett.trunc_norm_init_std)
        
        # Add embedding matrix (shared by the encoder and decoder inputs)
        with tf.device('/cpu:0'), tf.variable_scope('embedding'):
            embedding = tf.get_variable('embedding', [sett.vocab.size(), sett.emb_dim],
                                        dtype = tf.float32, initializer = trunc_norm_init)
            emb_q = tf.nn.embedding_lookup(embedding, questions)
            emb_p = tf.nn.embedding_lookup(embedding, passages_reshaped)
            #
        
        # mask and length
        mask_q = tf.cast(tf.cast(questions, dtype=tf.bool), dtype=tf.int32)           # [B, TQ]
        mask_p = tf.cast(tf.cast(passages_reshaped, dtype=tf.bool), dtype=tf.int32)   # [B*N, TP]
        q_len = tf.reduce_sum(mask_q, -1)
        p_len = tf.reduce_sum(mask_p, -1)
        
        # encoding
        encoded_q, encoded_p = do_encoding(sett, emb_q, emb_p, q_len, p_len, keep_prob)
        # encoded_q, [B, TQ, D]
        # encoded_p, [B*N, TP, D]
        #
        
        # reshape questions
        shape_questions = tf.shape(encoded_q)   # [B, TQ, D]
        qst_len = shape_questions[1]
        dim_encoding = encoded_q.get_shape().as_list()[2]
        #        
        encoded_q = tf.tile(tf.expand_dims(encoded_q, 1), [1, num_psg_per_question, 1, 1])   # [B, N, TQ, D]
        encoded_q = tf.reshape(encoded_q, [-1, qst_len, dim_encoding])                       # [B*N, TQ, D]
        #
        mask_q = tf.tile(tf.expand_dims(mask_q, 1), [1, num_psg_per_question, 1])     # [B, N, TQ]
        mask_q = tf.reshape(mask_q, [-1, qst_len])                                    # [B*N, TQ] 
        #
        
        # matching
        value_p = do_matching(sett, encoded_p, encoded_q, mask_p, mask_q, keep_prob)   # [B*N, TP, D]
        matched_p = tf.concat([encoded_p, value_p], axis=-1)
        
        # featuring
        feature_s, feature_e = do_featuring(sett, matched_p, p_len, keep_prob)   # [B*N, TP, D]
        
        # reshape for score
        D = sett.feat_size
        feature_s = tf.reshape(feature_s, [batch_size, num_psg_per_question, psg_length, D])  # [B, N, TP, D]
        feature_e = tf.reshape(feature_e, [batch_size, num_psg_per_question, psg_length, D])  # [B, N, TP, D]
        
        # score
        with tf.variable_scope('scores'):
            start_scores = dense(feature_s, 1, scope="s")   # [B, NP, TP, 1]
            end_scores = dense(feature_e, 1, scope="e")     # [B, NP, TP, 1]
            #
            start_scores = tf.squeeze(start_scores, -1)     # [B, NP, TP]
            end_scores = tf.squeeze(end_scores, -1)         # [B, NP, TP]
            #
            # mesh
            span_probs, idx_passage, idx_start, idx_end = do_meshing_scores(start_scores, end_scores)
            #
            # span_probs: [B, NP, TP, TP]
            # idx_passage: [B, ]
            #
        #
        output_tensors = {}
        output_tensors["span_probs"] = span_probs
        output_tensors["idx_passage"] = idx_passage
        output_tensors["idx_start"] = idx_start
        output_tensors["idx_end"] = idx_end
        #
        return output_tensors

    def build_loss_and_metric(self, output_tensors, label_tensors):
        """
        """
        span_prob = output_tensors["span_probs"]      # [B, N, T, T]
        passage_idx = label_tensors["batch_passage_idx"]    # [B, ]
        start_label = label_tensors["batch_start_label"]    # [B, ]
        end_label = label_tensors["batch_end_label"]        # [B, ]
        #
        batch_size = tf.shape(span_prob)[0]
        example_idx = tf.range(batch_size)            # [B, ]
        #
        indices = tf.stack([example_idx, passage_idx, start_label, end_label], axis=1)
        # [B, 4]
        #
        pred_probs = tf.gather_nd(span_prob, indices)  # [B, ]
        epsilon = 1e-10 # * tf.ones_like(pred_probs)
        loss = - tf.reduce_mean(tf.log(pred_probs + epsilon))
        #
        lossput_tensors = {}
        lossput_tensors["loss_model"] = loss
        lossput_tensors["pred_probs"] = pred_probs
        #
        return lossput_tensors



    
    