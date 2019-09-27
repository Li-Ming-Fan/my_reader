

import tensorflow as tf


from zoo_layers import rnn_layer
from zoo_layers import dense

#
def do_encoding(sett, emb_q, emb_p, len_q, len_p, keep_prob):
    """
    """
    with tf.variable_scope('encoding'):
        encoded_q = rnn_layer(emb_q, len_q, sett.encd_size, keep_prob, scope="q")
        output_q = dense(encoded_q, sett.encd_size, scope="q")
        #
        encoded_p = rnn_layer(emb_p, len_p, sett.encd_size, keep_prob, scope="p")
        output_p = dense(encoded_p, sett.encd_size, scope="p")
    
    return output_q, output_p


def do_matching(sett, encoded_p, encoded_q, mask_p, mask_q, keep_prob):
    """ encoded_p: [B*N, TP, D]
        encoded_q: [B*N, TQ, D]
        mask_q: [B*N, TQ]
    """
    with tf.variable_scope("matching"):
        dp = tf.nn.dropout(encoded_p, keep_prob=keep_prob)
        dq = tf.nn.dropout(encoded_q, keep_prob=keep_prob)
        TP = tf.shape(encoded_p)[1]
        
        with tf.variable_scope("attention"):
            rp = tf.nn.relu(dense(dp, sett.att_dim, use_bias=False, scope="p"))
            rq = tf.nn.relu(dense(dq, sett.att_dim, use_bias=False, scope="q"))
            #
            att_mat = tf.matmul(rp, tf.transpose(rq, [0, 2, 1])) / (sett.att_dim ** 0.5)
            # att_mat: [B*N, TP, TQ]
            #
            mask_q_d3 = tf.tile(tf.expand_dims(mask_q, axis=1), [1, TP, 1])
            att_mat = tf.add(att_mat, 1e30 * (tf.cast(mask_q_d3, tf.float32) - 1.0))
            #
            att_prob = tf.nn.softmax(att_mat)
            value_pooled = tf.matmul(att_prob, encoded_q)
            #
            return value_pooled
            #
    
def do_featuring(sett, matched_p, len_p, keep_prob):
    """
    """
    with tf.variable_scope('featuring'):
        feat_s = rnn_layer(matched_p, len_p, sett.feat_size, keep_prob, scope="s")
        feat_s = dense(feat_s, sett.feat_size, scope="s")
        #
        input_e = tf.concat([feat_s, matched_p], -1)
        #
        feat_e = rnn_layer(input_e, len_p, sett.feat_size, keep_prob, scope="e")
        feat_e = dense(feat_e, sett.feat_size, scope="e")
    
    return feat_s, feat_e


#
def do_meshing_scores(start_scores, end_scores):
    """ start_scores: [B, N, TP]
        end_scores: [B, N, TP]
    """
    TP = tf.shape(start_scores)[2]
    #
    # mesh
    s_tiled = tf.tile(tf.expand_dims(start_scores, 3), [1, 1, 1, TP])
    e_tiled = tf.tile(tf.expand_dims(end_scores, 2), [1, 1, TP, 1])
    span_probs = s_tiled * e_tiled
    #
    # 取右上三角
    # span_probs = tf.linalg.band_part(span_probs, 0, -1)  # [B, NP, TP, TP]
    mask = tf.linalg.band_part(tf.ones_like(span_probs), 0, -1)
    span_probs = span_probs + 1e30 * (mask - 1.0)
    #
    # reshape & norm
    shape_probs = tf.shape(span_probs)
    B = shape_probs[0]
    NP = shape_probs[1]
    # TP = shape_probs[2]
    #
    span_probs_reshaped = tf.reshape(span_probs, [B, -1])   # [B, N*T*T]
    span_probs_reshaped = tf.nn.softmax(span_probs_reshaped, -1)
    #
    span_probs_normed = tf.reshape(span_probs_reshaped, [B, NP, TP, TP])   # [B, N, T, T]
    #
    
    #
    # 找出最大位置
    posi_1d = tf.argmax(span_probs_reshaped, -1, output_type=tf.int32)
    #
    # parse
    TP2 = TP * TP
    idx_passage = tf.floor_div(posi_1d, TP2)
    #
    posi_text = posi_1d - idx_passage * TP2
    idx_start = tf.floor_div(posi_text, TP)
    #
    idx_end = posi_text - idx_start * TP
    #
    
    #
    return span_probs_normed, idx_passage, idx_start, idx_end
    #

