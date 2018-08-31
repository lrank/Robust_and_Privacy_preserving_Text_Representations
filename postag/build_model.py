#! /usr/bin/env python

import os
import time
import datetime

import tensorflow as tf
import numpy as np

import tag_helpers
from adv_bilstm import adv_bilstm

from tensorflow.contrib import learn

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning rate alpha")

#  parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("num_tune_epochs", 50, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")



# Data Preparatopn
# ==================================================

np.random.seed(1001)
# Load data
FLAGS = tf.flags.FLAGS

max_doc_length, vocab_size, tag_size, \
    x_train, y_train, l_train, \
    x_tune, y_tune, g_tune, a_tune, l_tune, \
    x_dev_aave, y_dev_aave, l_dev_aave, \
    x_test_aave, y_test_aave, l_test_aave = tag_helpers.load_data()
# print( len(x_tune),len(y_tune), len(l_tune),len(g_tune), len(a_tune))


# FLAGS.batch_size
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.iteritems()):
    print("{}={}".format(attr.upper(), value))
print("")


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      intra_op_parallelism_threads=2,
      inter_op_parallelism_threads=2)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        rnn = adv_bilstm(
            seq_length = x_train.shape[1],
            vocab_size = vocab_size,
            embedding_size = FLAGS.embedding_dim,
            rnn_cell_size = 50,
            tag_size = tag_size,
            gender_num = 2,
            age_num = 2,
            )

        # Define Training procedure
        learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        adv_lambda = tf.placeholder(tf.float32, shape=[], name="adversarial_lambda")

        global_step = tf.Variable(0, name="global_step", trainable=False)
        all_var_list = tf.trainable_variables()

        optimizer_n = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                rnn.loss,
                global_step=global_step
                )

        var_gend = [var for var in all_var_list if 'gender' in var.name]
        var_age = [var for var in all_var_list if 'age' in var.name]
        assert( len(var_gend) == 2 and len(var_age) == 2 )
        var_d = var_gend + var_age
        disc_loss = rnn.gender_loss + rnn.age_loss
        optimizer_d = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                adv_lambda * disc_loss,
                var_list=var_d
                )

        var_g = [var for var in all_var_list if var not in var_d]
        optimizer_g = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                rnn.loss - adv_lambda * disc_loss,
                var_list=var_g,
                global_step=global_step
                )

        def tag_acc(y, l, p):
            cor = 0.
            total = 0.
            for i in range( len(l) ):
                for j in range( l[i] ):
                    total += 1
                    if y[i][j] == p[i][j]:
                        cor += 1
            return cor / total

        def tag_train_step(batch_x, batch_y, batch_l, optimizer, lr = 1e-4):
            feed_dict = {
              rnn.input_x: batch_x,
              rnn.input_y: batch_y,
              rnn.input_l: batch_l,
              learning_rate: lr,
              rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, tag_preds = sess.run(
                [optimizer, global_step,
                rnn.loss, rnn.predictions],
                feed_dict)
            # time_str = datetime.datetime.now().isoformat()
            print("0\t{}\t{}\t{}".format(step, loss, tag_acc(batch_y, batch_l, tag_preds) ) )

        def dev_step(batch_x, batch_y, batch_l, batch_g, batch_a, data_id=1):
            feed_dict = {
              rnn.input_x: batch_x,
              rnn.input_y: batch_y,
              rnn.input_l: batch_l,
              rnn.input_g: batch_g,
              rnn.input_a: batch_a,
              adv_lambda: 0.,
              rnn.dropout_keep_prob: 1.
            }
            step, loss, tag_preds, l_g, acc_g, l_a, acc_a = sess.run(
                [global_step,
                rnn.loss, rnn.predictions,
                rnn.gender_loss, rnn.gender_acc,
                rnn.age_loss, rnn.age_acc],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            ''' data_id: train:0 dev:1 test:2 '''
            acc = tag_acc(batch_y, batch_l, tag_preds)
            print("{}\t{}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}".format(
                data_id, step,
                loss, acc,
                l_g, acc_g, l_a, acc_a)
            )
            return acc

        def tag_tune_step(batch_x, batch_y, batch_l, batch_g, batch_a, optimizer, adv_lamb=0, lr = 1e-4, data_id=3):
            feed_dict = {
              rnn.input_x: batch_x,
              rnn.input_y: batch_y,
              rnn.input_l: batch_l,
              rnn.input_g: batch_g,
              rnn.input_a: batch_a,
              adv_lambda: adv_lamb,
              learning_rate: lr,
              rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, loss, tag_preds, l_g, acc_g, l_a, acc_a = sess.run(
                [optimizer, global_step,
                rnn.loss, rnn.predictions,
                rnn.gender_loss, rnn.gender_acc,
                rnn.age_loss, rnn.age_acc],
                feed_dict)
            # time_str = datetime.datetime.now().isoformat()
            acc = tag_acc(batch_y, batch_l, tag_preds)
            print("{}\t{}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}".format(
                data_id, step,
                loss, acc,
                l_g, acc_g, l_a, acc_a)
            )
            return acc

        def test_step(batch_x, batch_y, batch_l, data_id=3):
            feed_dict = {
              rnn.input_x: batch_x,
              rnn.input_y: batch_y,
              rnn.input_l: batch_l,
              adv_lambda: 0.,
              rnn.dropout_keep_prob: 1.
            }
            step, loss, tag_preds = sess.run(
                [global_step,
                rnn.loss, rnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            ''' data_id: F:3 M:4 O:5 U:6 '''
            acc = tag_acc(batch_y, batch_l, tag_preds)
            print("{}\t{}\t{:g}\t{:g}".format(
                data_id, step,
                loss, acc)
            )
            return acc


        #10-fold cross-validation
        CV_fold = 10
        xv_iter = tag_helpers.cross_validation_iter( data = [x_tune, y_tune, l_tune, g_tune, a_tune], fold = CV_fold)

        fold_dev_res = []
        fold_test_res = []
        fold_tune_dev_res = []
        fold_tune_test_res = []

        for cv in range(CV_fold):
            print("CV:{}".format(cv))

            x_tune, y_tune, l_tune, g_tune, a_tune, \
                x_dev, y_dev, l_dev, g_dev, a_dev, \
                x_test, y_test, l_test, g_test, a_test \
                = xv_iter.fetch_next()
            
            #count & split
            F, M, O, U = 0, 0, 0, 0
            # x_f_dev, y_f_dev, l_f_dev = [], [], []
            x_f_test, y_f_test, l_f_test = [], [], []
            
            # x_m_dev, y_m_dev, l_m_dev = [], [], []
            x_m_test, y_m_test, l_m_test = [], [], []
            
            # x_o_dev, y_o_dev, l_o_dev = [], [], []
            x_o_test, y_o_test, l_o_test = [], [], []
            
            # x_u_dev, y_u_dev, l_u_dev = [], [], []
            x_u_test, y_u_test, l_u_test = [], [], []
            
            for i in range( len(x_test) ):
                if a_test[i][0] == 1:
                    O += 1
                    x_o_test.append( x_test[i] )
                    y_o_test.append( y_test[i] )
                    l_o_test.append( l_test[i] )
                else:
                    U += 1
                    x_u_test.append( x_test[i] )
                    y_u_test.append( y_test[i] )
                    l_u_test.append( l_test[i] )

                if g_test[i][0] == 1:
                    F += 1
                    x_f_test.append( x_test[i] )
                    y_f_test.append( y_test[i] )
                    l_f_test.append( l_test[i] )
                else:
                    M += 1
                    x_m_test.append( x_test[i] )
                    y_m_test.append( y_test[i] )
                    l_m_test.append( l_test[i] )

            print("In fold count -f:m = {}:{} \t\t-O:U = {}:{}.".format(F, M, O, U))


            sess.run(tf.global_variables_initializer())
            # sess.run(rnn.emb_W.assign(w_embs))

            # Training loop. For each batch...
            print("phase 1: training tagger by web_eng")
            data_size = len(x_train)
            best_dev_score = 0.
            best_test_score = [0.] * 4

            training_batch_iter = tag_helpers.batch_iter(
                data=[x_train, y_train, l_train],
                batch_size=FLAGS.batch_size,
                )
            training_learning_rate = FLAGS.learning_rate

            train_epochs = FLAGS.num_epochs
            for _ in range(train_epochs * data_size / FLAGS.batch_size):
                batch_x, batch_y, batch_l = training_batch_iter.next_full_batch()
                tag_train_step( batch_x, batch_y, batch_l, optimizer_n, lr=training_learning_rate )
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    acc = dev_step( x_dev, y_dev, l_dev, g_dev, a_dev, 1)
                    acc_f = test_step(x_f_test, y_f_test, l_f_test, 3)
                    acc_m = test_step(x_m_test, y_m_test, l_m_test, 4)
                    acc_o = test_step(x_o_test, y_o_test, l_o_test, 5)
                    acc_u = test_step(x_u_test, y_u_test, l_u_test, 6)
                    if acc > best_dev_score:
                        best_dev_score = acc
                        best_test_score = [acc_f, acc_m, acc_o, acc_u]
            

            print("Phase 2: tuning the tagger")
            tune_data_size = len(x_tune)
            best_tune_dev_score = 0.
            best_tune_test_score = [0.] * 4

            tune_batch_iter = tag_helpers.batch_iter(
                data=[x_tune, y_tune, l_tune, g_tune, a_tune],
                batch_size=FLAGS.batch_size,
                )
            tune_epocha = FLAGS.num_tune_epochs
            for _ in range(tune_epocha * tune_data_size / FLAGS.batch_size):
                batch_x, batch_y, batch_l, batch_g, batch_a = tune_batch_iter.next_full_batch()

                tag_tune_step(batch_x, batch_y, batch_l, batch_g, batch_a, \
                    optimizer_d, adv_lamb = 1e-3, lr = training_learning_rate)
                
                # tag_train_step( batch_x, batch_y, batch_l, optimizer_n, lr=training_learning_rate )

                tag_tune_step(batch_x, batch_y, batch_l, batch_g, batch_a, \
                    optimizer_g, adv_lamb = 1e-4, lr = training_learning_rate)

                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    acc = dev_step( x_dev, y_dev, l_dev, g_dev, a_dev, 1)
                    acc_f = test_step(x_f_test, y_f_test, l_f_test, 3)
                    acc_m = test_step(x_m_test, y_m_test, l_m_test, 4)
                    acc_o = test_step(x_o_test, y_o_test, l_o_test, 5)
                    acc_u = test_step(x_u_test, y_u_test, l_u_test, 6)
                    if acc > best_tune_dev_score:
                        best_tune_dev_score = acc
                        best_tune_test_score = [acc_f, acc_m, acc_o, acc_u]

            print("Phase 1 best dev acc {} with test acc {}".format(best_dev_score, best_test_score))
            print("Phase 2 best dev acc {} with test acc {}".format(best_tune_dev_score, best_tune_test_score))

            fold_dev_res.append( best_dev_score )
            fold_test_res.append( best_test_score )
            fold_tune_dev_res.append( best_tune_dev_score )
            fold_tune_test_res.append( best_tune_test_score )

print("Phase 1 dev score {} with {}".format(np.average(fold_dev_res, axis = 0), fold_dev_res ) )
print("Phase 1 test score {} with {}".format(np.average(fold_test_res, axis = 0), fold_test_res ) )
print("Phase 2 dev score {} with {}".format(np.average(fold_tune_dev_res, axis = 0), fold_tune_dev_res ) )
print("Phase 2 test score {} with {}".format(np.average(fold_tune_test_res, axis = 0), fold_tune_test_res ) )
