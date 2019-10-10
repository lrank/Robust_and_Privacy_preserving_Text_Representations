#! /usr/bin/env python

import os
import time
import datetime
import cPickle

import tensorflow as tf
import numpy as np

import data_helpers
from text_cnn import TextCNN

from tensorflow.contrib import learn

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 50, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate alpha")
tf.flags.DEFINE_float("lr_lambda", 1e-3, "lr lambda")

tf.flags.DEFINE_string("adv_attribute", "age", "age, gender, location, or all")

#  parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200 --> 100 by lrank)")
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

max_doc_length, vocab_size, \
    text_x, locations, genders, ages, ratings, \
    emb_size, w_embs = data_helpers.load_trustpilot()


FLAGS.embedding_dim = emb_size
FLAGS.batch_size
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
        cnn = TextCNN(
            sequence_length=text_x.shape[1],
            vocab_size=vocab_size,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
            num_filters=FLAGS.num_filters,
            num_ratings=ratings.shape[1],
            num_locations=locations.shape[1],
            num_genders=genders.shape[1],
            num_ages=ages.shape[1],
            hidden_size=300,
            l2_reg_lambda=FLAGS.l2_reg_lambda
            )

        # Define Training procedure
        learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        adv_lambda = tf.placeholder(tf.float32, shape=[], name="adversarial_lambda")

        global_step = tf.Variable(0, name="global_step", trainable=False)
        all_var_list = tf.trainable_variables()

        optimizer_n = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.rating_loss,
#                cnn.location_loss,
                global_step=global_step
                )

        var_loca = [var for var in all_var_list if 'location' in var.name]
        # print var_loca
        var_gend = [var for var in all_var_list if 'gender' in var.name]
        # print var_gend
        var_age = [var for var in all_var_list if 'age' in var.name]
        # print var_gend
        assert( len(var_loca) == 4 and len(var_gend) == 4 and len(var_age) == 4 )

        if FLAGS.adv_attribute == "age":
            var_d = var_age
            disc_loss = cnn.age_loss
        elif FLAGS.adv_attribute == "gender":
            var_d = var_gend
            disc_loss = cnn.gender_loss
        elif FLAGS.adv_attribute == "location":
            var_d = var_loca
            disc_loss = cnn.location_loss
        elif FLAGS.adv_attribute == "all":
            var_d = var_loca + var_gend + var_age
            disc_loss = cnn.location_loss + cnn.gender_loss + cnn.age_loss  
        else:
            assert(False)

        optimizer_d = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                adv_lambda * disc_loss,
                var_list=var_d
                )

        #define attackers opts
        var_attack_l = [var for var in all_var_list if 'l_attacker' in var.name]
        optimizer_attack_l = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.location_attacker_loss,
                var_list=var_attack_l,
                global_step=global_step
                )

        var_attack_g = [var for var in all_var_list if 'g_attacker' in var.name]
        optimizer_attack_g = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.gender_attacker_loss,
                var_list=var_attack_g,
                global_step=global_step
                )

        var_attack_a = [var for var in all_var_list if 'a_attacker' in var.name]
        optimizer_attack_a = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.age_attacker_loss,
                var_list=var_attack_a,
                global_step=global_step
                )
        assert( len(var_attack_l) == 4 and len(var_attack_g) == 4 and len(var_attack_a) == 4 )

        #representation opt
        var_g = [var for var in all_var_list if var not in (var_loca + var_gend + var_age + var_attack_l + var_attack_g + var_attack_a)]
        optimizer_g = tf.train.AdamOptimizer(
            learning_rate=learning_rate
            ).minimize(
                cnn.rating_loss - adv_lambda * disc_loss,
                var_list=var_g,
                global_step=global_step
                )

        def incomp_acc(logit, label):
            c_cor = 0.
            c_total = 0.
            gt = np.argmax(label, axis=1)
            for i in range( len( logit )):
                if gt[i] == 0 and np.abs(label[i][gt[i]]) < 1e-5:
                    continue
                c_total += 1.
                if gt[i] == logit[i]:
                    c_cor += 1.
            if np.abs(c_total) < 1e-5:
                return -1
            return c_cor / c_total

        def dev_step(batch_x, batch_loc, batch_gen, batch_age, batch_rat, data_id=1):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: batch_x,
              cnn.input_rating: batch_rat,
              cnn.input_location: batch_loc,
              cnn.input_gender: batch_gen,
              cnn.input_age: batch_age,
              adv_lambda: 0.,
              cnn.dropout_keep_prob: 1.
            }
            step, l_rat, a_rat, l_loc, p_loc, l_gen, p_gen, l_age, p_age = sess.run(
                [global_step,
                cnn.rating_loss, cnn.rating_accuracy,
                cnn.location_loss, cnn.location_pred,
                cnn.gender_loss, cnn.gender_pred,
                cnn.age_loss, cnn.age_pred],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            ''' data_id: train:0 dev:1 test:2 '''
            a_loc = incomp_acc(p_loc, batch_loc)
            a_gen = incomp_acc(p_gen, batch_gen)
            a_age = incomp_acc(p_age, batch_age)
            print("{}\t{}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}".format(
                data_id, step,
                l_rat, a_rat,
                l_loc, a_loc,
                l_gen, a_gen,
                l_age, a_age )
            )
            return a_rat, a_loc, a_gen, a_age

        def train_step(batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer, adv_lam=0, lr = 1e-4):
            """1
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: batch_x,
              cnn.input_rating: batch_rat,
              cnn.input_location: batch_loc,
              cnn.input_gender: batch_gen,
              cnn.input_age: batch_age,
              learning_rate: lr,
              adv_lambda: adv_lam,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, l_rat, a_rat, l_loc, p_loc, l_gen, p_gen, l_age, p_age = sess.run(
                [optimizer, global_step,
                cnn.rating_loss, cnn.rating_accuracy,
                cnn.location_loss, cnn.location_pred,
                cnn.gender_loss, cnn.gender_pred,
                cnn.age_loss, cnn.age_pred],
                feed_dict)
            
            time_str = datetime.datetime.now().isoformat()
            print("0\t{}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}".format(
                step,
                l_rat, a_rat,
                l_loc, incomp_acc(p_loc, batch_loc),
                l_gen, incomp_acc(p_gen, batch_gen),
                l_age, incomp_acc(p_age, batch_age)
                )
            )

        def train_attacker_step(batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer, adv_lam=0, lr = 1e-4):
            """1
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: batch_x,
              cnn.input_rating: batch_rat,
              cnn.input_location: batch_loc,
              cnn.input_gender: batch_gen,
              cnn.input_age: batch_age,
              learning_rate: lr,
              adv_lambda: adv_lam,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }

            _, step, l_loc, p_loc = sess.run(
                [optimizer_attack_l, global_step,
                cnn.location_attacker_loss, cnn.location_attacker_pred],
                feed_dict)

            _, step, l_gen, p_gen = sess.run(
                [optimizer_attack_g, global_step,
                cnn.gender_attacker_loss, cnn.gender_attacker_pred],
                feed_dict)

            _, step, l_age, p_age = sess.run(
                [optimizer_attack_a, global_step,
                cnn.age_attacker_loss, cnn.age_attacker_pred],
                feed_dict)
            
            time_str = datetime.datetime.now().isoformat()
            print("0\t{}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}".format(
                step,
                l_rat, a_rat,
                l_loc, incomp_acc(p_loc, batch_loc),
                l_gen, incomp_acc(p_gen, batch_gen),
                l_age, incomp_acc(p_age, batch_age)
                )
            )

        def train_attacker_step(batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer, adv_lam=0, lr = 1e-4):
            """1
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: batch_x,
              cnn.input_rating: batch_rat,
              cnn.input_location: batch_loc,
              cnn.input_gender: batch_gen,
              cnn.input_age: batch_age,
              learning_rate: lr,
              adv_lambda: adv_lam,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, l_rat, a_rat, l_loc, p_loc, l_gen, p_gen, l_age, p_age = sess.run(
                [optimizer, global_step,
                cnn.rating_loss, cnn.rating_accuracy,
                cnn.location_attacker_loss, cnn.location_attacker_pred,
                cnn.gender_attacker_loss, cnn.gender_attacker_pred,
                cnn.age_attacker_loss, cnn.age_attacker_pred],
                feed_dict)
            
            time_str = datetime.datetime.now().isoformat()
            print("0\t{}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}\t{:g}".format(
                step,
                l_rat, a_rat,
                l_loc, incomp_acc(p_loc, batch_loc),
                l_gen, incomp_acc(p_gen, batch_gen),
                l_age, incomp_acc(p_age, batch_age)
                )
            )


        #data_split
        x_train, loc_train, gen_train, age_train, rat_train, \
        x_dev, loc_dev, gen_dev, age_dev, rat_dev, \
        x_test, loc_test, gen_test, age_test, rat_test = data_helpers.data_split_train_dev_test(
            [text_x, locations, genders, ages, ratings],
            shuffle=True
            )

        sess.run(tf.global_variables_initializer())
        sess.run(cnn.emb_W.assign(w_embs))

        # Training loop. For each batch...
        data_size = len(x_train)
        best_dev_score = 0.
        bset_test_score = 0.
        training_batch_iter = data_helpers.batch_iter(
            data=[x_train, loc_train, gen_train, age_train, rat_train],
            batch_size=FLAGS.batch_size,
            is_balance=True,
            bal_label_index=4,
            )
        training_learning_rate = FLAGS.learning_rate        

        
        # for _ in range(FLAGS.num_epochs * data_size / FLAGS.batch_size):
        for _ in range(100):

            current_step = tf.train.global_step(sess, global_step)
            lr_lamb = (current_step / 100) / 1000.0
            
            if lr_lamb > FLAGS.lr_lambda:
                lr_lamb = FLAGS.lr_lambda

            batch_x, batch_loc, batch_gen, batch_age, batch_rat = training_batch_iter.next_balanced_label_batch()

            train_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_d, adv_lam=lr_lamb, lr=training_learning_rate)
            train_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_g, adv_lam=lr_lamb, lr=training_learning_rate)

            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                # print("\nEvaluation:")
                print("3\t{:g}".format(lr_lamb))
                acc, _, _, _ = dev_step( x_dev, loc_dev, gen_dev, age_dev, rat_dev, 1)
                test_score = dev_step( x_test, loc_test, gen_test, age_test, rat_test, 2)
                if acc > best_dev_score:
                    best_dev_score = acc
                    best_test_score = test_score

        print("training attack")
        for _ in range(FLAGS.num_epochs * data_size / FLAGS.batch_size):
            batch_x, batch_loc, batch_gen, batch_age, batch_rat = training_batch_iter.next_balanced_label_batch()
            train_attacker_step( batch_x, batch_loc, batch_gen, batch_age, batch_rat, optimizer_attack_l, adv_lam=lr_lamb, lr=training_learning_rate * 0.1 )

            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                test_score, a_l, a_g, a_a = dev_attacker_step( x_test, loc_test, gen_test, age_test, rat_test, 2)

print best_dev_score, best_test_score
