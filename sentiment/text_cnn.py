import tensorflow as tf
import numpy as np


class TextCNN(object):

    def cnn(self, scope_number, embedded_chars_expanded, sequence_length, embedding_size, filter_sizes, num_filters):
        with tf.variable_scope("cnn%s" % scope_number) as scope:
        # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.variable_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.get_variable(
                        name="W",
                        shape=filter_shape,
                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
                        )
                    b = tf.get_variable(
                        name="b",
                        shape=[num_filters],
                        initializer=tf.constant_initializer(0.1)
                        )
                    conv = tf.nn.conv2d(
                        embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            pooled = tf.concat(pooled_outputs, 3)
            num_filters_total = num_filters * len(filter_sizes)
            return tf.reshape(pooled, [-1, num_filters_total])

        
    def wx_plus_b(self, scope_name, x, size):
        with tf.variable_scope("full_connect_%s" % scope_name) as scope:
            W = tf.get_variable(
                name="W",
                shape=size,
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(
                name="b",
                shape=[size[1]],
                initializer=tf.constant_initializer(0.1, )
                )
            y = tf.nn.xw_plus_b(x, W, b, name="hidden")
            return y

        
    def gaussian_noise_layer(self, input_layer, std = 0.001):
        noise = tf.random_normal(shape=tf.shape(input_layer) , mean=0.0, stddev=std, dtype=tf.float32)
        return input_layer + noise

    #main enter
    def __init__(self, sequence_length, vocab_size,
            embedding_size, filter_sizes, num_filters,
            num_ratings, num_locations, num_genders, num_ages,
            hidden_size, l2_reg_lambda=0.0):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_rating = tf.placeholder(tf.float32, [None, num_ratings], name="input_rating_truth")
        self.input_location = tf.placeholder(tf.float32, [None, num_locations], name="input_location_truth")
        self.input_gender = tf.placeholder(tf.float32, [None, num_genders], name="input_gender_truth")
        self.input_age = tf.placeholder(tf.float32, [None, num_ages], name="input_age_truth")
        
        l2_loss = tf.constant(0.0)

        with tf.variable_scope("embedding"):
            self.emb_W = tf.get_variable(
                name="lookup_emb",
                shape=[vocab_size, embedding_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
                trainable=False
                )
            embedded_chars = tf.nn.embedding_lookup(self.emb_W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
            self.embedded_chars_expanded = self.gaussian_noise_layer( self.embedded_chars_expanded )
        
        #hidden 0 = cnn+pooling output
        self.pub_h_pool = self.cnn("shared", self.embedded_chars_expanded, sequence_length, embedding_size, filter_sizes, num_filters)

        # Add dropout
        with tf.name_scope("dropout"):
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.h_drop = tf.nn.dropout(self.pub_h_pool, self.dropout_keep_prob)
        

        with tf.variable_scope("rating"):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=self.h_drop,
                size=[num_filters * len(filter_sizes), hidden_size]
            )
            h1 = tf.nn.relu(h1, name="relu")
            self.rating_scores = self.wx_plus_b(
                scope_name='score',
                x=h1,
                size=[hidden_size, num_ratings]
                )
            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.rating_scores,
                    labels=self.input_rating
                    )
                self.rating_loss = tf.reduce_mean(losses, name="task_loss")

            with tf.name_scope("accuracy"):
                self.rating_pred = tf.argmax(self.rating_scores, 1, name="predictions")
                cor_pred = tf.cast(
                    tf.equal( self.rating_pred, tf.argmax(self.input_rating, 1) ),
                    "float"
                    )
                # self.rating_cor_pred = tf.reduce_sum( cor_pred, name="prediction_number" )
                self.rating_accuracy = tf.reduce_mean( cor_pred, name="accuracy" )
        
        
        with tf.variable_scope("location"):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=self.h_drop,
                size=[num_filters * len(filter_sizes), hidden_size]
            )
            h1 = tf.nn.relu(h1, name="relu")
            self.location_scores = self.wx_plus_b(
                scope_name="score",
                x=h1,
                size=[hidden_size, num_locations]
                )
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.location_scores,
                    labels=self.input_location
                    )
                self.location_loss = tf.reduce_mean(losses)
            with tf.name_scope("accuracy"):
                self.location_pred = tf.argmax(self.location_scores, 1, name="predictions")
                cor_pred = tf.cast(
                    tf.equal(self.location_pred, tf.argmax(self.input_location, 1) ),
                    "float"
                    )
                self.location_accuracy = tf.reduce_mean(cor_pred, name="acc")


        with tf.variable_scope("gender"):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=self.h_drop,
                size=[num_filters * len(filter_sizes), hidden_size]
            )
            h1 = tf.nn.relu(h1, name="relu")
            self.gender_score = self.wx_plus_b(
                scope_name="score",
                x=h1,
                size=[hidden_size, num_genders]
                )
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.gender_score,
                    labels=self.input_gender
                    )
                self.gender_loss = tf.reduce_mean( losses )
            with tf.name_scope("acc"):
                self.gender_pred = tf.argmax( self.gender_score, 1, name='predictions')
                cor_pred = tf.cast(
                    tf.equal(self.gender_pred, tf.argmax(self.input_gender, 1) ),
                    "float"
                    )
                self.gender_accuracy = tf.reduce_mean(cor_pred, name='acc')


        with tf.variable_scope("age"):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=self.h_drop,
                size=[num_filters * len(filter_sizes), hidden_size]
            )
            h1 = tf.nn.relu(h1, name="relu")
            self.age_score = self.wx_plus_b(
                scope_name="score",
                x=h1,
                size=[hidden_size, num_ages]
                )
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.age_score,
                    labels=self.input_age
                    )
                self.age_loss = tf.reduce_mean( losses )
            with tf.name_scope("acc"):
                self.age_pred = tf.argmax( self.age_score, 1, name='predictions')
                cor_pred = tf.cast(
                    tf.equal(self.age_pred, tf.argmax(self.input_age, 1) ),
                    "float"
                    )
                self.age_accuracy = tf.reduce_mean(cor_pred, name='acc')


        with tf.variable_scope("l_attacker"):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=self.h_drop,
                size=[num_filters * len(filter_sizes), hidden_size]
            )
            h1 = tf.nn.relu(h1, name="relu")
            self.location_attacker_scores = self.wx_plus_b(
                scope_name="score",
                x=h1,
                size=[hidden_size, num_locations]
                )
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.location_attacker_scores,
                    labels=self.input_location
                    )
                self.location_attacker_loss = tf.reduce_mean(losses)
            with tf.name_scope("accuracy"):
                self.location_attacker_pred = tf.argmax(self.location_attacker_scores, 1, name="predictions")
                cor_pred = tf.cast(
                    tf.equal(self.location_attacker_pred, tf.argmax(self.input_location, 1) ),
                    "float"
                    )
                self.location_attacker_accuracy = tf.reduce_mean(cor_pred, name="acc")


        with tf.variable_scope("g_attacker"):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=self.h_drop,
                size=[num_filters * len(filter_sizes), hidden_size]
            )
            h1 = tf.nn.relu(h1, name="relu")
            self.gender_attacker_score = self.wx_plus_b(
                scope_name="score",
                x=h1,
                size=[hidden_size, num_genders]
                )
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.gender_attacker_score,
                    labels=self.input_gender
                    )
                self.gender_attacker_loss = tf.reduce_mean( losses )
            with tf.name_scope("acc"):
                self.gender_attacker_pred = tf.argmax( self.gender_attacker_score, 1, name='predictions')
                cor_pred = tf.cast(
                    tf.equal(self.gender_attacker_pred, tf.argmax(self.input_gender, 1) ),
                    "float"
                    )
                self.gender_accuracy = tf.reduce_mean(cor_pred, name='acc')


        with tf.variable_scope("a_attacker"):
            h1 = self.wx_plus_b(
                scope_name="h1",
                x=self.h_drop,
                size=[num_filters * len(filter_sizes), hidden_size]
            )
            h1 = tf.nn.relu(h1, name="relu")
            self.age_attacker_score = self.wx_plus_b(
                scope_name="score",
                x=h1,
                size=[hidden_size, num_ages]
                )
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.age_attacker_score,
                    labels=self.input_age
                    )
                self.age_attacker_loss = tf.reduce_mean( losses )
            with tf.name_scope("acc"):
                self.age_attacker_pred = tf.argmax( self.age_attacker_score, 1, name='predictions')
                cor_pred = tf.cast(
                    tf.equal(self.age_attacker_pred, tf.argmax(self.input_age, 1) ),
                    "float"
                    )
                self.age_attacker_accuracy = tf.reduce_mean(cor_pred, name='acc')
