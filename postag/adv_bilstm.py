import tensorflow as tf
import numpy as np



class adv_bilstm(object):

    def __init__(self, seq_length, vocab_size, embedding_size, rnn_cell_size, tag_size,\
        gender_num = 2, age_num = 2):

        self.input_x = tf.placeholder(tf.int32, shape=[None, seq_length], name="input_x")
        self.input_l = tf.placeholder(tf.int32, [None], name = "input_words")
        self.input_y = tf.placeholder(tf.int32, shape=[None, seq_length], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.variable_scope("embedding"):
            self.emb_W = tf.get_variable(
                name="lookup_emb",
                shape=[vocab_size, embedding_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0),
                trainable=True
                )
            embedded_chars = tf.nn.embedding_lookup(self.emb_W, self.input_x)

        with tf.variable_scope("rnn"):
            fw_cell = tf.contrib.rnn.GRUCell(rnn_cell_size)
            bw_cell = tf.contrib.rnn.GRUCell(rnn_cell_size)
            outputs, final_state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = fw_cell,
                cell_bw = bw_cell,
                inputs = embedded_chars,
                sequence_length = self.input_l,
                dtype=tf.float32
                )

        with tf.variable_scope("output"):
            w = tf.get_variable(
                name="softmax_w",
                shape=[rnn_cell_size * 2, tag_size],
                initializer=tf.truncated_normal_initializer(stddev=0.05),
                dtype=tf.float32
                )
            b = tf.get_variable(
                name="softmax_b",
                shape=[tag_size],
                initializer=tf.constant_initializer(value=0.),
                dtype=tf.float32
                )

            output = tf.concat([outputs[0], outputs[1]], axis = 2, name = "concat_output")
            output = tf.nn.dropout(output, self.dropout_keep_prob)
            # output_score = tf.reshape(
            #     tf.nn.xw_plus_b(
            #         x=tf.reshape(
            #             output,
            #             shape = [-1, rnn_cell_size * 2]
            #             ),
            #         weights=w,
            #         biases=b
            #         ),
            #     shape = [-1, seq_length, tag_size],
            #     name="score"
            #     )

            output_score = tf.nn.bias_add( tf.tensordot(output, w, axes=[[2], [0]]), b )

        seq_mask = tf.sequence_mask(
                    lengths=self.input_l,
                    maxlen=seq_length,
                    dtype=tf.float32,
                    name="loss_masks"
                    )
        with tf.variable_scope("loss"):
            self.loss = tf.contrib.seq2seq.sequence_loss(
                logits=output_score,
                targets=self.input_y,
                weights=seq_mask,
                # average_across_batch=False,
                name="total_tag_loss"
                )

            self.predictions = tf.argmax(output_score, axis=2)
            #accuracy should be calced outside considering the variations of sentence length

        final_output = tf.concat( [final_state[0], final_state[1]], axis = 1, name = "concat_final_output")
        final_output = tf.nn.dropout(final_output, self.dropout_keep_prob)
        
        with tf.variable_scope("gender_discriminator"):
            self.input_g = tf.placeholder(tf.int32, shape=[None, gender_num], name="input_gender")
            w = tf.get_variable(
                name="softmax_w",
                shape=[rnn_cell_size * 2, gender_num],
                initializer=tf.truncated_normal_initializer(stddev=0.05),
                dtype=tf.float32
                )
            b = tf.get_variable(
                name="softmax_b",
                shape=[gender_num],
                initializer=tf.constant_initializer(value=0.),
                dtype=tf.float32
                )
            gender_score = tf.nn.bias_add( tf.tensordot(final_output, w, axes = [[1], [0]]), b)
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=gender_score,
                    labels=self.input_g
                    )
                self.gender_loss = tf.reduce_mean(losses, name="loss")

                self.gender_pred = tf.argmax(gender_score, 1, name="predictions")
                cor_pred = tf.cast( tf.equal( self.gender_pred, tf.argmax(self.input_g, 1) ), "float" )
                self.gender_acc = tf.reduce_mean( cor_pred, name="accuracy" )


        with tf.variable_scope("age_discriminator"):
            self.input_a = tf.placeholder(tf.int32, shape=[None, age_num], name="input_age")
            w = tf.get_variable(
                name="softmax_w",
                shape=[rnn_cell_size * 2, age_num],
                initializer=tf.truncated_normal_initializer(stddev=0.05),
                dtype=tf.float32
                )
            b = tf.get_variable(
                name="softmax_b",
                shape=[age_num],
                initializer=tf.constant_initializer(value=0.),
                dtype=tf.float32
                )
            age_score = tf.nn.bias_add( tf.tensordot(final_output, w, axes = [[1], [0]]), b)
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=age_score,
                    labels=self.input_a
                    )
                self.age_loss = tf.reduce_mean(losses, name="loss")

                self.age_pred = tf.argmax(age_score, 1, name="predictions")
                cor_pred = tf.cast( tf.equal( self.age_pred, tf.argmax(self.input_a, 1) ), "float" )
                self.age_acc = tf.reduce_mean( cor_pred, name="accuracy" )


            # output_state = tf.concat( [final_state[0], final_state[1]], axis=1, name="concated_final_state")
            # output_langua_score = tf.nn.xw_plus_b(
            #     x=output_state,
            #     weights=w,
            #     biases=b,
            #     name="language_score"
            #     )

            # langua_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #     labels=self.input_d,
            #     logits=output_langua_score,
            #     )

            # output_langua_score = tf.reshape(
            #     tf.nn.xw_plus_b(
            #         x=tf.reshape(
            #             output,
            #             shape = [-1, rnn_cell_size * 2]
            #             ),
            #         weights=w,
            #         biases=b
            #         ),
            #     shape = [-1, seq_length, language_size],
            #     name="language_score"
            #     )

            # langua_losses = tf.contrib.seq2seq.sequence_loss(
            #     logits=output_langua_score,
            #     targets=self.input_d,
            #     weights=seq_mask,
            #     name="linguage_loss",
            #     )

            # self.langua_loss = langua_losses
            # self.lang_preds = tf.argmax(output_langua_score, axis=2)
            # cor_lingua_num = tf.cast( tf.equal(lang_preds, tf.cast(self.input_d, tf.int64) ), tf.float32, name="cor_ling_num")
            # self.acc = tf.reduce_mean(cor_lingua_num, name="ling_acc")
