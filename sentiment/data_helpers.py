import sys
import numpy as np
from tensorflow.contrib import learn
import cPickle
import re

from collections import Counter

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_text_and_label(filename):
    positive_examples = list(open(filename + '.pos', "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(filename + '.neg', "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]

    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


#embedding operation
def load_pretrained_embeddings(filename, reverse_vocab):
    examples = list(open(filename, "r").readlines())
    embs = {}
    for s in examples:
        s = s.strip().split(" ")
        w = s[0]
        a = np.array(s[1:], dtype=float)
        embs[w] = a
    w_embs = []
    emb_size = len(a)

    for i in range(len(reverse_vocab)):
        w = reverse_vocab[i]
        if w in embs:
            w_embs.append( embs[w] )
        else:
            w_embs.append( np.random.uniform(-0.25,0.25, emb_size) )
    return emb_size, np.array(w_embs)

def load_embedding_dict(filename):
    examples = list(open(filename, "r").readlines())
    embs = {}
    for s in examples:
        s = s.strip().split(" ")
        w = s[0]
        a = np.array(s[1:], dtype=float)
        embs[w] = a
    emb_size = len(a)
    return emb_size, embs

def trans_pretrained_embeddings(emb_size, d, reverse_vocab):
    w_embs = []
    w_embs.append( np.random.uniform(-0.001,0.001, emb_size) )
    w_embs.append( np.random.uniform(-0.25,0.25, emb_size) )

    for i in range(2, len(reverse_vocab)):
        w = reverse_vocab[i]
        # if i > 1:
        assert(w in d)
        w_embs.append( d[w] )
        # else:
            # w_embs.append( np.random.uniform(-0.25,0.25, emb_size) )
    return np.array(w_embs)

def load_bin_vec(filename, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, dim = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * dim
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, k=300):
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  


def load_data(source_filename, target_filename, reload=False):
    print("Loading data...")
    if reload:
        tmp = cPickle.load( open("data.p", "rb") )

        max_sent_length, vocab_size, label_size, \
            source_x, source_label, \
            target_x, target_label, \
            emb_size, w_embs = tmp

    else:
        source_text, source_label = load_text_and_label(source_filename)
        target_text, target_label = load_text_and_label(target_filename)

        #sent_length
        upper_doc = source_text + target_text
        max_sent_length = max([len(x.split(" ")) for x in upper_doc])
        if max_sent_length > 1000:
            max_sent_length = 1000
        print "Max sentence Length:" + str(max_sent_length)

        #Vocab
        vocab_processor = VOCAB_processor(max_sent_length)
        vocab_size = vocab_processor.fit(upper_doc)
        print("Vocabulary Size: {:d}".format( vocab_size ))
        source_x = vocab_processor.transform( source_text )
        target_x = vocab_processor.transform( target_text )

        label_size = 2 # binary

        # loading pre-trained word embs
        emb_size = 300
        w2v_d = load_bin_vec(
            filename="/home/yitong/Data/GoogleNews-vectors-negative300.bin",
            vocab=vocab_processor.vocab
            )
        add_unknown_words(w2v_d, vocab_processor.vocab)
        w_embs = trans_pretrained_embeddings(
            emb_size=emb_size,
            d=w2v_d,
            reverse_vocab=vocab_processor.reverse_vocab
        )
        print("Embedding vector Size: {:d}".format( emb_size ))

        # cPickle.dump(
        #     [max_sent_length, vocab_size, label_size, \
        #         source_x, source_label, \
        #         target_x, target_label, \
        #         emb_size, w_embs
        #     ],
        #     open("data.p", "wb")
        #     )
    
    print("Done!")
    return  max_sent_length, vocab_size, label_size, \
            source_x, source_label, \
            target_x, target_label, \
            emb_size, w_embs


#load source data, dev data, target data.
def load_data_s_d_t(source_filename, dev_filename, target_filename, reload=False):
    print("Loading data...")
    if reload:
        pass
        # tmp = cPickle.load( open("data.p", "rb") )

        # max_sent_length, vocab_size, label_size, \
        #     source_x, source_label, \
        #     target_x, target_label, \
            # emb_size, w_embs = tmp

    else:
        source_text, source_label = load_text_and_label(source_filename)
        dev_text, dev_label = load_text_and_label(dev_filename)
        target_text, target_label = load_text_and_label(target_filename)

        #sent_length
        upper_doc = source_text + dev_text + target_text
        max_sent_length = max([len(x.split(" ")) for x in upper_doc])
        if max_sent_length > 1000:
            max_sent_length = 1000
        print "Max sentence Length:" + str(max_sent_length)

        #Vocab
        vocab_processor = VOCAB_processor(max_sent_length)
        vocab_size = vocab_processor.fit(upper_doc)
        print("Vocabulary Size: {:d}".format( vocab_size ))
        source_x = vocab_processor.transform( source_text )
        dev_x = vocab_processor.transform( dev_text )
        target_x = vocab_processor.transform( target_text )

        label_size = 2 # binary

        # loading pre-trained word embs
        emb_size = 300
        w2v_d = load_bin_vec(
            # filename="/short/cp1/yl6962/data/GoogleNews-vectors-negative300.bin",
            filename="/home/yitong/Data/GoogleNews-vectors-negative300.bin",
            vocab=vocab_processor.vocab
            )
        add_unknown_words(w2v_d, vocab_processor.vocab)
        w_embs = trans_pretrained_embeddings(
            emb_size=emb_size,
            d=w2v_d,
            reverse_vocab=vocab_processor.reverse_vocab
        )
        print("Embedding vector Size: {:d}".format( emb_size ))

        # cPickle.dump(
        #     [max_sent_length, vocab_size, label_size, \
        #         source_x, source_label, \
        #         target_x, target_label, \
        #         emb_size, w_embs
        #     ],
        #     open("data.p", "wb")
        #     )
    
    print("Done!")
    return  max_sent_length, vocab_size, label_size, \
            source_x, source_label, \
            dev_x, dev_label, \
            target_x, target_label, \
            emb_size, w_embs



class VOCAB_processor(object):
    def __init__(self, max_leng):
        self.max_len = max_leng
        self.vocab = {"<PAD>" : 0, "<unk>": 1}
        self.reverse_vocab = {0 : "<PAD>", 1 : "<unk>"}
        self.vocab_size = 2

    # x := [ sent ("str") * ]
    def fit(self, x, d = None):
        # if d != None:
        size = self.vocab_size
        for s in x:
            s = s.split(" ")
            for w in s:
                # w not in d, labeled as <unk>
                if d != None and d.get(w, None) == None:
                    self.vocab[w] = 1
                elif self.vocab.get(w, -1) == -1:
                    self.vocab[w] = size
                    self.reverse_vocab[size] = w
                    size += 1
        self.vocab_size = size
        return self.vocab_size

    def transform(self, x):
        trans = []
        for s in x:
            s = s.split(" ")
            array = []
            for i, w in enumerate(s):
                if i >= self.max_len:
                    break
                array.append(self.vocab[w])
            array = array + [0] * ( self.max_len - len(s) )
            trans.append( array )
        return np.array( trans, dtype = "int32" )



class batch_iter(object):
    #data := list of np.darray
    def __init__(self, data, batch_size, is_shuffle=True, is_balance=False, bal_label_index = 0):
        assert( len(data) > 0 )
        self.data = data
        self.batch_size = batch_size
        self.data_size = len( data[0] )
        assert (self.data_size >= self.batch_size)

        self.index = self.data_size
        self.is_shuffle = is_shuffle

        if is_balance:
            self.bal_label_index = bal_label_index
            self.labels = Counter( np.argmax(self.data[bal_label_index], axis=1 ) )
            self.max_cate_num = (batch_size + len(self.labels) - 1) / len(self.labels)

    def fetch_batch(self, start, end):
        batch_list = []
        for data in self.data:
            batch_list.append(data[start: end])
        return batch_list

    def fetch_batch_by_indices(self, indices):
        batch_list = []
        for data in self.data:
            tmp_data = []
            for i in indices:
                tmp_data.append( data[i] )
            batch_list.append( np.array(tmp_data) )
        return batch_list

    def shuffle(self):
        shuffle_indices = np.random.permutation( np.arange(self.data_size) )
        for i in range(len(self.data)):
            self.data[i] = (self.data[i])[shuffle_indices]

    def next_full_batch(self):
        if self.index < self.data_size - self.batch_size:
            self.index += self.batch_size
            return self.fetch_batch(self.index - self.batch_size, self.index)
        else:
            if self.is_shuffle:
                self.shuffle()
            self.index = self.batch_size
            return self.fetch_batch(0, self.batch_size)

    def next_balanced_label_batch(self):
        indices = []
        c = {}
        for i in self.labels:
            c[i] = 0

        while len(indices) < self.batch_size:
            if self.index >= self.data_size:
                if self.is_shuffle:
                    self.shuffle()
                self.index = 0
            label = np.argmax( self.data[self.bal_label_index][self.index] )
            if c[ label ] < self.max_cate_num:
                c[ label ] += 1
                indices.append(self.index)
            self.index += 1

        return self.fetch_batch_by_indices(indices)



# class data_split(object):
#     def __init__(self, data, CV_num):
#         shuffle_indices = np.random.permutation( np.arange(self.data_size) )
#         self.data = data[shuffle_indices]
#         self.cv = CV_num

#     def fetch_data(self, cv = 0):
#         



def load_glove_vec(filename, vocab, dim=50):
    """
    Loads 300x1 word vecs from Glove word2vec
    """

    file_vocab_size = 0
    word_vecs = {}
    with open(filename, "rb") as f:
        for line in iter(f.readline, ''):
            file_vocab_size += 1
            line = line.split()
            word = line[0]
            if word in vocab:
                word_vecs[word] = np.array( map(float, line[1:]), dtype='float32')
                assert( len(word_vecs[word]) == dim )

    return word_vecs

def index2onehot(indices):
    # print set(indices)
    max_index = max(indices) + 1
    ret = [ ]
    for ind in indices:
        tmp = [0] * max_index
        if ind != -1:
            tmp[ind] = 1
        # else:
        #     tmp = [ 1.0 / max_index ] * max_index
        ret.append(tmp)
    return np.array( ret )

def load_trustpilot(reload = False):
    print("Loading TrustPilot data...")

    if reload:
        pass
        # tmp = cPickle.load( open("data.p", "rb") )

        # max_sent_length, vocab_size, \
        #     text, location, genders, ratings
        #     source_x, source_label, \

    else:
        # filename = "../trustpilot/merge.en.random10k"
        filename = "../trustpilot/merge.en.downsample"
        examples = cPickle.load( open(filename, "rb") )
        print("Total No. of instances:{}".format(len(examples)))

        #sent_length
        locations = index2onehot( [ int(e[0]) for e in examples ] )
        genders = index2onehot( [ int(e[1]) for e in examples ] )
        ages = index2onehot( [ int(e[2]) for e in examples ] )
        text = [ e[3] for e in examples ]
        ratings = index2onehot( [ int(e[4]) for e in examples ] )

        upper_doc = text
        max_sent_length = max([len(x.split(" ")) for x in upper_doc]) 
        print "Max sentence Length:" + str(max_sent_length)
        #Trim max length of docs
        if max_sent_length > 256:
            max_sent_length = 256
            print "Trunct Max sentence Length:" + str(max_sent_length)

        #Vocab
        vocab_processor = VOCAB_processor(max_sent_length)
        vocab_size = vocab_processor.fit(upper_doc)
        print("Vocabulary Size: {:d}".format( vocab_size ))
        source_x = vocab_processor.transform( text )

        # loading pre-trained word embs
        # w2v_d = load_bin_vec(
        #     # filename="/short/cp1/yl6962/data/GoogleNews-vectors-negative300.bin",
        #     filename="/home/yitong/Data/GoogleNews-vectors-negative300.bin",
        #     vocab=vocab_processor.vocab
        #     )
        emb_size = 50
        w2v_d = load_glove_vec(
            filename = "/home/yitongl4/Data/glove.6B.50d.txt",
            vocab = vocab_processor.vocab,
            dim = emb_size
            )
        add_unknown_words(w2v_d, vocab_processor.vocab, k = emb_size)
        w_embs = trans_pretrained_embeddings(
            emb_size=emb_size,
            d=w2v_d,
            reverse_vocab=vocab_processor.reverse_vocab
        )
        print("Embedding vector Size: {:d}".format( emb_size ))
        # cPickle.dump(
        #     [max_sent_length, vocab_size, \
        #         text, locations, genders, ratings, \
        #         emb_size, w_embs
        #     ],
        #     open("data.p", "wb")
        #     )


    print("Done!")
    return  max_sent_length, vocab_size, \
            source_x, locations, genders, ages, ratings, \
            emb_size, w_embs


#TODO: X-validation pro-class
#x must be a list here
def data_split_train_dev_test(x, shuffle=False):
    if shuffle:
        shuffle_indices = np.random.permutation( np.arange(len(x[0])) )
        for i in range(len(x)):
            x[i] = (x[i])[shuffle_indices]

    assert( len(x) > 1 )
    data_length = len(x[0])
    l1 = data_length * 8 / 10
    l2 = data_length * 9 / 10
    ret_list = []
    for li in x:
        ret_list.append( li[:l1] )
    for li in x:
        ret_list.append( li[l1+1:l2] )
    for li in x:
        ret_list.append( li[l2+1:] )

    return ret_list


class X_validation_iter(object):
    def __init__(self, data, fold=10, is_shuffle=True, use_dev=True):
        self.data = data
        self.data_size = len( self.data[0] )
        for d in self.data:
            assert( len(d) == self.data_size )
        self.fold = fold
        self.cur_test_cv = 0
        self.cur_dev_cv = 0
        self.use_dev = use_dev
        self.test_scores = []
        if self.use_dev:
            self.cur_dev_cv = -1
            self.dev_scores = []

        self.is_shuffle = is_shuffle
        if self.is_shuffle:
            self.shuffle()

    # split data by their IDs
    def next_fold(self):
        ret_list = []
        self.cur_test_cv = (self.cur_test_cv + 1) % self.fold
        self.cur_dev_cv = (self.cur_dev_cv + 1) % self.fold            

        # training data
        for d in self.data:
            tmp = []
            for i in range(self.data_size):
                if i % self.fold != self.cur_test_cv and i % self.fold != self.cur_dev_cv:
                    tmp.append( d[i] )
            ret_list.append( np.array(tmp) )

        # dev data
        if self.use_dev:
            for d in self.data:
                tmp = []
                for i in range(self.data_size):
                    if i % self.fold == self.cur_dev_cv:
                        tmp.append( d[i] )
                ret_list.append( np.array(tmp) )

        # test data
        for d in self.data:
            tmp = []
            for i in range(self.data_size):
                if i % self.fold == self.cur_test_cv:
                    tmp.append( d[i] )
            ret_list.append( np.array(tmp) )

        return ret_list

    
    def shuffle(self):
        shuffle_indices = np.random.permutation( np.arange(self.data_size) )
        for i in range(len(self.data)):
            self.data[i] = (self.data[i])[shuffle_indices]
