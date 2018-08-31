import sys
import numpy as np
from tensorflow.contrib import learn
from nltk.tag.mapping import map_tag

def load_web_eng(filename = "", trans = False):
    lines = list( open(filename, "r").readlines() )
    lines = [ l.strip() for l in lines]

    doc = []
    tags = []
    sent_w = []
    sent_t = []
    for l in lines:
        if l == '':
            doc.append(sent_w)
            tags.append(sent_t)
            sent_w = []
            sent_t = []
        else:
            w, t = l.split('\t')
            if t != "-NONE-":
                sent_w.append( w.lower() )
                if trans:
                    sent_t.append( map_tag('en-ptb', 'universal', t) )
                else:
                    sent_t.append( t )
    return doc, tags


server = "/home/yitongl4/"

def load_trustpilots():
	all_sents = []
	all_tags = []
	all_genders = []
	all_ages = []
	filenames = ["en.O45-UKC1_WORST-F.data.TT.tagged.gold",
		"en.O45-UKC1_WORST-M.data.TT.tagged.gold",
		"en.O45-UKH2_SOSO-F.data.TT.tagged.gold",
		"en.O45-UKH2_SOSO-M.data.TT.tagged.gold",
		"en.O45-UKN0_BEST-F.data.TT.tagged.gold",
		"en.O45-UKN0_BEST-M.data.TT.tagged.gold",
		"en.U35-UKC1_WORST-F.data.TT.tagged.gold",
		"en.U35-UKC1_WORST-M.data.TT.tagged.gold",
		"en.U35-UKH2_SOSO-F.data.TT.tagged.gold",
		"en.U35-UKH2_SOSO-M.data.TT.tagged.gold",
		"en.U35-UKN0_BEST-F.data.TT.tagged.gold",
		"en.U35-UKN0_BEST-M.data.TT.tagged.gold"
		]
	for i, filename in enumerate(filenames):
		sents, tags = load_web_eng(server + "Data/tagging_age/data/en/" + filename)
		if i < 6: 
			ages = np.array( [[ 1, 0 ]] * len(sents) ) #over 45
		else:
			ages = np.array( [[ 0, 1 ]] * len(sents) ) #under 35
		if i % 2 == 0:
			genders = np.array( [[ 1, 0 ]] * len(sents) ) # F
		else:
			genders = np.array( [[ 0, 1 ]] * len(sents) ) # M

		all_sents.extend(sents)
		all_tags.extend(tags)
		all_genders.extend(genders)
		all_ages.extend(ages)
	return all_sents, all_tags, np.array(all_genders), np.array(all_ages)


def load_data():
    sents_train, tags_train = load_web_eng(filename = server + "Data/eng_web_tbk/data/web_eng_penn.txt", trans = True)
    print("Totally load training {} data".format( len(sents_train) ))
    length_train = np.array( [len(x) for x in sents_train] )
    max_sent_length = max(length_train)
    print("Max sentence Length: {} ".format(max_sent_length) )

    #Vocab
    vocab_processor = VOCAB_processor( max_sent_length )
    vocab_size = vocab_processor.fit( sents_train )
    print("Vocabulary Size: {:d}".format( vocab_size ))
    x_train = vocab_processor.transform( sents_train )

    #Tag
    tag_processor = TAG_processor( max_sent_length )
    tag_size = tag_processor.fit( tags_train )
    print("POSTAG Size: {:d}".format( tag_size ) )
    print tag_processor.vocab
    y_train = tag_processor.transform( tags_train )

    sents_tune, tags_tune, genders_tune, ages_tune = load_trustpilots()
    print("Totally load tuning {} data".format( len(sents_tune) ))
    length_tune = np.array( [len(x) for x in sents_tune] )
    x_tune = vocab_processor.transform( sents_tune )
    print("Totally {} *unk* with {} total tokens in tune data".format( vocab_processor.total_unk, vocab_processor.total_token ))
    y_tune = tag_processor.transform( tags_tune )
    # print (tags_tune[0])
    # print (y_tune[0])

    sents_dev_aave, tags_dev_aave = load_web_eng(filename = server + "Data/aavepos2015/aave.dev")
    print("Totally load {} aave dev data".format( len(sents_dev_aave) ))
    length_dev_aave = np.array( [len(x) for x in sents_dev_aave] )
    x_dev_aave = vocab_processor.transform( sents_dev_aave )
    print("Totally {} *unk* with {} total tokens in aave dev data".format( vocab_processor.total_unk, vocab_processor.total_token ))
    y_dev_aave = tag_processor.transform( tags_dev_aave )

    sents_test_aave, tags_test_aave = load_web_eng( filename = server + "Data/aavepos2015/aave.tweets.test")
    print("Totally load {} aave test data".format( len(sents_test_aave) ))
    length_test_aave = np.array( [len(x) for x in sents_test_aave])
    x_test_aave = vocab_processor.transform( sents_test_aave )
    print("Totally {} *unk* with {} total tokens in aave test data".format( vocab_processor.total_unk, vocab_processor.total_token ))
    y_test_aave = tag_processor.transform( tags_test_aave )


    
    return max_sent_length, vocab_size, tag_size, x_train, y_train, length_train, \
    	x_tune, y_tune, genders_tune, ages_tune, length_tune, \
        x_dev_aave, y_dev_aave, length_dev_aave, \
        x_test_aave, y_test_aave, length_test_aave



class VOCAB_processor(object):
    def __init__(self, max_leng):
        self.max_len = max_leng
        self.vocab = {"<PAD>" : 0, "<unk>": 1}
        self.reverse_vocab = {0 : "<PAD>", 1 : "<unk>"}
        self.vocab_size = 2
        self.total_unk = 0
        self.total_token = 0

    # x := [ sent ("str") * ]
    def fit(self, x, d = None):
        # if d != None:
        size = self.vocab_size
        for s in x:
            # s = s.split(" ")
            for w in s:
                # w not in d, labeled as <unk>
#                if d != None and d.get(w, None) == None:
                if d != None and w not in d:
                    self.vocab[w] = 1
                elif self.vocab.get(w, -1) == -1:
                    self.vocab[w] = size
                    self.reverse_vocab[size] = w
                    size += 1
        self.vocab_size = size
        return self.vocab_size

    def transform(self, x):
        self.total_unk = 0
        self.total_token = 0
        trans = []
        for s in x:
            # s = s.split(" ")
            array = []
            for w in s:
                if w in self.vocab:
                    array.append(self.vocab[w])
                else:
                    array.append(self.vocab[ "<unk>" ])
                    self.total_unk += 1
                self.total_token += 1
            array = array + [0] * ( self.max_len - len(s) )
            trans.append( array )
        return np.array( trans, dtype = "int32" )

class TAG_processor(object):
    def __init__(self, max_leng):
        self.max_len = max_leng
        self.vocab = {}
        self.vocab_size = 0

    # x := [ sent ("str") * ]
    def fit(self, x, d = None):
        # if d != None:
        size = self.vocab_size
        for s in x:
            # s = s.split(" ")
            for w in s:
                if self.vocab.get(w, -1) == -1:
                    self.vocab[w] = size
                    size += 1
        self.vocab_size = size
        return self.vocab_size

    def transform(self, x):
        trans = []
        for s in x:
            # s = s.split(" ")
            array = []
            for w in s:
                array.append(self.vocab[w])
            array = array + [0] * ( self.max_len - len(s) )
            trans.append( array )
        return np.array( trans, dtype = "int32" )


class batch_iter(object):
    #data := list of np.darray
    def __init__(self, data, batch_size, is_shuffle=True):
        assert( len(data) > 0 )
        self.data = data
        self.batch_size = batch_size
        self.data_size = len( data[0] )
        assert (self.data_size >= self.batch_size)

        self.index = self.data_size
        self.is_shuffle = is_shuffle

    def fetch_batch(self, start, end):
        batch_list = []
        for data in self.data:
            batch_list.append(data[start: end])
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


def data_split(x, y, z, d, e, split_fraction = 0.5):
    assert(len(x) == len(y) and len(y) == len(z) )
    assert(len(x) == len(d) and len(y) == len(e) )
    l = int( len(x) * split_fraction )
    return x[:l], y[:l], z[:l], d[:l], e[:l], \
    	x[l + 1:], y[l + 1:], z[l + 1:], d[l + 1:], e[l + 1:]


#this is a quick iter not for general usage
class cross_validation_iter(object):

    def __init__(self, data, fold = 10):
        for i in range(1, len(data)):
            assert( len(data[0]) == len(data[i]) )
        self.data = data
        self.fold = fold
        self.cv = 0

    def fetch_next(self):
        p_train = [ ]
        p_dev = [ ]
        p_test = [ ]
        for _ in range( len(self.data)):
            p_train.append( [ ] )
            p_dev.append( [ ] )
            p_test.append( [ ] )

        cv = self.cv
        for i in range( len(self.data[0]) ):
            if i % self.fold == cv:
                for k in range( len(self.data) ):
                    p_dev[k].append( self.data[k][i] )

            elif (i + 1) % self.fold == cv:
                for k in range( len(self.data) ):
                    p_test[k].append( self.data[k][i] )
                    
            else:
                for k in range( len(self.data) ):
                    p_train[k].append( self.data[k][i] )

        self.cv = (self.cv + 1) % 10

        return np.array(p_train[0]), np.array(p_train[1]), np.array(p_train[2]), np.array(p_train[3]), np.array(p_train[4]), \
            np.array(p_dev[0]), np.array(p_dev[1]), np.array(p_dev[2]), np.array(p_dev[3]), np.array(p_dev[4]), \
            np.array(p_test[0]), np.array(p_test[1]), np.array(p_test[2]), np.array(p_test[3]), np.array(p_test[4])

