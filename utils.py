"""
VOCAB CLASS
"""
from collections import defaultdict, Counter
import regex as re
import html
import unicodedata
import numpy as np

class Lang:
    def __init__(self, name, _min_w_count=2):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        self.VOCAB_SIZE = 0
        
        # Special tokens
        self.SOS = '<S>'
        self.EOS = '</S>'
        self.UNK = '<UNK>'
        self.PAD = '<PAD>'
        self.EOL = '<EOL>' # Newline token
        self.iPAD = 0
        self.iSOS = 1
        self.iEOS = 2
        self.iUNK = 3
        self.iEOL = 4
        
        self.min_count = _min_w_count
    
    def eos_id(self):
        return self.iEOS
    
    def bos_id(self):
        return self.iSOS
    
    def normalizeSentence(self, s, escape_html=True):
        """Normalizes a space delimited sentence 'line'
        
        Arguments:
        s -- string representing a sentence
        
        Returns:
        w_arr -- array containing the normalized words of the normalized sentence
        """
        
        # Turn a Unicode string to plain ASCII, thanks to
        # http://stackoverflow.com/a/518232/2809427
        def unicodeToAscii(s):
            return ''.join(
                c for c in unicodedata.normalize('NFD', s)
                if unicodedata.category(c) != 'Mn'
            )
        
        # Escape HTML
        if escape_html:
            s = html.unescape(s)
        #line = '<s> ' + line.strip() + ' </s>'
        
        #allowed_punct = ".!?,-"
        #s = unicodeToAscii(s.lower().strip())
        #s = re.sub(r"([%s])" % allowed_punct, r" \1 ", s)
        #s = re.sub(r"[^a-zA-Z0-9%s]+" % allowed_punct, r" ", s)
        
        """
        Alternative
        To keep hyphens without spaces intact
        """
        
        
        allowed_punct = ".!?,=<>"
        s = unicodeToAscii(s.lower().strip())
        s = re.sub(r"([%s])" % allowed_punct, r" \1 ", s)
        
        # Convert Newlines to <EOL>
        s = re.sub(r"[\r\n]+", r" %s " % self.EOL, s)
        
        s = re.sub(r"[^a-zA-Z0-9%s]+" % allowed_punct, r" ", s)
    
        s = s.strip()
        
        # Tokenize + Strip + Split
        tokenized_l = re.split('[ ]+', s)
        
        return tokenized_l
    def encodeSentence(self, l):
        l = self.normalizeSentence(l)
        el = []
        for w in l:
            if w in self.word2index:
                el.append(self.word2index[w])
            else:
                el.append(self.iUNK)
        return el
    
    def EncodeAsIds(self, l):
        return self.encodeSentence(l)
    
    def sample_encode_as_ids(self, l, *args): 
        return self.encodeSentence(l)
    
    def decodeSentence(self, el):
        dl = []
        for i in el:
            if i == self.iEOS:
                break
            dl.append(self.index2word[i])
        
        return re.sub(r" *%s *" % self.EOL, '\n', ' '.join(dl))
        #return [self.index2word[i] for i in el if i != self.iEOS]
    
    def DecodeIds(self, el):
        return ' '.join(self.decodeSentence(el))
    
    def limitVocab(self, max_size):
        """
        Decreases the vocab size by keeping top-k entries of the vocab based on word2count
        """
        if self.VOCAB_SIZE <= max_size:
            print(f'Current vocab size is {self.VOCAB_SIZE}, no need to decrease size')
            return
#         self.word2index = {}
#         # self.word2count = {}
#         self.index2word = {}
        self.VOCAB_SIZE = max_size
        
#         self.SOS = '<s>'
#         self.EOS = '</s>'
#         self.UNK = '<unk>'
#         self.iSOS = 0
#         self.iEOS = 1
#         self.iUNK = 2
        
        c = Counter(self.word2count)
        m = c.most_common(1)[0][1]
        # c[self.EOL] = m + 5
        c[self.PAD] = m + 4
        c[self.SOS] = m + 3
        c[self.EOS] = m + 2
        c[self.UNK] = m + 1
        
        
        list_of_wc = c.most_common(max_size)
        self.index2word = {i:w for i, (w, _) in enumerate(list_of_wc)}
        self.word2index = {w:i for i, (w, _) in enumerate(list_of_wc)}
        
    def buildLang(self, corpus_gen, sentenceFilterFunct=lambda x: x, num_lines=-1, care_for_newline=False):
        """Creates a language from corpus txt
        
        Keyword Args:
        corpus_file -- input file. contains text data from which the vocab is to be built
        
        """
        
        def auto_id():
            """Generator function for auto-increment id(0)"""
            i = 0
            while(True):
                yield i
                i += 1
        
        ID_gen1 = auto_id()
        word2i = defaultdict(lambda: next(ID_gen1))
        wordCount = defaultdict(int)
        i2word = {}
        
        i2word[word2i[self.PAD]] = self.PAD # 0: PAD
        i2word[word2i[self.SOS]] = self.SOS # 1: SOS
        i2word[word2i[self.EOS]] = self.EOS # 2: EOS
        i2word[word2i[self.UNK]] = self.UNK # 3: UNK
        i2word[word2i[self.EOL]] = self.EOL # 4: EOL
        
        re_space = re.compile('[ ]+')

        #with open(corpus_gen) as fr:
        # with open(data_path + 'train.en') as fr, open(data_path+'normalized.train.en', 'w') as fw:
        fr = corpus_gen
        
        i = 0
        status = 0
        for line in fr:
            i+=1
#             if num_lines > 0:
#                 status = i/num_lines*100
#                 print(f"Progress: {status:0.2F}")
                
            # Build word2i and i2word     
            for t in sentenceFilterFunct(self.normalizeSentence(line)):
                wordCount[t] += 1
                if wordCount[t] >= self.min_count:
                    i2word[word2i[t]] = t
            if care_for_newline:
                wordCount[self.EOL] += 1

        self.word2index = dict(word2i)
        self.index2word = i2word
        self.word2count = dict(wordCount)
        self.VOCAB_SIZE = len(self.word2index)
        print("Vocabulary created...")
        print(f"Vocab Size: {self.VOCAB_SIZE}")
        print(f"Number of lines in corpus: {i}")
        
#     def writeVocab(self, vocab_file):
#         """Writes the vocab to a txt file
        
#         Args:
#         vocab_file -- output file. generated vocab is written to this file
#         """
#         with open(vocab_file, 'w') as fw:
#             for i in range(len(self.word2index)):
#                 w = self.index2word[i]
#                 if (w in [self.EOS, self.SOS, self.UNK]):
#                     fw.write(w + '\n')
#                 elif (self.word2count[w] >= self.min_count):
#                     fw.write(w + '\n')
                    
#         print(f"Vocabulary successfully written to {vocab_file}")
        
#     def read_vocab(self, voc_file):        
#         """Reads vocab from a .txt and updates the lang object
#         Keyword arguments:
#         voc_file: the path of the vocab txt file
#         """
#         with open(voc_file) as f:
#             vocab = [w.strip() for w in f.readlines()]
#         ### DICT VOCAB 
#         self.word2index = {}
#         self.index2word = {}
#         i = 0
#         for w in vocab:
#             self.word2index[w] = i
#             self.index2word[i] = w
#             i += 1
#         self.iSOS = self.word2index[self.SOS]
#         self.iEOS = self.word2index[self.EOS]
#         self.iUNK = self.word2index[self.UNK]
#         self.VOCAB_SIZE = len(self.word2index)
#         #return word2index, index2word, len(word2index)
        
#         print('Vocab read from ', voc_file)
#         print('Size of vocab: ', len(self.word2index))