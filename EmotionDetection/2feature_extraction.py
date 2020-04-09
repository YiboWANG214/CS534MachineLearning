import re
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

emoticon_list = [
    ':‑)', ':)', ':-]', ':]', ':-3', ':3', ':->', ':>', '8-)', '8)', ':-}', ':}', ':o)', ':c)',	':^)', '=]', '=)',
    ':‑d', ':d', '8‑d', '8d', 'x‑d', 'xd', 'x‑d', 'xd', '=d', '=3', 'b^d',
    ':-))',
    ':‑(', ':(', ':‑c', ':c', ':‑<', ':<', ':‑[', ':[', ':-||', '>:[', ':{', ':@', '>:(',
    ":'‑(", ":'(",
    ":'‑)", ":')",
    "-_-"
]

class feature_extraction:

    def __init__(self, data, data1):
        self.data = data
        self.data_withemo = data1
        self.tfidf_vectorizer = None
        self.tfidf_vectorizer1 = None
        self.tfidf_vectorizer2= None
        self.tfidf_vectorizer3 = None
        self.unigram_vectorizer = None
        self.bigram_vectorizer = None
        self.trigram_vectorizer = None
        self.bigram_vectorizer1 = None
        self.trigram_vectorizer1 = None
        self.emoji_map = None
        self.emoticon_list = emoticon_list
        self.emoticon_set = None
        self.maxfeature = 1000


    # def tfidf(self, X=None):
    #     if X is not None:
    #         return self.tfidf_vectorizer.transform(X)
    #     else:
    #         # nltk.download('wordnet')
    #         # tw = nltk.tokenize.TweetTokenizer()
    #         # wnl = nltk.stem.WordNetLemmatizer()
    #         self.tfidf_vectorizer = TfidfVectorizer(
    #             ngram_range=(1, 2),
    #             # min_df=10,
    #             # max_df=0.95,
    #             # sublinear_tf=True,
    #             # norm='l2',
    #             # smooth_idf=True,
    #             # tokenizer=lambda x: [wnl.lemmatize(t) for t in tw.tokenize(x)]
    #         ).fit(self.data)
    #         return self.tfidf_vectorizer.transform(self.data)

    def tfidf1(self, X=None):
        if X is not None:
            return self.tfidf_vectorizer1.transform(X)
        else:
            # nltk.download('wordnet')
            # tw = nltk.tokenize.TweetTokenizer()
            # wnl = nltk.stem.WordNetLemmatizer()
            self.tfidf_vectorizer1 = TfidfVectorizer(
                ngram_range=(1, 1),
                max_features=self.maxfeature
                # min_df=10,
                # max_df=0.95,
                # sublinear_tf=True,
                # norm='l2',
                # smooth_idf=True,
                # tokenizer=lambda x: [wnl.lemmatize(t) for t in tw.tokenize(x)]
            ).fit(self.data)
            return self.tfidf_vectorizer1.transform(self.data)

    def tfidf2(self, X=None):
        if X is not None:
            return self.tfidf_vectorizer2.transform(X)
        else:
            # nltk.download('wordnet')
            # tw = nltk.tokenize.TweetTokenizer()
            # wnl = nltk.stem.WordNetLemmatizer()
            self.tfidf_vectorizer2 = TfidfVectorizer(
                ngram_range=(2, 2),
                max_features=self.maxfeature
                # min_df=10,
                # max_df=0.95,
                # sublinear_tf=True,
                # norm='l2',
                # smooth_idf=True,
                # tokenizer=lambda x: [wnl.lemmatize(t) for t in tw.tokenize(x)]
            ).fit(self.data)
            return self.tfidf_vectorizer2.transform(self.data)
    def tfidf3(self, X=None):
        if X is not None:
            return self.tfidf_vectorizer3.transform(X)
        else:
            # nltk.download('wordnet')
            # tw = nltk.tokenize.TweetTokenizer()
            # wnl = nltk.stem.WordNetLemmatizer()
            self.tfidf_vectorizer3 = TfidfVectorizer(
                ngram_range=(3, 3),
                max_features=self.maxfeature
                # min_df=10,
                # max_df=0.95,
                # sublinear_tf=True,
                # norm='l2',
                # smooth_idf=True,
                # tokenizer=lambda x: [wnl.lemmatize(t) for t in tw.tokenize(x)]
            ).fit(self.data)
            return self.tfidf_vectorizer3.transform(self.data)
    def unigram(self, X=None):
        if X is not None:
            return self.unigram_vectorizer.transform(X)
        else:
            self.unigram_vectorizer = CountVectorizer(ngram_range=(1, 1),max_features=1000).fit(self.data)
            return self.unigram_vectorizer.transform(self.data)

    def bigram(self, X=None):
        if X is not None:
            return self.bigram_vectorizer.transform(X)
        else:
            self.bigram_vectorizer = CountVectorizer(ngram_range=(2, 2),max_features=1000).fit(self.data)
            return self.bigram_vectorizer.transform(self.data)

    def trigram(self, X=None):
        if X is not None:
            return self.trigram_vectorizer.transform(X)
        else:
            self.trigram_vectorizer = CountVectorizer(ngram_range=(3, 3),max_features=1000).fit(self.data)
            return self.trigram_vectorizer.transform(self.data)


    # def bigram1(self, X=None):
    #     if X is not None:
    #         return self.bigram_vectorizer1.transform(X)
    #     else:
    #         self.bigram_vectorizer1 = CountVectorizer(ngram_range=(1, 2)).fit(self.data)
    #         return self.bigram_vectorizer1.transform(self.data)
    #
    # def trigram1(self, X=None):
    #     if X is not None:
    #         return self.trigram_vectorizer1.transform(X)
    #     else:
    #         self.trigram_vectorizer1 = CountVectorizer(ngram_range=(1, 3)).fit(self.data)
    #         return self.trigram_vectorizer1.transform(self.data)
    def embeddings(self, X=None):
        if X is None:
            X = self.data
        nlp = spacy.load('en_core_web_sm')
        word_embeddings = [nlp(x).vector for x in X]
        return word_embeddings

    # find all the emojis and return a map (emoji: index)
    def emoji_lst(self):
        emoji_map = dict()
        for sample in self.data_withemo:
            emojis = re.findall(r'[\U00010000-\U0010ffff]', sample)
            for emoji in emojis:
                if emoji not in emoji_map:
                    emoji_map[emoji] = len(emoji_map)
        self.emoji_map = emoji_map

    #  return a vector of emoji counts in each row of X
    def get_emoji_vec(self, X):
        if X is None:
            self.emoji_lst()
            X = self.data_withemo
        emoji_vector = np.empty((0, len(X)))
        for i in range(len(self.emoji_map)):
            emoji_vector = np.append(emoji_vector, np.zeros((1, len(X))), axis=0)
        for i, text in enumerate(X):
            emojis = re.findall(r'[\U00010000-\U0010ffff]', text)
            for emoji in emojis:
                if emoji in self.emoji_map:
                    emoji_vector[self.emoji_map[emoji]][i] += 1
        return emoji_vector.T

    # find all the emoticons in training set and return a map (emoticon: index)
    def emoticon_lst(self):
        emoticon_map = dict()
        emoticon_set = set()
        for sample in self.data_withemo:
            emoticons = self.find_emoticon(sample)
            for emoticon in emoticons:
                emoticon_set.add(emoticon)
                if emoticon not in emoticon_map:
                    emoticon_map[emoticon] = len(emoticon_map)
        self.emoticon_set = emoticon_set
        self.emoticon_map = emoticon_map


    # find emoticons in the given text
    def find_emoticon(self, text):
        emoticons = []
        for t in text.split():
            if t in self.emoticon_list:
                emoticons.append(t)
        return emoticons

    #  return a vector of emoticon counts in each row of X
    def get_emoticon_vec(self, X):
        if X is None:
            self.emoticon_lst()
            X = self.data_withemo
        #  initialize the vector
        emoticon_vector = np.empty((0, len(X)))
        for i in range(len(self.emoticon_set)):
            emoticon_vector = np.append(emoticon_vector, np.zeros((1, len(X))), axis=0)
        for i, text in enumerate(X):
            #  find the emoticons in the text and update the vector
            for t in text.split():
                if t in self.emoticon_set:
                    emoticon_vector[self.emoticon_map[t]][i] += 1
        return emoticon_vector.T

    def get_features1(self, X=None, X_emo=None):

        tfidf1 = self.tfidf1(X)
        tfidf2 = self.tfidf2(X)
        tfidf3 = self.tfidf3(X)
        emoji = self.get_emoji_vec(X_emo)
        emoticon = self.get_emoticon_vec(X_emo)

        print("size of emoji map: ", len(self.emoji_map))
        print("size of emoticon map: ", len(self.emoticon_map))

        return [tfidf1, tfidf2, tfidf3, np.array(emoji), np.array(emoticon)]

    def get_features(self, X=None, X_emo=None):

        # tfidf = self.tfidf(X)
        unigram = self.unigram(X)
        # print(unigram.shape)
        bigram = self.bigram(X)
        # print(bigram.shape)
        trigram = self.trigram(X)
        # print(trigram.shape)
        emoji = self.get_emoji_vec(X_emo)
        emoticon = self.get_emoticon_vec(X_emo)
        # embeddings = self.embeddings(X)
        # all_features = hstack((unigram, bigram, trigram, emoji, emoticon))
        print("size of emoji map: ", len(self.emoji_map))
        print("size of emoticon map: ", len(self.emoticon_map))

        return [unigram, bigram, trigram, np.array(emoji), np.array(emoticon)]


    def getBigram(self, X=None):
        bigram = self.bigram1(X)
        return bigram

    def getTrigram(self, X=None):
        trigram = self.trigram1(X)
        return trigram
