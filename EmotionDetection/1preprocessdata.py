import pandas as pd
from IPython.display import display
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
import string
from textblob import TextBlob, Word
import importlib, sys
import utils
import spacy

importlib.reload(sys.modules['utils'])
import utils

import re
import emoji

pd.set_option('display.max_colwidth', -1)  # set max col width in order we can some more content

# punctuation = '["\'?,\.]'  # I will replace all these punctuation with ''
abbr_dict1 = {
    "what's": "what is",
    "what're": "what are",
    "who's": "who is",
    "who're": "who are",
    "where's": "where is",
    "where're": "where are",
    "when's": "when is",
    "when're": "when are",
    "how's": "how is",
    "how're": "how are",

    "i'm": "i am",
    "we're": "we are",
    "you're": "you are",
    "they're": "they are",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "there're": "there are",

    "i've": "i have",
    "we've": "we have",
    "you've": "you have",
    "they've": "they have",
    "who've": "who have",
    "would've": "would have",
    "not've": "not have",

    "i'll": "i will",
    "we'll": "we will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "they'll": "they will",

    "isn't": "is not",
    "wasn't": "was not",
    "aren't": "are not",
    "weren't": "were not",
    "can't": "can not",
    "couldn't": "could not",
    "don't": "do not",
    "didn't": "did not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "doesn't": "does not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",

    "whats": "what is",
    "whatre": "what are",
    "whos": "who is",
    "whore": "who are",
    "wheres": "where is",
    "wherere": "where are",
    "whens": "when is",
    "whenre": "when are",
    "hows": "how is",
    "howre": "how are",

    "youre": "you are",
    "theyre": "they are",
    "its": "it is",
    "hes": "he is",
    "shes": "she is",
    "thats": "that is",
    "theres": "there is",
    "therere": "there are",

    "weve": "we have",
    "youve": "you have",
    "theyve": "they have",
    "whove": "who have",
    "wouldve": "would have",
    "notve": "not have",

    "youll": "you will",
    "itll": "it will",
    "theyll": "they will",

    "isnt": "is not",
    "wasnt": "was not",
    "arent": "are not",
    "werent": "were not",
    "cant": "can not",
    "couldnt": "could not",
    "dont": "do not",
    "didnt": "did not",
    "shouldnt": "should not",
    "wouldnt": "would not",
    "doesnt": "does not",
    "havent": "have not",
    "hasnt": "has not",
    "hadnt": "had not",
    "wont": "will not",

    "whatsup": "what is up",
    "bcoz": "because",
    "wlcm": "welcome",
    "ntng": "nothing",
    "gf": "girlfriend",
    "bf": "boyfriend",

    "wrkng": "work",
    "convo": "conversation",
    "luv": "love",
    "txt": "text",
    "knw": "know",
    "sry": "sorry",
    "srry": "sorry",
    "chating": "chat",
    "frnds": "friend",
    "urself": "yourself",
    "myslf": "myself",
    "ofcourse": "of course",
    "murgi": "chicken",
    "srsly": "seriously",
    "oky": "okay",
    "rofl": "haha",
    "lmao": "haha",
    "tbh": "to be honest",
    "tmrw": "tomorrow",
    "opps": "oops",
    "alrighty": "alright",
    "dumpass": "dump ass",
    "frds": "friend",
    "comeon": "come on",
    "hogaya": "done",
    "frst": "first",
    "msgs": "messages",
    "ttyl": "talk to you later",
    "thts": "that is",
    "ikr": "i know right ?",
    "thanku": "thank you",
    "whts": "what is",
    "ypu": "you",
    "ryt": "right",
    "anytym": "anytime",
    "bitch(.*)": "bitch",
    "lyk": "like",
    "nopes": "nope",
    "ehat": "what",
    "hppnd": "happened",
    "btwn": "between",
    "sended": "send",
    "mrng": "morning",
    "iwant": "i want",
    "prblm": "problem",
    "tomorow": "tomorrow",
    "ahaha": "haha",
    "nvm": "never mind",
    "tlk": "talk",
    "bbey": "baby",
    "gril": "grill",
    "roboy": "robot",
    "intrest": "interest",
    "lonley": "lonely",
    "abt": "about",
    "bday": "birthday",
    "ohkay": "okay",

    "wat": "what",
    "watever": "whatever",
    "rply": "reply",
    "tnx": "thanks",
    "awsm": "awesome",
    "bestfriend": "best friend",
    "smthing": "something",
    "awesom": "awesome",
    "waana": "wanna",
    "iand": "i and",
    "probs": "problem",
    "meanin": "meaning",
    "diffrent": "different",
    "jzt": "just",
    "orignal": "original",
    "friendam?": "friend",
    "vry": "very",
    "evrything": "everything",
    "histery": "history",
    "plz": "please",
    "beauti4": "beautiful",

    # punctuation: '',
    # '\s+': ' ',  # replace multi space with one single space
}

abbr_dict2 = {
    "u": "you",
    "ur": "your",
    "r": "are",
    "c": "see",
    "4": "for",
    "m": "am",
    "n": "and",
    "ans": "answer",
    "wt": "what",
    "dnt": "do not",
    "sec": "second",
    "min": "minute",
    "fr": "for",
    "wrk": "work",
    "frm": "from",
    "btr": "better",
    "abt": "about",
    "ne": "me",
    "ive": "i have",
    "ill": "i will",

}

emoticon_list = [
    ':‑)', ':)', ':-]', ':]', ':-3', ':3', ':->', ':>', '8-)', '8)', ':-}', ':}', ':o)', ':c)', ':^)', '=]', '=)',
    ':‑D', ':D', '8‑D', '8D', 'x‑D', 'XD', 'x‑D', 'xD', '=D', '=3', 'b^d', ';)', ':-P',
    ':-))', ':-)', ':))',
    ':‑(', ':(', ':‑c', ':c', ':‑<', ':<', ':‑[', ':[', ':-||', '>:[', ':{', ':@', '>:(',
    ":'‑(", ":'(", '):',
    ":'‑)", ":')",
    "-_-", '(^・^)',
    ':]', ':/',
]


def clean(data):
    data.turn1 = data.turn1.str.lower()  # conver to lower case
    data.turn2 = data.turn2.str.lower()
    data.turn3 = data.turn3.str.lower()
    data.turn1 = data.turn1.astype(str)
    data.turn2 = data.turn2.astype(str)
    data.turn3 = data.turn3.astype(str)
    return data


def punctuation(data):
    exclude = set(string.punctuation)
    for i in data:
        if i in exclude:
            data = data.replace(i, " ")
    # data = ''.join(ch for ch in data if ch not in exclude)
    return data


def abbreviation1(data):
    data.replace(abbr_dict1, regex=True, inplace=True)
    return data


def abbreviation2(data):
    tokens = word_tokenize(data)
    key = list(abbr_dict2.keys())
    same = set(tokens) & set(key)
    list_same = list(same)
    for i in list_same:
        expand = abbr_dict2[i]
        tokens[tokens.index(i)] = expand
    data = ' '.join(tokens)
    return data


def cleanText(text, remEmojis=2):
    text = text.lower()
    text = re.sub(r"’", "'", text)
    text = re.sub(r"`", "'", text)

    text = re.sub(r"\b[y]+\b", "why", text)
    text = re.sub(r"\bpl[s]+\b", "please", text)
    text = re.sub(r"\bpl[z]+\b", "please", text)
    text = re.sub(r"\baren't\b", "are not", text)
    text = re.sub(r"\bb[e]?coz\b", "because", text)
    text = re.sub(r"\by[a]+r\b", "yaar", text)
    text = re.sub(r"\bth[a]?nx\b", "thanks", text)
    text = re.sub(r"\bye[s]+\b", "yes", text)
    text = re.sub(r"\b[o]+[k]+\b", "okay", text)
    text = re.sub(r"\bha[ha]+\b", "haha", text)
    text = re.sub(r"\bhe[he]+\b", "haha", text)
    text = re.sub(r"\bby[y]+\b", "bye", text)
    text = re.sub(r"\b[l]+[o]+[l]+\b", "haha", text)
    text = re.sub(r"\bum[m]+\b", "umm", text)
    text = re.sub(r"\bho[o]+\b", "who", text)
    text = re.sub(r"\bhe[y]+\b", "hey", text)
    text = re.sub(r"\bkn[o]+[w]+\b", "know", text)
    text = re.sub(r"\byu[p]+\b", "yup", text)
    text = re.sub(r"\bhref\b", "", text)
    text = re.sub(r"\byo[u]+\b", "you", text)
    text = re.sub(r"\bye[a]+[h]*\b", "yeah", text)
    text = re.sub(r"\byo[u]+\b", "you", text)
    text = re.sub(r"\bx‑d\b", "haha", text)
    text = re.sub(r"\bna[h]+\b", "nah", text)
    text = re.sub(r"\bto[o]+\b", "too", text)
    text = re.sub(r"\b'i\b", "i", text)
    text = re.sub(r"\b'you\b", "you", text)
    text = re.sub(r"\bnt[n]?g\b", "nothing", text)
    text = re.sub(r"\bi\b", "I", text)
    text = re.sub(r"\b[m]+[e]+\b", "me", text)
    text = re.sub(r"\b[o]+[k]+[a]+[y]+\b", "okay", text)
    text = re.sub(r"\b[o]+[h]+\b", "oh", text)
    text = re.sub(r"\b[b]+[y]+[e]+\b", "bye", text)
    text = re.sub(r"\by[a]+\b", "ya", text)
    text = re.sub(r"\b[w]+[h]+[y]+\b", "why", text)
    text = re.sub(r"\b[w]+[o]+[w]+\b", "wow", text)
    text = re.sub(r"\b[a]+[w]+\b", "aww", text)
    text = re.sub(r"\b[h]+[m]+\b", "hmm", text)
    text = re.sub(r"\bwh[a]+\b", "what", text)
    text = re.sub(r"\bn[a]+[h]*\b", "no", text)
    text = re.sub(r"\bncr[a]+[z]+[y]+\b", "crazy", text)
    text = re.sub(r"\b[y]+[a]*[r]+\b", "friend", text)
    text = re.sub(r"\bbestie\b", "best friend", text)
    text = re.sub(r"\bnthg\b", "nothing", text)
    text = re.sub(r"\bgtg\b", "got to go", text)
    text = re.sub(r"\bwhre\b", "where", text)
    text = re.sub(r"\bplese\b", "please", text)
    text = re.sub(r"\bbakwas\b", "nonsense", text)
    text = re.sub(r"\bdafuq\b", "the fuck", text)
    text = re.sub(r"\bdespo\b", "desperate", text)
    text = re.sub(r"\bdespo\b", "desperate", text)
    text = re.sub(r"\bchatbot\b", "chat bot", text)
    text = re.sub(r"\bselfie\b", "picture", text)
    text = re.sub(r"\b[a]+[c]+[h]+[a]+\b", "okay", text)
    text = re.sub(r"\b[n]+[o]+[w]+\b", "now", text)
    text = re.sub(r"\b[w]+[o]+[a]+[h]+\b", "woah", text)
    text = re.sub(r"\bfrnd[z]+\b", "friends", text)
    text = re.sub(r"iam | iam", "i am", text)
    text = re.sub(r"\bawsome\b", "awesome", text)
    text = re.sub(r"\bc[\']?mon\b", "come on", text)
    text = re.sub(r"\bsorr\b", "sorry", text)
    text = re.sub(r"\biknow\b", "i know", text)
    text = re.sub(r"\btelme\b", "tell me", text)
    text = re.sub(r"\bi and\b", "i and", text)
    text = re.sub(r"\bnomber\b", "number", text)
    text = re.sub(r"\bni8\b", "night", text)
    text = re.sub(r"\bwaiti[n]+\b", "waiting", text)
    text = re.sub(r"\b[s]+[o]+\b", "so", text)
    text = re.sub(r"\b[y]+[o]+\b", "yo", text)
    text = re.sub(r"\b[h]+[e]+[l]+[o]+\b", "hello", text)
    text = re.sub(r"\bfavs\b", "favourite", text)
    text = re.sub(r"\bwatsup\b", "what is up", text)
    text = re.sub(r"\bchathero\b", "chat hero", text)
    text = re.sub(r"\broght\b", "right", text)
    text = re.sub(r"\bjabh\b", "when", text)
    text = re.sub(r"\bb\'coz\b", "because", text)
    text = re.sub(r"\bmovi\b", "movie", text)
    text = re.sub(r"\bfuuny\b", "funny", text)
    text = re.sub(r"\bfack\b", "fuck", text)
    text = re.sub(r"\bgarls\b", "girls", text)
    text = re.sub(r"\bnambar\b", "number", text)
    text = re.sub(r"\biand\b", "i and", text)
    text = re.sub(r"\bvedio\b", "video", text)
    text = re.sub(r"\bvidio\b", "video", text)
    text = re.sub(r"\bwhear\b", "where", text)
    text = re.sub(r"\bb[a]?tao\b", "tell", text)
    text = re.sub(r"\bple[z]+\b", "please", text)
    text = re.sub(r"\bofcorse\b", "of course", text)
    text = re.sub(r"\bobvios\b", "obvious", text)
    text = re.sub(r"\bhalp\b", "help", text)
    text = re.sub(r"\bperoson\b", "person", text)
    text = re.sub(r"\bveggi[e]+\b", "vegetable", text)
    text = re.sub(r"\b[a]+[l]+\b", "all", text)
    text = re.sub(r"\bhwo\b", "how", text)
    text = re.sub(r"\bcopuen\b", "coupon", text)
    text = re.sub(r"\beverithing\b", "everything", text)
    text = re.sub(r"\bkisliye\b", "why", text)
    text = re.sub(r"\bconputer\b", "computer", text)
    text = re.sub(r"\bwh7\b", "why", text)
    text = re.sub(r"\bemotionol\b", "emotional", text)
    text = re.sub(r"\bdonnt\b", "do not", text)
    text = re.sub(r"\bcsnt\b", "can not", text)
    text = re.sub(r"\bfemalr\b", "female", text)
    text = re.sub(r"\baroung\b", "around", text)
    text = re.sub(r"\bsombody\b", "somebody", text)
    text = re.sub(r"\bsoch\b", "think", text)
    text = re.sub(r"\balgoritm\b", "algorithm", text)
    text = re.sub(r"\bmaked\b", "made", text)
    text = re.sub(r"\b[f]+[u]+[n]+[n]+[y]+\b", "funny", text)
    text = re.sub(r"\b[i]+\b", "i", text)
    text = re.sub(r"\b[s]+[l]+[e]+[p]+\b", "sleep", text)
    text = re.sub(r"\bsoch\b", "think", text)
    text = re.sub(r"\bbrb\b", "be right back", text)
    text = re.sub(r"\b[g]+o[o]+[d]+\b", "good", text)
    text = re.sub(r"\bsmthng\b", "something", text)
    text = re.sub(r"\bmarrige\b", "marriage", text)
    text = re.sub(r"\bth[o]+\b", "though", text)
    text = re.sub(r"\bi\'am\b", "i am", text)
    text = re.sub(r"\byou\'d\b", "you would", text)
    text = re.sub(r"\brommate\b", "roommate", text)
    text = re.sub(r"\b[n]+[o]+\b", "no", text)
    text = re.sub(r"\bmayb\b", "maybe", text)
    text = re.sub(r"\bmayb\b", "maybe", text)
    text = re.sub(r"\bbeby\b", "baby", text)
    text = re.sub(r"\bhws\b", "how is", text)
    text = re.sub(r"\breferal\b", "referral", text)
    text = re.sub(r"\bdefinetly\b", "definitely", text)
    text = re.sub(r"\biwill\b", "i will", text)
    text = re.sub(r"\bthik\b", "okay", text)
    text = re.sub(r"\bwerid\b", "weird", text)
    text = re.sub(r"\btolk\b", "talk", text)
    text = re.sub(r"\bwrking\b", "working", text)
    text = re.sub(r"\bbsy\b", "busy", text)
    text = re.sub(r"\bfrinds\b", "friends", text)
    text = re.sub(r"\bjuat\b", "just", text)
    text = re.sub(r"\byout\b", "your", text)
    text = re.sub(r"\bdoesnot\b", "does not", text)
    text = re.sub(r"\bjoaking\b", "joking", text)

    text = re.sub(r"\bunderstant\b", "understand", text)
    text = re.sub(r"\btheb\b", "then", text)
    text = re.sub(r"\btimw\b", "time", text)
    text = re.sub(r"\bso[r]+y\b", "sorry", text)
    text = re.sub(r"\bthin[g]+[s]+\b", "things", text)
    text = re.sub(r"\bthnk[s]+\b", "thanks", text)
    text = re.sub(r"\b[y]+[e]+[a]+[h]+\b", "yeah", text)
    text = re.sub(r"\b[h]+[e]+[l]+[p]+\b", "help", text)
    text = re.sub(r"\bkanguage\b", "language", text)
    text = re.sub(r"\bbirth[d]+[a]+[y]+\b", "birthday", text)
    text = re.sub(r"\bothr\b", "other", text)
    text = re.sub(r"\bwrshp\b", "worship", text)
    text = re.sub(r"\b[r]+[i]+[g]+[h]+[t]+\b", "right", text)
    text = re.sub(r"\bunders[t]+and\b", "understand", text)
    text = re.sub(r"\bsemd\b", "send", text)
    text = re.sub(r"\btommrow\b", "tomorrow", text)
    text = re.sub(r"\b[p]+[a]+[k]+[a]+\b", "sure", text)
    text = re.sub(r"\btomarow\b", "tomorrow", text)
    text = re.sub(r"\bbeacause\b", "because", text)
    text = re.sub(r"\b[l]+[o]+[v]+[e]+\b", "love", text)
    text = re.sub(r"\b[t]+[h]+[a]+[n]+[k]+[s]+\b", "thanks", text)
    text = re.sub(r"\bstuding\b", "studying", text)
    text = re.sub(r"\b[l]+[i]+[f]+[e]+\b", "life", text)
    text = re.sub(r"\bwithaut\b", "without", text)
    text = re.sub(r"\bshowin\b", "showing", text)
    text = re.sub(r"\bplx\b", "please", text)
    text = re.sub(r"\bsuoerb\b", "superb", text)
    text = re.sub(r"\bniether\b", "neither", text)
    text = re.sub(r"\bintrast\b", "interest", text)
    text = re.sub(r"\bthiz\b", "this", text)
    text = re.sub(r"\bmidnigt\b", "midnight", text)
    text = re.sub(r"\bcumon\b", "come on", text)
    text = re.sub(r"\blitteraly\b", "literally", text)
    text = re.sub(r"\bofline\b", "offline", text)
    text = re.sub(r"\bnotjin\b", "nothing", text)
    text = re.sub(r"\btlking\b", "talking", text)
    text = re.sub(r"\bwhatelse\b", "what else", text)
    text = re.sub(r"\bbabay\b", "baby", text)
    text = re.sub(r"\bvertual\b", "virtual", text)
    text = re.sub(r"\b[m]+[o]+[r]+[e]+\b", "more", text)
    text = re.sub(r"\bbecz\b", "because", text)
    text = re.sub(r"\biove\b", "love", text)
    text = re.sub(r"\beverythng\b", "everything", text)
    text = re.sub(r"\byoue\b", "your", text)
    text = re.sub(r"\bphoro\b", "photo", text)
    text = re.sub(r"\birrating\b", "irritating", text)
    text = re.sub(r"\bborimg\b", "boring", text)
    text = re.sub(r"\bonpy\b", "only", text)
    text = re.sub(r"\b[s]+[w]+[e]+[t]+\b", "sweet", text)
    text = re.sub(r"\brelationshpis\b", "relationships", text)
    text = re.sub(r"\bisit\b", "is it", text)
    text = re.sub(r"\btimepass\b", "time pass", text)
    text = re.sub(r"\bdrawin\b", "drawing", text)
    text = re.sub(r"\bpossitive\b", "positive", text)
    text = re.sub(r"\bmoblie\b", "mobile", text)
    text = re.sub(r"\braferal\b", "referrel", text)
    text = re.sub(r"\bdoller\b", "dollar", text)
    text = re.sub(r"\b[s]+[t]+[u]+[p]+[i]+[d]+\b", "stupid", text)
    text = re.sub(r"\babaout\b", "about", text)
    text = re.sub(r"\bundrstnd\b", "understand", text)
    text = re.sub(r"\bwhar\b", "what", text)
    text = re.sub(r"\bcoment\b", "comment", text)
    text = re.sub(r"\b[b]+[c]+[o]+[z]+\b", "because", text)
    text = re.sub(r"\bfebret\b", "favourite", text)
    text = re.sub(r"\bbeautyfull\b", "beautiful", text)
    text = re.sub(r"\bphoro\b", "photo", text)
    text = re.sub(r"\bbarthdey\b", "birthday", text)
    text = re.sub(r"\bhpy\b", "happy", text)
    text = re.sub(r"\beverythings\b", "everything is", text)
    text = re.sub(r"\bsofteware\b", "software", text)
    text = re.sub(r"\bha[p]+en\b", "happen", text)
    text = re.sub(r"\bconvence\b", "convince", text)
    text = re.sub(r"\bsecrt\b", "secret", text)
    text = re.sub(r"\babuot\b", "about", text)
    text = re.sub(r"\bchicolate\b", "chocolate", text)
    text = re.sub(r"\biscoming\b", "is coming", text)
    text = re.sub(r"\bacter\b", "actor", text)
    text = re.sub(r"\bsometning\b", "something", text)
    text = re.sub(r"\bhomour\b", "humor", text)
    text = re.sub(r"\bintreasted\b", "interested", text)
    text = re.sub(r"\bmesseged\b", "messaged", text)
    text = re.sub(r"\blieng\b", "lying", text)
    text = re.sub(r"\bvegitarians\b", "vegetarian", text)
    text = re.sub(r"\bautowala\b", "driver", text)
    text = re.sub(r"\bcudnt\b", "could not", text)
    text = re.sub(r"\bgols\b", "goals", text)
    text = re.sub(r"\bcleaverbot\b", "clever bot", text)
    text = re.sub(r"\bpartywonderful\b", "party wonderful", text)
    text = re.sub(r"\bbussiness\b", "business", text)
    text = re.sub(r"\biscoming\b", "is coming", text)
    text = re.sub(r"\bexplean\b", "explain", text)
    text = re.sub(r"\bwecome\b", "welcome", text)
    text = re.sub(r"\b[w]+[a]+[y]+\b", "way", text)
    text = re.sub(r"\bsupreb\b", "superb", text)
    text = re.sub(r"\bshoping\b", "shopping", text)
    text = re.sub(r"\bemojifortune\b", "emoji fortune", text)
    text = re.sub(r"\bplessure\b", "pleasure", text)
    text = re.sub(r"\basswhole\b", "ass whole", text)
    text = re.sub(r"\blyier\b", "lier", text)
    text = re.sub(r"\bhappyhalloween\b", "happy halloween", text)
    text = re.sub(r"\bgenious\b", "genius", text)
    text = re.sub(r"\b[e]+[x]+[a]+[c]+[t]+[l]+[y]+\b", "exactly", text)
    text = re.sub(r"\buniversiry\b", "university", text)
    text = re.sub(r"\bsamething\b", "same thing", text)
    text = re.sub(r"\b[h]+[o]+[w]+\b", "how", text)
    text = re.sub(r"\bokau\b", "okay", text)
    text = re.sub(r"\bgirfreind\b", "girlfriend", text)
    text = re.sub(r"\batleas\b", "at least", text)
    text = re.sub(r"\b[c]+[u]+[t]+[e]+\b", "cute", text)
    text = re.sub(r"\beverybodyincluding\b", "everybody including", text)
    text = re.sub(r"\b[w]+[o]+[r]+[d]+\b", "word", text)
    text = re.sub(r"\breplyme\b", "tell me", text)
    text = re.sub(r"\bjudw[a]+\b", "twin", text)
    text = re.sub(r"\bbestfrnd\b", "best friend", text)
    text = re.sub(r"\bloveme\b", "love me", text)
    text = re.sub(r"\basusual\b", "as usual", text)
    text = re.sub(r"\bocourse\b", "of course", text)
    text = re.sub(r"\bfotographs\b", "photographs", text)
    text = re.sub(r"\bjudw[a]+\b", "twin", text)
    text = re.sub(r"\bjudw[a]+\b", "twin", text)

    return text


def lemmatization(data):
    s = " "
    words_lemma = []
    lemmatizer = WordNetLemmatizer()
    stemer = PorterStemmer()
    nlp = spacy.load('en', disable=['parser', 'ner'])
    data = nlp(data)
    data = " ".join([token.lemma_ for token in data])

    words = word_tokenize(data)
    for word in words:
        word = lemmatizer.lemmatize(word)
        # word = stemer.stem(word)
        word = Word(word)
        word.lemmatize()
        words_lemma.append(word)
    data = s.join(words_lemma)
    return data


def spell_check(data):
    data = TextBlob(data)
    data = data.correct()
    return data


def remove_stopwords(data):
    s = " "
    stop_words = set(stopwords.words('english'))
    stop_words.remove("not")
    stop_words.remove('each')
    stop_words.remove('same')
    stop_words.remove('above')
    stop_words.remove('up')
    stop_words.remove('few')
    stop_words.remove('below')
    stop_words.remove('or')
    stop_words.remove('before')
    stop_words.remove('while')
    stop_words.remove('should')
    stop_words.remove('once')
    stop_words.remove('no')
    stop_words.remove('after')
    stop_words.remove('some')
    stop_words.remove('through')
    stop_words.remove('only')
    stop_words.remove('nor')
    stop_words.remove('more')
    stop_words.remove('most')
    stop_words.remove('further')
    stop_words.remove('against')
    stop_words.add("-PRON-")
    tokens = word_tokenize(data)
    data = [i for i in tokens if not i in stop_words]
    data = s.join(data)
    return data


def delete_PRON(data):
    s = " "
    tokens = word_tokenize(data)
    PRON = set("")
    PRON.add("-PRON-")
    data = [i for i in tokens if not i in PRON]
    data = s.join(data)
    return data


def emoticon(data):
    list = emoticon_list
    for i in list:
        if data.find(i) is not -1:
            m = data.find(i)
            n = len(i)
            data = data[:m] + ' ' + data[m:m + n] + ' ' + data[m + n:]
    return data


def delete_emojis(data):
    data = emoji.demojize(data)
    data = re.sub(':\S+?:', ' ', data)
    return data


train = pd.read_csv('/Users/yibowang/PycharmProjects/untitled/testwithoutlabels.txt', encoding='utf-8', delimiter='\t')

# train = clean(train)
# # print(train[0:10])
# train = abbreviation1(train)

n = len(train)
for i in range(0, n):
    for j in range(1, 4):
        a = train.iat[i, j]
        # a = abbreviation2(a)
        # a = delete_emojis(a)
        # a = cleanText(a)
        a = emoticon(a)
        # a = punctuation(a)
        # a = lemmatization(a)
        # # a = delete_PRON(a)
        # a = remove_stopwords(a)
        # # a = spell_check(a)
        a = str(a)
        train.iat[i, j] = a
display(train.head(5))
train.to_csv('preprocessed_test_without_labels_emoticons_CleanText.csv', index=False)

# data = pd.read_csv('/Users/yibowang/PycharmProjects/untitled/preprocessed_train_delete_stopwords_cleanText.csv')
#
# n = len(data)
# for i in range(n):
#     for j in range(1, 4):
#         a = data.iat[i, j]
#         a = str(a)
#         a = delete_PRON(a)
#         data.iat[i, j] = a
# data.to_csv('preprocessed_train_delete_stopwords_delete_PRON.csv', index=False)
