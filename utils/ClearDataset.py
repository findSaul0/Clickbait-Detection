import nltk
from nltk.corpus import stopwords
import string as s

### THIS FUNCTION SPLIT THE DATA INTO WORDS ###
def tokenization(text):
    lst=text.split()
    return lst


### CONVERTING WORDS TO LOWERCASE ###
def lowercasing(lst):
    new_lst=[]
    for i in lst:
        i=i.lower()
        new_lst.append(i)
    return new_lst


### REMOVING STOP WORDS ###
def remove_stopwords(lst):
    stop=stopwords.words('english')
    new_lst=[]
    for i in lst:
        if i not in stop:
            new_lst.append(i)
    return new_lst

### REMOVING PUNCTUATION ###
def remove_punctuations(lst):
    new_lst=[]
    for i in lst:
        for j in s.punctuation:
            i=i.replace(j,'')
        new_lst.append(i)
    return new_lst


### REMOVING NUMBERS ###
def remove_numbers(lst):
    nodig_lst=[]
    new_lst=[]
    for i in lst:
        for j in s.digits:
            i=i.replace(j,'')
        nodig_lst.append(i)
    for i in nodig_lst:
        if i!='':
            new_lst.append(i)
    return new_lst

### REMOVING EXTRA SPACES ###
def remove_spaces(lst):
    new_lst=[]
    for i in lst:
        i=i.strip()
        new_lst.append(i)
    return new_lst

### LAMMATIZATION ###
lemmatizer=nltk.stem.WordNetLemmatizer()
def lemmatzation(lst):
    #nltk.download("wordnet")
    new_lst=[]
    for i in lst:
        i=lemmatizer.lemmatize(i)
        new_lst.append(i)
    return new_lst