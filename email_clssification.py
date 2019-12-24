import os
import io
import numpy
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from collections import Counter
from pandas import DataFrame
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def load_data(dir):
    list = []
    for file in os.listdir(dir):
         with open(dir + "/" + file, "rb") as f:
             body = f.read().decode("utf-8", errors="ignore").splitlines()
             list.append(' '.join(body))
    return list

def preprocess(text):
    text = text.lower()
    return tokens


BASE_DATA_DIR="emails"
# load and tag data
ham = [(text, 'ham') for text in load_data(BASE_DATA_DIR + '/ham')]
spam = [(text, 'spam') for text in load_data(BASE_DATA_DIR + '/spam')]
allData = ham + spam



def preprocess(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'[a-z]+')
    tokens=tokenizer.tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    stoplist = stopwords.words('english')
    #stoplist.append('subject')
    tokens = [t for t in tokens if not t in stoplist]
    return tokens

allData = [(preprocess(text), label) for (text,label) in allData]

random.shuffle(allData)
splitp = 0.80 # 80/20 split
train = allData[:int(splitp*len(allData))]
test = allData[int(splitp*len(allData)):]
data = pd.DataFrame(allData,columns=['emails tokens','classes'])
print(data)
#PS: am not using the SpamDict and HamDict and am keeping this part just to have the spam token list
#and ham token list to use later in calculating the probability of each word in the emails
SpamDict = {}
HamDict = {}
spam_token_list = []
ham_token_list = []
def featurizeTokens(tokens, is_spam):
    for x in tokens :
        #print(x + ' ' + str(is_spam))
        if (is_spam):
            spam_token_list.append(tokens)
            if x not in SpamDict:
                SpamDict[x]=0
            SpamDict[x] += 1
        else :
            ham_token_list.append(tokens)
            if x not in HamDict:
                HamDict[x]=0
            HamDict[x] += 1



for (tokens,label) in train:
    featurizeTokens(tokens, label == 'spam')


def tokenConditionalProbability(dataset):

    # Number of samples in dataset
    sampleSize = len(dataset)

    # Dictionary of token-probability pairs
    conditionalProbabilities = {}

    # Count probability of occurence of each token
    flatten = []
    flatten[len(flatten):] = [ token for sample in dataset for token in sample ]
    tokenCount = Counter(flatten)
    conditionalProbabilities = { key : value / sampleSize for key, value in tokenCount.items()}

    return conditionalProbabilities

spamTokensConditionalProbabilities = tokenConditionalProbability(spam_token_list) # Dictionary
hamTokensConditionalProbabilities = tokenConditionalProbability(ham_token_list) # Dictionary

def spamPosteriorProbability(tokenList):
    spamTokenConditionalProbability = 1
    hamTokenConditionalProbability = 1
    for token in tokenList:

        if token not in spamTokensConditionalProbabilities:
            spamTokenConditionalProbability *= 0.01 # To minimize false positive,i did not use Laplace smoothing
        else:
            spamTokenConditionalProbability *= spamTokensConditionalProbabilities[token]

        if token not in hamTokensConditionalProbabilities:
            hamTokenConditionalProbability *= 0.01 # To mininize false negative,i did not use Laplace smoothing
        else:
            hamTokenConditionalProbability *= hamTokensConditionalProbabilities[token]

    return 'spam' if spamTokenConditionalProbability>hamTokenConditionalProbability else 'ham'



test_output=[]
for tokens in test:
    test_output.append((tokens[0],spamPosteriorProbability(tokens[0])))


truePositive = trueNegative = falsePositive = falseNegative = 0
for real, pred in zip(test, test_output):
    expected = real[1]
    found = pred[1]
    if expected == 'spam':
        if found=='ham':
            truePositive += 1
        else:
            falseNegative += 1
    elif expected == 'ham':
        if found=='spam':
            falsePositive += 1
        else:
            trueNegative += 1

print('{0} = {1}'.format('True Positive', truePositive))
print('{0} = {1}'.format('False Negative', falseNegative))
print('{0} = {1}'.format('False Positive', falsePositive))
print('{0} = {1}'.format('True Negative', trueNegative))
print()

#Confusion Matrix
d = {'Tested Spam' : pd.Series([truePositive, falsePositive, truePositive + falsePositive], index=['Expected Spam','Expected Ham', 'Total'])}
df = pd.DataFrame(d)
df['Tested Ham'] = pd.Series([falseNegative, trueNegative, falseNegative + trueNegative], index=['Expected Spam','Expected Ham', 'Total'])
df['Total'] = pd.Series([truePositive + falseNegative, falsePositive + trueNegative, truePositive + falseNegative + falsePositive + trueNegative], index=['Expected Spam','Expected Ham', 'Total'])
print('                  Confusion Matrix')
print(df)
print()
print('Accuracy =', (truePositive + trueNegative)/len(test) * 100, '%')
print('Precision =', truePositive / (truePositive + falsePositive) * 100, '%')


TruePosRate=truePositive/len(test)
FalsePosRate=falsePositive/len(test)
FalseNegRate=falseNegative/len(test)
TrueNegRate=trueNegative/len(test)
df = pd.DataFrame([[TruePosRate, FalsePosRate], [FalseNegRate, TrueNegRate]])
fig = plt.figure()
ax = sn.heatmap(100*df, vmin=0, vmax=100, cmap='Blues', annot=True, fmt='.2f', annot_kws={"size":20}, linewidths=0.5)
ax.set_xlabel('Truth')
ax.set_ylabel('Prediction')
ax.set_xticklabels(['spam', 'ham'])
ax.set_yticklabels(['spam', 'ham'])
plt.show()
