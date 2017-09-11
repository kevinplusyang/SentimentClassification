import json
import random

# Calculate uniGram function
def uniGram( fileName, outputName):
    # Open file
    with open(fileName) as f:
        Sentences = []
        for line in f:
            Sentences.append(line)

    uniGram_count_dic = {}
    wordCount = 0
    for Sentence in Sentences:
        words = Sentence.split()
        for word in words:
            wordCount += 1
            word = word.lower()
            if word not in uniGram_count_dic:
                uniGram_count_dic[word] = 1
            else:
                uniGram_count_dic[word] += 1

    # Get the possibility of each word
    for word in uniGram_count_dic:
        uniGram_count_dic[word] = float(uniGram_count_dic[word]) / wordCount

    with open(outputName, 'w') as file:
        file.write(json.dumps(uniGram_count_dic))

    for word in uniGram_count_dic:
        uniGram_count_dic[word] = float(uniGram_count_dic[word]) * wordCount
    return uniGram_count_dic

def biGram(fileName, outputName):
    #read file
    with open(fileName) as f:
        Sentences = []
        for line in f:
            Sentences.append(line)

    bigram_count_dic = {}
    w_first_count = {}
    bigram_dic = {}
    
    for line in Sentences:
        words = line.split()
        n = len(words)
        for i in range(n-1):
            if i == 0 :
                w_first = '*'
            else:
                w_first = words[i-1].lower()

            w_second = words[i].lower()

            #creat w_first's key and value
            if w_first not in bigram_count_dic:
                #count w_second
                w_second_count = {}
                w_second_count[w_second] = 1
                #add w_fisrt to dic, assaign value to w_second_count dic
                bigram_count_dic[w_first] = w_second_count
                w_first_count[w_first] = 1

            else:
                w_first_count[w_first] += 1
                if w_second not in bigram_count_dic[w_first]:
                    bigram_count_dic[w_first][w_second] = 1
                else:
                    bigram_count_dic[w_first][w_second] += 1

    #get possibility dictionary
    for k in bigram_count_dic:
        n = w_first_count[k]
        w_second = {}
        for w in bigram_count_dic[k]:
            w_second[w]= float(bigram_count_dic[k][w])/n
        bigram_dic[k] = w_second   

    with open(outputName, 'w') as file:
        file.write(json.dumps(bigram_dic))

    return bigram_count_dic, w_first_count

def popWord(uniGram_count_dic, carry):
    rand = random.randint(0, carry)
    for word in uniGram_count_dic:
        if uniGram_count_dic[word] > rand:
            targetWord = word
            break
    return targetWord


def uniGenerator():
    uniGram_count_dic = uniGram('./SentimentDataset/Train/neg.txt', 'unigram_output_neg.txt')

    print uniGram_count_dic
    carry = 0
    for word in uniGram_count_dic:
        uniGram_count_dic[word] += carry
        carry = uniGram_count_dic[word]

    sentence = ""
    while True:
        targetWord = popWord(uniGram_count_dic, carry)
        sentence += targetWord + " "
        if targetWord == "." or targetWord == "?" or targetWord == "!":
            break

    print sentence.capitalize()

def biGenerator():
    bigram_count_dic, w_first_count = biGram('./SentimentDataset/Dev/neg.txt','bigram_output_neg.txt' )
    w_first_dic = bigram_count_dic['*']
    print w_first_dic
    carry = 0
    for word in w_first_dic:    
        w_first_dic[word] += carry
        carry = w_first_dic[word]
    targetWord = popWord(w_first_dic, carry)
    print targetWord
    print w_first_dic
    
        
    
    
    
    


uniGram( './SentimentDataset/Dev/neg.txt','unigram_output_neg.txt' )
biGram('./SentimentDataset/Dev/neg.txt','bigram_output_neg.txt' )
uniGenerator()

