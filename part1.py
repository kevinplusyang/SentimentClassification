

import json
import random

# Calculate uniGram function
def uniGram(fileName):
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
    #for word in uniGram_count_dic:
        #uniGram_count_dic[word] = float(uniGram_count_dic[word]) / wordCount

    #with open(outputName, 'w') as file:
        #file.write(json.dumps(uniGram_count_dic))

    #for word in uniGram_count_dic:
        #uniGram_count_dic[word] = float(uniGram_count_dic[word]) * wordCount
    return uniGram_count_dic, wordCount

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
        for i in range(n):
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
        if uniGram_count_dic[word] >= rand:
            targetWord = word
            break
    return targetWord


# def uniGenerator(fileName, outputName):
#     uniGram_count_dic = uniGram(fileName, outputName)
#
#     carry = 0
#     for word in uniGram_count_dic:
#         uniGram_count_dic[word] += carry
#         carry = uniGram_count_dic[word]
#
#     sentence = ""
#     while True:
#         targetWord = popWord(uniGram_count_dic, carry)
#         sentence += targetWord + " "
#         if targetWord == "." or targetWord == "?" or targetWord == "!":
#             break
#
#     print "Unigram Generator: "
#     print sentence.capitalize()

def biGenerator(fileName, outputName, targetWord_input = None):
    bigram_count_dic, w_first_count = biGram(fileName, outputName)
    w_first_dic = bigram_count_dic['*']
    #generator without giving seed
    if targetWord_input == None:
        carry = 0
        for word in w_first_dic:    
            w_first_dic[word] += carry
            carry = w_first_dic[word]
        targetWord = popWord(w_first_dic, carry)
        sentence = "" + targetWord + " "
    #generator with giving seed   
    else:
        Words = targetWord_input.split()
        #only use the latest word in seed to do bigram generating
        targetWord = Words[-1].lower()
        sentence = "" + targetWord_input + " "

    while True:
        if targetWord not in bigram_count_dic:
            break
        second_word_dic = bigram_count_dic[targetWord].copy()
        carry = 0
        for word in second_word_dic:
            second_word_dic[word] += carry
            carry = second_word_dic[word]
        targetWord = popWord(second_word_dic, carry)
        sentence += targetWord + " "
        if targetWord == "." or targetWord == "?" or targetWord == "!":
            break

    print "Bigram Generator: "
    print sentence.capitalize()

# #negative unigram generator
# uniGenerator('./SentimentDataset/Train/neg.txt','unigram_output_neg.txt' )
# #negative bigram generator without seed
# biGenerator('./SentimentDataset/Train/neg.txt','unigram_output_neg.txt')
# #negative bigram generator with seed 'I have'
# biGenerator('./SentimentDataset/Train/neg.txt','unigram_output_neg.txt','I have')
# #positive unigram generator
# uniGenerator('./SentimentDataset/Train/pos.txt','unigram_output_pos.txt' )
# #positive unigram generator without seed
# biGenerator('./SentimentDataset/Train/pos.txt','unigram_output_pos.txt')
# #positive bigram generator with seed 'I have'
# biGenerator('./SentimentDataset/Train/pos.txt','unigram_output_pos.txt', 'I have')


