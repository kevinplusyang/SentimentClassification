import json
import random
import numpy as np

def uniGramCount(fileName):
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

    return uniGram_count_dic, wordCount


def uniGramCountForTwoFiles(fileName1, fileName2):
    # Open file
    with open(fileName1) as f:
        Sentences = []
        for line in f:
            Sentences.append(line)

    with open(fileName2) as f:
        Sentences2 = []
        for line2 in f:
            Sentences.append(line2)

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

    return uniGram_count_dic, wordCount


def handleUnk(fileName1, fileName2):
    pos_Dic_No_Unk, pos_Count_No_Unk = uniGramCount(fileName1)
    neg_Dic_No_Unk, neg_Count_No_Unk = uniGramCount(fileName2)
    combination_Dic, combination_Count = uniGramCountForTwoFiles(fileName1, fileName2)

    pos_Dic_Unk = {}
    pos_Dic_Unk["unk"] = 0

    neg_Dic_Unk = {}
    neg_Dic_Unk["unk"] = 0

    combination_Dic_Unk = {}
    combination_Dic_Unk["unk"] = 0

    for word in pos_Dic_No_Unk:
        if pos_Dic_No_Unk[word] == 1:
            if combination_Dic[word] == 1:
                pos_Dic_Unk["unk"] += 1
            else:
                pos_Dic_Unk[word] = pos_Dic_No_Unk[word]
        else:
            pos_Dic_Unk[word] = pos_Dic_No_Unk[word]

    for word in neg_Dic_No_Unk:
        if neg_Dic_No_Unk[word] == 1:
            if combination_Dic[word] == 1:
                neg_Dic_Unk["unk"] += 1
            else:
                neg_Dic_Unk[word] = neg_Dic_No_Unk[word]
        else:
            neg_Dic_Unk[word] = neg_Dic_No_Unk[word]



    for word in combination_Dic:
        if combination_Dic[word] == 1:
            combination_Dic_Unk["unk"] +=1
        else:
            combination_Dic_Unk[word] = combination_Dic[word]

    return pos_Dic_Unk, neg_Dic_Unk, combination_Dic_Unk


def biGram(fileName, word_dic):
    # read file
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
            if i == 0:
                w_first = '*'
            else:
                w_first = words[i - 1].lower()
                if w_first not in word_dic:
                    w_first = "unk"

            w_second = words[i].lower()
            if w_second not in word_dic:
                w_second = "unk"

            # creat w_first's key and value
            if w_first not in bigram_count_dic:
                # count w_second
                w_second_count = {}
                w_second_count[w_second] = 1
                # add w_fisrt to dic, assaign value to w_second_count dic
                bigram_count_dic[w_first] = w_second_count
                w_first_count[w_first] = 1

            else:
                w_first_count[w_first] += 1
                if w_second not in bigram_count_dic[w_first]:
                    bigram_count_dic[w_first][w_second] = 1
                else:
                    bigram_count_dic[w_first][w_second] += 1

    # get possibility dictionary
    for k in bigram_count_dic:
        n = w_first_count[k]
        w_second = {}
        for w in bigram_count_dic[k]:
            w_second[w] = float(bigram_count_dic[k][w]) / n
        bigram_dic[k] = w_second

    return bigram_count_dic, w_first_count




def biGram_Possibility_Unsmooth(bigram_count_dic, w_first_count,  word_dic, sentence):

    possibility = 1
    words = sentence.split()
    n = len(words)
    for i in range(n):
        if i == 0:
            w_first = '*'
        else:
            w_first = words[i - 1].lower()
            if w_first not in word_dic:
                w_first = "unk"

        w_second = words[i].lower()
        if w_second not in word_dic:
            w_second = "unk"

        if w_second not in bigram_count_dic[w_first]:
            possibility = 0
        else:
            possibility *= float(bigram_count_dic[w_first][w_second]) / w_first_count[w_first]

    return possibility





def biGram_Possibility_Smooth(bigram_count_dic, w_first_count,  word_dic, combination_Dic_Unk, sentence, lamada):

    possibility = np.log(1)
    words = sentence.split()
    n = len(words)
    for i in range(n):
        if i == 0:
            w_first = '*'
        else:
            w_first = words[i - 1].lower()
            if w_first not in word_dic:
                w_first = "unk"

        w_second = words[i].lower()
        if w_second not in word_dic:
            w_second = "unk"

        if w_first in bigram_count_dic:
            if w_second not in bigram_count_dic[w_first]:
                possibility += np.log(float(0 + lamada) / (w_first_count[w_first] + lamada * len(combination_Dic_Unk)))
            else:
                possibility += np.log(float(bigram_count_dic[w_first][w_second] + lamada) / (
                w_first_count[w_first] + lamada * len(combination_Dic_Unk)))

    return possibility

def uniGram_Possibility_Smooth(Dic_Unk, sentence, lamada):
    words = sentence.split()
    possibility = 0

    total_word_count = 0
    word_kind = 0
    for word in Dic_Unk:
        word_kind += 1
        total_word_count += Dic_Unk[word]



    for word in words:
        if word not in Dic_Unk:
            word = "unk"
        possibility += np.log(float(Dic_Unk[word] + lamada) / (total_word_count + lamada * word_kind) )

    return possibility


def sentence_Count(fileName):
    count = 0
    with open(fileName) as f:
        for line in f:
            count += 1
    return count



def findLamada(fileName1, fileName2):
    pos_Dic_Unk, neg_Dic_Unk, combination_Dic_Unk = handleUnk(fileName1, fileName2)
    print pos_Dic_Unk
    print neg_Dic_Unk
    print combination_Dic_Unk

    pos_bigram_count_dic, pos_w_first_count = biGram(fileName1, pos_Dic_Unk)
    neg_bigram_count_dic, neg_w_first_count = biGram(fileName2, neg_Dic_Unk)




    # pos_possibility = biGram_Possibility_Unsmooth(bigram_count_dic, w_first_count, neg_Dic_Unk, "Ming is beautiful too .")

    pos_sentence_count = sentence_Count(fileName1)
    neg_sentence_count = sentence_Count(fileName2)
    print pos_sentence_count
    print neg_sentence_count
    pos_weight = np.log(float(pos_sentence_count) / (pos_sentence_count + neg_sentence_count))
    neg_weight = np.log(float(neg_sentence_count) / (pos_sentence_count + neg_sentence_count))


    lamada = 0.01
    maxAcc = 0
    maxLamada = 0

    while (lamada < 0.1) :
        pos_count = 0
        pos_total = 0
        with open('./SentimentDataset/Train/pos_test.txt') as f:
            for line in f:
                pos_total += 1
                pos_possibility = biGram_Possibility_Smooth(pos_bigram_count_dic, pos_w_first_count, pos_Dic_Unk,
                                                            combination_Dic_Unk, line, lamada)
                neg_possibility = biGram_Possibility_Smooth(neg_bigram_count_dic, neg_w_first_count, neg_Dic_Unk,
                                                            combination_Dic_Unk, line, lamada)
                pos = pos_possibility + pos_weight
                neg = neg_possibility + neg_weight
                if (pos > neg):
                    pos_count += 1

        neg_count = 0
        neg_total = 0

        with open('./SentimentDataset/Train/neg_test.txt') as f:
            for line in f:
                neg_total += 1
                pos_possibility = biGram_Possibility_Smooth(pos_bigram_count_dic, pos_w_first_count, pos_Dic_Unk,
                                                            combination_Dic_Unk, line, lamada)
                neg_possibility = biGram_Possibility_Smooth(neg_bigram_count_dic, neg_w_first_count, neg_Dic_Unk,
                                                            combination_Dic_Unk, line, lamada)
                pos = pos_possibility + pos_weight
                neg = neg_possibility + neg_weight
                if (neg > pos):
                    neg_count += 1

        print "====================="
        print "Lamada"
        print lamada
        print "----Pos Accuracy----"
        print float(pos_count) / pos_total

        print "----Neg Accuracy----"
        print float(neg_count) / neg_total

        print "----Total Accuracy----"
        print float(pos_count + neg_count) / (pos_total + neg_total)

        if float(pos_count + neg_count) / (pos_total + neg_total) > maxAcc:
            maxAcc = float(pos_count + neg_count) / (pos_total + neg_total)
            maxLamada = lamada

        lamada += 0.01



    print maxLamada


def main(fileName1, fileName2):
    pos_Dic_Unk, neg_Dic_Unk, combination_Dic_Unk = handleUnk(fileName1, fileName2)
    print pos_Dic_Unk
    print neg_Dic_Unk
    print combination_Dic_Unk

    pos_bigram_count_dic, pos_w_first_count = biGram(fileName1, pos_Dic_Unk)
    neg_bigram_count_dic, neg_w_first_count = biGram(fileName2, neg_Dic_Unk)




    # pos_possibility = biGram_Possibility_Unsmooth(bigram_count_dic, w_first_count, neg_Dic_Unk, "Ming is beautiful too .")

    pos_sentence_count = sentence_Count(fileName1)
    neg_sentence_count = sentence_Count(fileName2)
    print pos_sentence_count
    print neg_sentence_count
    pos_weight = np.log(float(pos_sentence_count) / (pos_sentence_count + neg_sentence_count))
    neg_weight = np.log(float(neg_sentence_count) / (pos_sentence_count + neg_sentence_count))


    lamada = 0.009
    i= 1
    with open('./SentimentDataset/Test/test.txt') as f:
        for line in f:
            # print i

            pos_possibility = biGram_Possibility_Smooth(pos_bigram_count_dic, pos_w_first_count, pos_Dic_Unk,
                                                            combination_Dic_Unk, line, lamada)
            neg_possibility = biGram_Possibility_Smooth(neg_bigram_count_dic, neg_w_first_count, neg_Dic_Unk,
                                                            combination_Dic_Unk, line, lamada)
            pos = pos_possibility + pos_weight
            neg = neg_possibility + neg_weight
            i += 1
            if (pos > neg):
                print 0
            else:
                print 1


def perp(fileName1, fileName2, targetFile1, targetFile2):
    pos_Dic_Unk, neg_Dic_Unk, combination_Dic_Unk = handleUnk(fileName1, fileName2)
    pos_bigram_count_dic, pos_w_first_count = biGram(fileName1, pos_Dic_Unk)
    neg_bigram_count_dic, neg_w_first_count = biGram(fileName2, neg_Dic_Unk)


    pos_N = 0
    with open('./SentimentDataset/Dev/pos.txt') as f:
        for line in f:
            words = line.split()
            pos_N += len(words)
    print pos_N

    neg_N = 0
    with open('./SentimentDataset/Dev/neg.txt') as f:
        for line in f:
            words = line.split()
            neg_N += len(words)
    print neg_N



    pos_pos_bi_pp = 0
    pos_neg_bi_pp = 0
    lamada = 0.0009

    with open('./SentimentDataset/Dev/pos.txt') as f:
        for line in f:
            pos_pos_bi_pp += biGram_Possibility_Smooth(pos_bigram_count_dic, pos_w_first_count, pos_Dic_Unk,
                                                combination_Dic_Unk, line, lamada)
            pos_neg_bi_pp += biGram_Possibility_Smooth(neg_bigram_count_dic, neg_w_first_count, neg_Dic_Unk,
                                                combination_Dic_Unk, line, lamada)

    print "------POS BI TXT PP---------"
    pos_pos_bi_pp = pos_pos_bi_pp * -1 / pos_N
    pos_neg_bi_pp = pos_neg_bi_pp * -1 / pos_N
    print pos_pos_bi_pp
    print pos_neg_bi_pp

    neg_pos_bi_pp = 0
    neg_neg_bi_pp = 0
    lamada = 0.0009

    with open('./SentimentDataset/Dev/neg.txt') as f:
        for line in f:
            neg_pos_bi_pp += biGram_Possibility_Smooth(pos_bigram_count_dic, pos_w_first_count, pos_Dic_Unk,
                                                combination_Dic_Unk, line, lamada)
            neg_neg_bi_pp += biGram_Possibility_Smooth(neg_bigram_count_dic, neg_w_first_count, neg_Dic_Unk,
                                                combination_Dic_Unk, line, lamada)

    print "------NEG BI TXT PP---------"
    neg_pos_bi_pp = neg_pos_bi_pp * -1 / neg_N
    neg_neg_bi_pp = neg_neg_bi_pp * -1 / neg_N
    print neg_pos_bi_pp
    print neg_neg_bi_pp


    pos_pos_uni_pp = 0
    pos_neg_uni_pp = 0
    lamada = 0.0009

    with open('./SentimentDataset/Dev/pos.txt') as f:
        for line in f:
            pos_pos_uni_pp += uniGram_Possibility_Smooth(pos_Dic_Unk, line, lamada)
            pos_neg_uni_pp += uniGram_Possibility_Smooth(neg_Dic_Unk, line, lamada)

    print "------POS UNI TXT PP---------"
    pos_pos_uni_pp = pos_pos_uni_pp * -1 / pos_N
    pos_neg_uni_pp = pos_neg_uni_pp * -1 / pos_N
    print pos_pos_uni_pp
    print pos_neg_uni_pp


    neg_pos_uni_pp = 0
    neg_neg_uni_pp = 0
    lamada = 0.0009

    with open('./SentimentDataset/Dev/neg.txt') as f:
        for line in f:
            neg_pos_uni_pp += uniGram_Possibility_Smooth(pos_Dic_Unk, line, lamada)
            neg_neg_uni_pp += uniGram_Possibility_Smooth(neg_Dic_Unk, line, lamada)

    print "------NEG UNI TXT PP---------"
    neg_pos_uni_pp = neg_pos_uni_pp * -1 / neg_N
    neg_neg_uni_pp = neg_neg_uni_pp * -1 / neg_N
    print neg_pos_uni_pp
    print neg_neg_uni_pp



# This funciton will find and output the best Lamada value
findLamada('./SentimentDataset/Train/pos_train.txt', './SentimentDataset/Train/neg_train.txt')

# This function can output the result of recognition of each sentece in the test.txt. 0 represent positive
# 1 represent negative
main('./SentimentDataset/Train/pos.txt', './SentimentDataset/Train/neg.txt')

#The function is used to calculate perplexity
perp('./SentimentDataset/Train/pos.txt', './SentimentDataset/Train/neg.txt', './SentimentDataset/Dev/pos.txt', './SentimentDataset/Dev/neg.txt')
