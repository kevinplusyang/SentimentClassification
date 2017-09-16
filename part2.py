import part1
import numpy as np


# Replace words with count 1 to unk
def handle_uni_unknown(fileName):
    un_handled_dic, word_count = part1.uniGram(fileName)
    handled_dic = {}
    handled_dic["unk"] = 0
    for word in un_handled_dic:
        if un_handled_dic[word] == 1:
            handled_dic["unk"] +=1
        else:
            handled_dic[word] = un_handled_dic[word]

    for word in handled_dic:
        handled_dic[word] = float(handled_dic[word]) / word_count

    return handled_dic


def handle_bi_unknown(fileName):
    # read file
    with open(fileName) as f:
        Sentences = []
        for line in f:
            Sentences.append(line)

    bigram_count_dic = {}
    w_first_count = {}
    bigram_dic = {}

    uni_possibility_dic = handle_uni_unknown(fileName)


    for line in Sentences:
        words = line.split()
        n = len(words)
        for i in range(n):
            if i == 0:
                w_first = '*'
            else:
                if words[i - 1].lower() not in uni_possibility_dic:
                    w_first = "unk"

                else:
                    w_first = words[i - 1].lower()

            if words[i].lower() not in uni_possibility_dic:
                w_second = "unk"
            else:
                w_second = words[i].lower()

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



def emotion(sentence, pos_bigram_count_dic, pos_w_first_count, neg_bigram_count_dic, neg_w_first_count, pos_uni_dic, neg_uni_dic, lamada):
    words = sentence.split()
    n = len(words)

    pos_possibility = 0
    neg_possibility = 0

    for i in range(n):
        if i == 0:
            w_first = '*'
        else:
            if words[i - 1].lower() not in pos_uni_dic:
                w_first = "unk"
            else:
                w_first = words[i - 1].lower()

        if words[i].lower() not in pos_uni_dic:
            w_second = "unk"
        else:
            w_second = words[i].lower()

        if w_second in pos_bigram_count_dic[w_first]:
            temp_possibility = float(pos_bigram_count_dic[w_first][w_second] + lamada) / (pos_w_first_count[w_first] + len(pos_uni_dic) * lamada)
        else:
            temp_possibility = float(lamada) / (pos_w_first_count[w_first] + len(pos_uni_dic) * lamada)

        pos_possibility += np.log(temp_possibility)

    for i in range(n):
        if i == 0:
            w_first = '*'
        else:
            if words[i - 1].lower() not in neg_uni_dic:
                w_first = "unk"
            else:
                w_first = words[i - 1].lower()

        if words[i].lower() not in neg_uni_dic:
            w_second = "unk"
        else:
            w_second = words[i].lower()

        if w_second in neg_bigram_count_dic[w_first]:
            temp_possibility = float(neg_bigram_count_dic[w_first][w_second] + lamada) / (
            neg_w_first_count[w_first] + len(neg_uni_dic) * lamada)
        else:
            temp_possibility = float(lamada) / (neg_w_first_count[w_first] + len(neg_uni_dic) * lamada)

        neg_possibility += np.log(temp_possibility)


    if neg_possibility > pos_possibility:
        return 1
    else:
        return 0



def main():
    neg_bigram_count_dic, neg_w_first_count = handle_bi_unknown("./SentimentDataset/Train/neg.txt")
    pos_bigram_count_dic, pos_w_first_count = handle_bi_unknown("./SentimentDataset/Train/pos.txt")
    pos_uni_dic = handle_uni_unknown("./SentimentDataset/Train/pos.txt")
    neg_uni_dic = handle_uni_unknown("./SentimentDataset/Train/neg.txt")
    print emotion("this movie is good .", pos_bigram_count_dic, pos_w_first_count, neg_bigram_count_dic, neg_w_first_count, pos_uni_dic,
            neg_uni_dic, 0.5)



main()



def splitText(fileName, outputName_train, outputName_test):
    with open(fileName) as f:
        Sentences = []
        for line in f:
            Sentences.append(line)
        n = len(Sentences)
        train=[]
        test=[]
        for i in range(n):
            if i % 4 == 0:
                test.append(Sentences[i])
            else:
                train.append(Sentences[i])

        with open(outputName_test, 'w') as file:
            for sen in test:
                file.write(sen)

        with open(outputName_train, 'w') as file:
            for sen in train:
                file.write(sen)

splitText("./SentimentDataset/Train/neg.txt", "./SentimentDataset/Train/neg_train.txt", "./SentimentDataset/Train/neg_test.txt")
splitText("./SentimentDataset/Train/pos.txt", "./SentimentDataset/Train/pos_train.txt", "./SentimentDataset/Train/pos_test.txt")


def findLamada():
    neg_bigram_count_dic, neg_w_first_count = handle_bi_unknown("./SentimentDataset/Train/neg_train.txt")
    pos_bigram_count_dic, pos_w_first_count = handle_bi_unknown("./SentimentDataset/Train/pos_train.txt")
    pos_uni_dic = handle_uni_unknown("./SentimentDataset/Train/pos_train.txt")
    neg_uni_dic = handle_uni_unknown("./SentimentDataset/Train/neg_train.txt")

    lamada_list = [0.000000001,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for lamada in lamada_list:
        pos_count = 0
        pos_total = 0
        with open("./SentimentDataset/Train/neg_train.txt") as f:
            for line in f:
                pos_total += 1
                if emotion(line, pos_bigram_count_dic, pos_w_first_count, neg_bigram_count_dic,
                              neg_w_first_count, pos_uni_dic,
                              neg_uni_dic, lamada) == 1:
                    pos_count += 1

        print "*******"
        print lamada
        print pos_total
        print pos_count
        print float(pos_count) / pos_total


findLamada()

handle_bi_unknown("./SentimentDataset/Train/neg.txt")
