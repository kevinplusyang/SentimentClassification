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



def emotion(sentence, pos_bigram_count_dic, pos_w_first_count, neg_bigram_count_dic, neg_w_first_count, pos_uni_dic, neg_uni_dic, pos_train, neg_train,lamada1, lamada2):
    words = sentence.split()
    n = len(words)

    pos_possibility = np.log(float(pos_train) / (pos_train + neg_train) )
    neg_possibility = np.log(float(neg_train) / (pos_train + neg_train) )

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
            temp_possibility = float(pos_bigram_count_dic[w_first][w_second] + lamada1) / (pos_w_first_count[w_first] + len(pos_uni_dic) * lamada1)
        else:
            temp_possibility = float(lamada1) / (pos_w_first_count[w_first] + len(pos_uni_dic) * lamada1)

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
            temp_possibility = float(neg_bigram_count_dic[w_first][w_second] + lamada2) / (
            neg_w_first_count[w_first] + len(neg_uni_dic) * lamada2)
        else:
            temp_possibility = float(lamada2) / (neg_w_first_count[w_first] + len(neg_uni_dic) * lamada2)

        neg_possibility += np.log(temp_possibility)


    if neg_possibility > pos_possibility:
        return 1
    else:
        return 0


#
# def main():
#     neg_bigram_count_dic, neg_w_first_count = handle_bi_unknown("./SentimentDataset/Train/neg.txt")
#     pos_bigram_count_dic, pos_w_first_count = handle_bi_unknown("./SentimentDataset/Train/pos.txt")
#     pos_uni_dic = handle_uni_unknown("./SentimentDataset/Train/pos.txt")
#     neg_uni_dic = handle_uni_unknown("./SentimentDataset/Train/neg.txt")
#     print emotion("this movie is good .", pos_bigram_count_dic, pos_w_first_count, neg_bigram_count_dic, neg_w_first_count, pos_uni_dic,
#             neg_uni_dic, 0.5)
#
#
#
# main()



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
    return len(train)

neg_train = splitText("./SentimentDataset/Train/neg.txt", "./SentimentDataset/Train/neg_train.txt", "./SentimentDataset/Train/neg_test.txt")
pos_train = splitText("./SentimentDataset/Train/pos.txt", "./SentimentDataset/Train/pos_train.txt", "./SentimentDataset/Train/pos_test.txt")


def findLamada():
    neg_bigram_count_dic, neg_w_first_count = handle_bi_unknown("./SentimentDataset/Train/neg_train.txt")
    pos_bigram_count_dic, pos_w_first_count = handle_bi_unknown("./SentimentDataset/Train/pos_train.txt")
    pos_uni_dic = handle_uni_unknown("./SentimentDataset/Train/pos_train.txt")
    neg_uni_dic = handle_uni_unknown("./SentimentDataset/Train/neg_train.txt")

    maxProbility = 0
    maxLamada1 = 0
    maxLamada2 = 0
    i = 0.01
    j = 0.01
    while (i < 1):
        while (j < 1):
            pos_count = 0
            pos_total = 0
            with open("./SentimentDataset/Train/pos_test.txt") as f:
                for line in f:
                    pos_total += 1
                    if emotion(line, pos_bigram_count_dic, pos_w_first_count, neg_bigram_count_dic,
                                  neg_w_first_count, pos_uni_dic,
                                  neg_uni_dic, pos_train, neg_train,i, j) == 0:
                        pos_count += 1

            neg_count = 0
            neg_total = 0
            with open("./SentimentDataset/Train/neg_test.txt") as f:
                for line in f:
                    neg_total += 1
                    if emotion(line, pos_bigram_count_dic, pos_w_first_count, neg_bigram_count_dic,
                               neg_w_first_count, pos_uni_dic,
                               neg_uni_dic, pos_train, neg_train, i, j) == 1:
                        neg_count += 1

            print "*******"
            print i
            print j
            print float(pos_count + neg_count) / (pos_total + neg_total)
            if float(pos_count + neg_count) / (pos_total + neg_total) > maxProbility:
                maxProbility = float(pos_count + neg_count) / (pos_total + neg_total)
                maxLamada1 = i
                maxLamada2 = j
            j += 0.01
        j = 0.01
        i += 0.01
        print i

    print "--------------"
    print maxLamada1
    print maxLamada2
    print maxProbility

findLamada()

handle_bi_unknown("./SentimentDataset/Train/neg.txt")
