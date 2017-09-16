import part1


# Replace words with count 1 to unk
def handle_unknown(fileName):
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




def bi_smoothing(fileName):
    unsmoothed_dic = part1.biGram('./SentimentDataset/Dev/pos.txt','unigram_output_pos.txt')
