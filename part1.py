# Calculate uniGram function
def uniGram( fileName ):
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

    for word in uniGram_count_dic:
        uniGram_count_dic[word] = float(uniGram_count_dic[word]) / wordCount

    print uniGram_count_dic
    print len(uniGram_count_dic)


uniGram( './SentimentDataset/Dev/neg.txt' )
