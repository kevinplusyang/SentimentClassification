with open("pos.txt") as f:
    words = []
    for line in f:
        for word in line.split():
            words.append(word)
    with open("pos_preprocess.txt",'w') as file:
        for word in words:
            file.write(word + " ")
        
        
            