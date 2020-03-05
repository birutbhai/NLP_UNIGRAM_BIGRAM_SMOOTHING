import warnings
warnings.filterwarnings("ignore")
import sys
import os
import argparse
target = "The standard Turbo engine is hard to work"
in_file="NLP6320_POSTaggedTrainingSet-Unix.txt"
total_words = 0
word_dict = dict()
bigram_dict= dict()
vocab_count = 0
def parse_file():
    global vocab_count
    global total_words
    with open(in_file) as fp:
        lines = fp.readlines()
    for line in lines:
        words = line.split()
        for word in words:
            tokens = word.split("_")
            token = tokens[0].lower()
            val = 1
            if token in word_dict:
                val += word_dict[token]
            else:
                vocab_count += 1
            word_dict.update({token:val})
            total_words += 1
def create_bigram_model():
    global bigram_dict
    with open(in_file) as fp:
        lines = fp.readlines()
        for line in lines:
            words = line.split()
            prev = ""
            for word in words:
                tokens = word.split("_")
                token = tokens[0].lower()
                val = 1
                if prev in bigram_dict:
                    if prev is "":
                        val = word_dict[token]
                    if token in bigram_dict[prev]:
                        val += bigram_dict[prev][token]
                else:
                    if prev is "":
                        val = word_dict[token]
                    bigram_dict.update({prev:dict()})
                bigram_dict[prev].update({token:val})
                prev = token
    o_file = open("bigram_model_no_smoothing.txt", "w")
    for key, value in bigram_dict.items():
        o_file.write("Previous word \"" + key + "\"\n")
        o_file.write(str(value))
        o_file.write("\n")
    o_file.close()

def unigram():
    print("Probability computed with the Unigram Model")
    print("Total words:", total_words)
    words = target.split()
    prob_sen = 1
    o_file = open("unigram_model.txt", "w")
    o_file.write("Total words:"+str(total_words)+"\n")
    for key, value in word_dict.items():
        o_file.write(key+" : " + str(value) +"\n")
    o_file.close()
    for orig_word in words:
        count = 0
        word = orig_word.lower()
        if word in word_dict:
            count = word_dict[word]
        print("Count of "+ word + " = " + str(count))
        prob = float(count)/float(total_words)
        print("Probability of "+ word + " = " + str(prob))
        prob_sen *= prob
    print("Probability of the input sentence: ", prob_sen)

def bigram_ns():
    create_bigram_model()
    print("Probability computed with the bigram model without smoothing")
    words = target.split()
    prob_sen = 1
    prev = ""
    for orig_word in words:
        count = 0
        word = orig_word.lower()
        if prev in bigram_dict:
            if word in bigram_dict[prev]:
                count = bigram_dict[prev][word]
        if prev is "":
             count = word_dict[word]
             print("Count of "+ word + " = " + str(count))
        else:
            print("Count of "+ word + "|"+prev+" = " + str(count))
        prev_count = 1
        if prev in word_dict:
            prev_count = word_dict[prev]
        elif prev is "":
            prev_count = total_words
        if prev is "":
            print("Total words:", total_words)
        else:
            print("Count of "+prev+" = " + str(prev_count))
        prob = float(count)/float(prev_count)
        if prev is "":
            print("Probability of "+ word + " = " + str(prob))
        else:
            print("Probability of "+ word + "|"+prev+" = " + str(prob))
        prob_sen *= prob
        prev = word
    print("Probability of the input sentence: ", prob_sen)

    return

def bigram_ao():
    create_bigram_model()
    print("Vocabulary count:"+str(vocab_count))
    ao_sm = dict()
    for key, value in bigram_dict.items():
        for sub_key, val in value.items():
            if key is "":
                #sm_val = float(val + 1)/float(vocab_count)
                sm_val = float(word_dict[sub_key])/float(total_words)
            else:
                sm_val = float(val + 1)/float(word_dict[key]+vocab_count)
            if key not in ao_sm:
                ao_sm.update({key:dict()})
            ao_sm[key].update({sub_key:sm_val})
    o_file = open("bigram_model_add_one_smoothing.txt", "w")
    for key, value in ao_sm.items():
        o_file.write("Previous word \"" + key + "\"\n")
        o_file.write(str(value))
        o_file.write("\n")

    print("Probability computed with the bigram model with add-one smoothing")
    words = target.split()
    prob_sen = 1
    prev = ""
    for orig_word in words:
        prob = 0
        word = orig_word.lower()
        if prev is "":
            if word in word_dict:
                prob = float(word_dict[word])/float(total_words)
        elif prev in ao_sm:
            if word in ao_sm[prev]:
                prob = ao_sm[prev][word]
        if prob is 0:
            prv_cnt = 0 
            if prev in word_dict:
                prv_cnt = word_dict[prev]
            prob = float(1)/float(vocab_count+prv_cnt)
        if prev is "":
             print("Probability of "+ word + " = " + str(prob))
        else:
            print("Probability of "+ word + "|"+prev+" = " + str(prob))
        prob_sen *= prob
        prev = word
    print("Probability of the input sentence: ", prob_sen)

    return

def bigram_gt():
    create_bigram_model()
    sm_gt = dict()
    bigram_count = 0
    dist_b = 0
    mod_prob = dict()
    for key, value in bigram_dict.items():
        for sub_key, val in value.items():
            if key is not "":
                if val not in sm_gt:
                    sm_gt.update({val:0})
                count = sm_gt[val] + 1
                sm_gt.update({val:count})
                bigram_count += val
                dist_b += 1
    for key, val in sm_gt.items():
        nxt_bukt = key + 1
        nb_c = 0
        if nxt_bukt in sm_gt:
            nb_c = sm_gt[nxt_bukt]
        prob = float(nxt_bukt * nb_c)/float(val * bigram_count)
        mod_prob.update({key:prob})
    o_file = open("bigram_model_good_turing_smoothing.txt", "w")
    for key, value in sm_gt.items():
        o_file.write("Bucket ID:\"" + str(key) + "\"\n")
        o_file.write("Types: "+str(value)+"\n")
        o_file.write("Probability: "+str(mod_prob[key]))
        o_file.write("\n")
    print("Probability computed with the bigram model with Good-Turing discounting based smoothing")
    words = target.split()
    prob_sen = 1
    prev = ""
    for orig_word in words:
        prob = 0
        word = orig_word.lower()
        new = True
        if prev is "":
            if word in word_dict:
                prob = float(word_dict[word])/float(total_words)
                new = False
        elif prev in bigram_dict:
            if word in bigram_dict[prev]:
                prob = mod_prob[bigram_dict[prev][word]]
                new = False
        if new is True:
            if 1 in sm_gt:
                prob = float(sm_gt[1])/float(bigram_count)
        if prev is "":
             print("Probability of "+ word + " = " + str(prob))
        else:
            print("Probability of "+ word + "|"+prev+" = " + str(prob))
        prob_sen *= prob
        prev = word
    print("Probability of the input sentence: ", prob_sen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ngram_type", type=str, help="Type of the n gram calculation", nargs='?',
                        default="UNIGRAM", const="UNIGRAM")
    args = parser.parse_args()
    type = args.ngram_type
    parse_file()
    if type == "UNIGRAM":
        unigram()
    elif type == "BIGRAM_NS":
         bigram_ns()
    elif type == "BIGRAM_AO":
         bigram_ao()
    elif type == "BIGRAM_GT":
         bigram_gt()
    else:
        print("Invalid option")


