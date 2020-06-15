import numpy as np
import re



# Assigns sentiment of positive or negative for a given review (given as a string of text)
# using a bag of words approach to train a naive bayes classifier.


def Bucket(pos,neg):
    # Returns a list of postive and a list of negative training data and a list of positive and 
    # a list of negative test data.

    neg_dat = []
    pos_dat = []

    for lines in open(pos, "r").readlines():
        line = lines.split('\t')
        pos_dat.append(line)
        
    for lines in open(neg, "r").readlines():
        line = lines.split('\t')
        neg_dat.append(line)
    
    trainpos_dat, testpos_dat = pos_dat[:84], pos_dat[84:]
    trainneg_dat, testneg_dat = neg_dat[:84], neg_dat[84:]
    
    return trainpos_dat, trainneg_dat, testpos_dat, testneg_dat





def SingleBucket(pos,neg):
    # Returns a list of positive and a list of negative training data.
    
    neg_dat = []
    pos_dat = []

    for lines in open(pos, "r").readlines():
        line = lines.split('\t')
        pos_dat.append(line)
        
    for lines in open(neg, "r").readlines():
        line = lines.split('\t')
        neg_dat.append(line)
    
    return pos_dat, neg_dat





def Vocab(pos,neg):
    # Returns a positive vocab and a negative vocab along with word count totals.

    pos_vocab = {}
    neg_vocab = {}
    pv_tot, nv_tot = 0,0
    
    for rev in pos:
        words = re.findall(r"[\w']+|[.,!?;]", rev[1])
        
        for word in words:
            pv_tot += 1
            
            if(word.lower() in pos_vocab.keys()):
                pos_vocab[word.lower()] += 1
            else:
                pos_vocab[word.lower()] = 1
    
    for rev in neg:
        words = re.findall(r"[\w']+|[.,!?;]", rev[1])
        
        for word in words:
            nv_tot += 1
            
            if(word.lower() in neg_vocab.keys()):
                neg_vocab[word.lower()] += 1
            else:
                neg_vocab[word.lower()] = 1
    
    return pos_vocab, neg_vocab, pv_tot, nv_tot





def TrainNaiveBayes(pos, neg):
    #Returns logprior, likelihood, positive vocab, and negative vocab.
    
    log_p = {}
    ploglike, nloglike = {}, {}
    
    pos_v, neg_v, pv_tot, nv_tot = Vocab(pos,neg)
    pv = len(pos_v)
    nv = len(neg_v)
    
    num_docs = len(pos) + len(neg)
    log_p["pos"] = np.log(len(pos) / num_docs)
    log_p["neg"] = np.log(len(neg) / num_docs)
    
    for word in pos_v.keys():
        ploglike[word] = np.log( (pos_v[word] + 1) / (pv_tot + pv) )
    for word in neg_v.keys():
        nloglike[word] = np.log( (neg_v[word] + 1) / (nv_tot + nv) )
        
    ploglike["UNK"] = np.log( 2 / (pv_tot + pv) )
    nloglike["UNK"] = np.log( 2 / (nv_tot + nv) )
    
    return log_p, ploglike, nloglike, pos_v, neg_v




def TestNaiveBayes(test_doc,log_p, ploglike, nloglike, pos_v, neg_v):
    # Returns a positive or negative classification for the test_doc, which should be given in
    # the form [[ID],[Review]].
    
    psum = log_p["pos"]
    nsum = log_p["neg"]
    
    words = re.findall(r"[\w']+|[.,!?;]", test_doc[1])
    for word in words:
        if (word.lower() in ploglike.keys()):
            psum += ploglike[word.lower()]
        else:
            psum += ploglike["UNK"]
            
        if (word.lower() in nloglike.keys()):
            nsum += nloglike[word.lower()]
        else:
            nsum += nloglike["UNK"]
    
    if (psum < nsum):           
        reviewclass = "POS"
    else:
        reviewclass = "NEG"
    
    return reviewclass




def NaiveBayesClassifier(training_pos, training_neg, testfile):
    # generates .txt file with assigned classifications to testfile.
    
    trainpos_dat, trainneg_dat = SingleBucket(training_pos,training_neg)
    log_p, ploglike, nloglike, pos_v, neg_v = TrainNaiveBayes(trainpos_dat,trainneg_dat)
    
    tf_dat = []
    for lines in open(testfile, "r").readlines():
        line = lines.split('\t')
        tf_dat.append(line)
    
    output = []
    for review in tf_dat:
        output.append(TestNaiveBayes(review,log_p, ploglike, nloglike, pos_v, neg_v))
    
    for i in range(0,len(tf_dat)):
        tf_dat[i] = str(tf_dat[i][0])+'\t'+str(output[i])+'\n'

    # Write text file
    out = open('tedder-mark-assgn3-out.txt','w')
    for line in tf_dat:
        out.write(line)
    out.close()

    
    
# Generates sentiment reviews and writes output text file with predicted sentiments.

train_pos = 'hotelNegT-train.txt'
train_neg = 'hotelPosT-train.txt'
test = 'testfile.txt'

NaiveBayesClassifier(train_pos, train_neg, test)