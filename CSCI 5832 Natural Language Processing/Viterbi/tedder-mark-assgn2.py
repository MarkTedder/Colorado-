import numpy as np


# BASELINE ALGORITHM

# Assigns part of speech tags to untagged validation text file using the most common
# tag used for a given word in the training set. Words in the untagged test file
# not found in the training set are assigned "UNK" tag. Predicted tags are then compared
# against the actual tags from the validation set for a baseline accuracy (for comparison
# against the tags predicted using the Viterbi algorithm).


def Bucket(training_file):
    # Returns a list of training data, test data and gold test data.
    # Splits data from file into 90% training data and 10% test data.
    # Gold test data is the test data with its tags still in place.
    
    # Finding total number of sentences in given .txt file, then 
    # calculating the number of sentences in the training data
    sentences = 0

    for lines in open(training_file, "r").readlines():
        line = lines.split('\t')
    
        if (len(line) < 3):
            sentences += 1
        
    train_sents = np.floor(sentences*(0.9)
                           
    # Seperate training data from test data and preserving a gold test
    # (Test data is the gold test data with the POS tags removed)
    sentences = 0 
    training = [] #training data
    test = []     #test data
    goldtest = [] #gold test data

    for lines in open(training_file, "r").readlines():
        line = lines.split('\t')
    
        if (sentences <= train_sents): 
            training.append(line)
        
            if (len(line) < 3):
                sentences += 1
            
        else:
            goldtest.append(lines)
        
            if(len(line)==3):
                test.append([line[0],line[1]])
            else:
                test.append(['\n'])
    
    return training,test,goldtest

                           

                           
def Vocab(training_data):
    # Returns dictionary containing all "words" used and their counts
    # as well as a total word count.

    # "s/e" representes the start / end of sentence                   
    vocab = {'s\e': {'s\e':0} }
    
    for x in training_data:
        
        if(len(x) == 3):
            pos = x[0]
            word = x[1]
            tag = x[2].strip()
            
            if(word in vocab.keys()):
                
                if(tag in vocab[word].keys()):
                    vocab[word][tag] += 1
                
                else:
                    vocab[word][tag] = 1
                    
            else:
                vocab[word]={tag:1}
        
        else:
            vocab['s\e']['s\e'] += 1
            
    count = len(list(vocab))
    
    return vocab, count

                           
                           

def Baseline(training,test):
    # Creates dictionary that contains words with their 
    # tags and how many times each tag for that word occurs.
    
    vocab = Vocab(training)

    # Create dictionary of words and their most frequent tag                        
    most_freq_tags ={}
    
    for word in vocab:
        most_f = 0
    
        for tag in vocab[word]:
        
            if(vocab[word][tag] > most_f):
                most_freq_tag = tag
                most_f = vocab[word][tag]
            
        most_freq_tags[word] = most_freq_tag
    
    # Create test data tags with most frequent tags given test file
    output_data = []
    
    for x in test:
        
        if(len(x)==2):
            pos = x[0]
            word = x[1]
            
            if(word in most_freq_tags):
                tag = most_freq_tags[word]
                
            else:
                tag = "UNK"
        
            output_data.append(str(pos)+'\t'+str(word)+'\t'+str(tag)+'\n')
        
        else:
            output_data.append(str(x[0]))
    
    return output_data
    
    # Write text file to output test data with tags in same format
    out = open('output.txt','w')
    
    for line in output_data:
        out.write(line)
        
    out.close()



                           
def Goldfile(gtest):
    # Writes gold.txt file for comparison to output.txt file.
                           
    gold = open('gold.txt','w')
    
    for line in gtest:
        gold.write(line)
    
    gold.close()




def Eval(gold, predict):
    # Prints the number of the predicted POS tags that match with the actual POS tags
    # along with the corresponding accuracy.
                           
    # Function assumes gold and predict input parameters have been read line by line
    # from the text file.
    
    pred =[]

    for x in open('output.txt','r').readlines():
        line = x.split('\t')
        pred.append(line)
    
    goldt =[]

    for y in open('gold.txt','r').readlines():
        line = y.split('\t')
        goldt.append(line)
    
        count = 0
        matches = 0
        r = len(gold)
    
        for i in range(0,r):
        
            if(len(gold[i])==3):
            
                if(gold[i][2]==predict[i][2]):
                    matches += 1
                
                count += 1
    
        acc = matches/count
    
        print(str(matches)+'(matches)/'+str(count)+'(count), '+str(acc)+'(accuracy)')
        


                           
# VITERBI ALGORITHM

                           
def Unigrams(training_data):
    # Returns a dictionary of tag unigram counts and a unique tag count.
    
    # Starting dictionary with s/e tag built in to account for state between sentences.    
    unitags = {'s/e':0}
    unique = 0
    
    for line in training_data: 
                           
        # define tag of the line
        if (len(line)==3):
            tag = line[2].strip()
        else:
            tag = 's/e'
        
        # add to tag count if tag is already in dictionary
        if (tag in unitags.keys()):
            unitags[tag] += 1
                           
        # add new tag to dictionary and add to unique tag count
        else:
            unitags[tag] = 1
            unique += 1
            
    return unitags, unique



                           

def Bigrams(training_data):
    # Returns a dictionary of tag bigram counts and unique bigram count.
    
    bitags = {}
    unique = 0
    
    for i in range(0,len(training_data)-1): 
            
            # define tags of current and next line             
            if (len(training_data[i])==3):
                tag1 = training_data[i][2].strip()
            else:
                tag1 = 's/e'
                
            if (len(training_data[i+1])==3):
                tag2 = training_data[i+1][2].strip()
            else:
                tag2 = 's/e'
            
            # define bigram of tags
            bigram = (tag1,tag2)
            
            # add tags bigram to dictionary
            if (bigram in bitags.keys()):
                bitags[bigram] += 1
            else:
                bitags[bigram] = 1
                           
                # add to unique bigram count if new
                unique += 1
    
    return bitags, unique





# Populate the transition probability matrix

def matrixA(training_data):
    # Returns an n x n matrix where n = tag unigram count (A = a_ij).
    
    tag_dict, tag_count = Unigrams(training_data)
    bitag_dict, bitag_count = Bigrams(training_data)
    
    tpmatrix = []
    
    for i in range(0,tag_count):
        row = []
        
        for j in range(0,tag_count):
            t1 = list(tag_dict)[i]
            t2 = list(tag_dict)[j]
            unicount = tag_dict[list(tag_dict)[i]]
            
            if (t1,t2) in bitag_dict.keys():
                bicount = bitag_dict[(t1,t2)]
            else:
                bicount = 0
            
            # add one smoothing
            bicount += 1
            unicount += tag_count
            
            p_ij = np.log(bicount / unicount)
                
            row.append(p_ij)
        
        tpmatrix.append(row)
        
        
    return np.matrix(tpmatrix)




# Populate the output probability matrix

def matrixB(training_data):
    # Returns a tag_count x word_count size matrix (B = b_ik).
    
    tag_dict, tag_count = Unigrams(training_data)
    vocab, word_count = Vocab(training_data)
    
    opmatrix = []
    
    for i in range(0,tag_count):
        row = []
        
        for k in range(0,word_count):
            tag = list(tag_dict)[i]
            word = list(vocab)[k]
            count_of_tag = tag_dict[tag]
            
            if tag in list(vocab[word]):
                word_tag_count = vocab[word][tag]
            else:
                word_tag_count = 0
                           
            # add one smoothing    
            word_tag_count += 1
            count_of_tag += word_count
            
            # log probability of the word given the tag
            p_ik = np.log(word_tag_count / count_of_tag)
                
            row.append(p_ik)
    
        opmatrix.append(row)
        
        
    return np.matrix(opmatrix)




# Constructing the Viterbi matrix to the find the most probable path.

 def Viterbi(training_data, test_file):
    # Returns best path i.e. sequence of tags given the oberservations.
                           
    A = matrixA(training_data)
    B = matrixB(training_data)
    tag_dict, tag_count = Unigrams(training_data)
    vocab, word_count = Vocab(training_data)
    
    # gathing observations from test data
    obs_count = 0  
    obs = []
    for lines in open(test_file, "r").readlines():
        line = lines.split('\t')
        obs_count += 1
        if len(line) < 2:
            obs.append('')
        else:
            obs.append(line[1].strip())
        
        
    # create empty viterbi matrix
    vitmatrix = [] 
                           
    for w in range(0,tag_count):
        row = []
        for t in range(0,obs_count):
            row.append(0)
        vitmatrix.append(row)
                           
    # list for most probable tags generated from viterbi matrix                      
    mptags = []
                           
    # fill first column with prob of tag given the observation
    for s in range(0,tag_count):
    
        if obs[0] in list(vocab):
            for k in range(0,word_count):
                if (obs[0] == list(vocab)[k]):
                    bprob = np.exp(B.A[s][k])
                    
        else:
            bprob = 1/(word_count+tag_count)
            
        vitmatrix[s][0] = np.log(bprob)
    
    amax = np.argmax(vitmatrix,axis=0)
    tag_index = amax[0]
    mpfirsttag = list(tag_dict)[tag_index]
    
    # track the most probable tag for each observation
    mptags.append(mpfirsttag)
         
    for t in range(1,obs_count):
        argmax = np.argmax(vitmatrix,axis=0)
        prev_amax = argmax[(t-1)]
        prev_max = vitmatrix[prev_amax][(t-1)]
        
        for s in range(0,tag_count):
            
            if obs[t] in list(vocab):
                for k in range(0,word_count):
                    if (obs[t] == list(vocab)[k]):
                        bprob = np.exp(B.A[s][k])
                    
            else:
                bprob = 1/(word_count+tag_count)
                
            aprob = np.exp(A.A[prev_amax][s])
            
            vitmatrix[s][t] = np.log(aprob*bprob)
    
        amax = np.argmax(vitmatrix,axis=0)
        tag_index = amax[t]
        mptag = list(tag_dict)[tag_index]
        
        mptags.append(mptag)
    
    return mptags





def Tags_to_txt(tags, test_file):
    # Takes tags generated with viterbi function with the original test_file
    # to generate .txt file with tags.
    
    output_data = []
    counter = 0
    
    for lines in open(test_file, "r").readlines():
        line = lines.split('\t')
        
        if(len(line)==2):
            pos = line[0]
            word = line[1].strip()
            new_line = str(pos)+'\t'+str(word)+'\t'+str(tags[counter])+'\n'
        else:
            new_line = '\n'
        
        output_data.append(new_line)
        counter += 1


                           
    # Write text file to output test data with tags in same format
    out = open('tedder-mark-assgn2-test-output.txt','w')
    
    for line in output_data:
        out.write(line)
        
    out.close()

                           
# Generates best sequence of tags and writes tagged output text file.
                           
training_file = 'berp-POS-training.txt'
turn_in_set = 'testset.txt'

training, test, gold = Bucket(training_file)
A = matrixA(training)
B = matrixB(training)
tags = Viterbi(training, turn_in_set)
Tags_to_txt(tags, turn_in_set)