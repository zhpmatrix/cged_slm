import kenlm
import heapq
import numpy as np
import pandas as pd
import ast
import math
from pyltp import Segmentor
from tqdm import tqdm

def seg_analysis():
    segger = Segmentor()
    segger.load('../data/Data_LCSTS_News_Taobao_Chatbot/model/cws.model')
   
    writer = open('split_words.txt', 'w')
    test_num = 7999
    data = pd.read_csv('data/aug/aug.data.train.lc', header=None, sep='\t',nrows=test_num)
    unequal_counter = 0
    for i in range(data.shape[0]):
        src_text = data.iloc[i][0]
        gs = ast.literal_eval(data.iloc[i][5])
        ls = segger.segment(src_text)
        ls = [word for word in ls]

        if len(gs) != len(ls):
            writer.write(' '.join(gs)+'\n'+' '.join(ls)+'\n\n')
            unequal_counter += 1
    print('unequal ratio:{}'.format(unequal_counter * 1.0 / data.shape[0]))
    writer.close()

def eval():
    ngram1 = 3
    ngram2 = 2
    topk = 1
    #model1 = kenlm.Model('data/lcsts/lcsts.'+str(ngram1)+'.bin')
    #model2 = kenlm.Model('data/lcsts/lcsts.'+str(ngram2)+'.bin')
    model1 = kenlm.Model('data/wiki/wiki.'+str(ngram1)+'.bin')
    model2 = kenlm.Model('data/wiki/wiki.'+str(ngram2)+'.bin')
    segger = Segmentor()
    segger.load('../data/Data_LCSTS_News_Taobao_Chatbot/model/cws.model')
    
    test_num = 7999
    data = pd.read_csv('data/aug/aug.data.train.lc', header=None, sep='\t',nrows=test_num)
    counter = 0
    err_len = 0
    bad_case = 0
    for i in tqdm ( range(data.shape[0]) ):
        input_ = data.iloc[i][0]
    
        real_split = ast.literal_eval(data.iloc[i][5])
        err_pos = int(data.iloc[i][4])
        err_word = real_split[err_pos]
        
        word = segger.segment(input_)
        word = [w for w in word]

        if len(real_split) != len(word):
            bad_case += 1
            #import pdb;pdb.set_trace()
            continue
        
        text = ' '.join(word)
        scores1 = model1.full_scores(text,bos=False, eos=False)
        scores1_ = list(scores1)
        
        scores2 = model2.full_scores(text,bos=False, eos=False)
        scores2_ = list(scores2)
        
        thres = -5.5 
        err_w_candidates = []
        
        
        #for j in range(len(scores1_)):
        #    if scores1_[j][0] <= thres and scores2_[j][0] <= thres:
        #        err_w_candidates.append(j)

        for j,elem in enumerate(scores2_):
            if elem[0] <= thres:
                err_w_candidates.append(j)
        
        #err_w_candidates = map(scores1_.index, heapq.nsmallest(topk, scores1_))
        err_list = list(err_w_candidates)
        err_len += len(err_list) 
        #print(text)        
        #print([text.split(' ')[i] for i in err_list],err_word)
        #print(list( err_list ), err_pos)
        
        if err_pos in list(err_list):
            counter += 1
        else:
            continue
            import pdb;pdb.set_trace()
    print('acc:', counter * 1.0/ test_num)
    print(err_len*1.0/test_num)

def get_ppl(scores):
    """ from StackOverflow """ 
    product_inv_prob = np.prod([math.pow(10.0, item[0]) for item in scores])
    n = len(scores)
    perplexity = math.pow(product_inv_prob, 1.0/n)
    return perplexity

def test():
    ngram = 3
    topk = 1
    model = kenlm.Model('data/lcsts/lcsts.'+str(ngram)+'.bin')
    with open('data/aug/my_test.seg', 'r') as reader:
    #with open('data/aug/test.wucuozi.seg.lc', 'r') as reader:
    #with open('data/aug/aug.data.train.seg', 'r') as reader:
        texts = reader.readlines()
    for text in texts:
        scores = model.full_scores(text.strip(), bos=False, eos=False)
        scores_ = list(scores)
        err_w_candidates = map(scores_.index, heapq.nsmallest(topk, scores_))
        print([text.split(' ')[i] for i in list( err_w_candidates )],text)
        
        #ppl = model.perplexity(text.strip())
        #print(ppl)
        #score = model.score(text.strip())
        #print(score)
if __name__ == '__main__':
    seg_analysis()
    #eval()
    #test()
