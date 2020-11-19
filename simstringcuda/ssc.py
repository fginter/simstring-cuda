"""

A simple implementation of string search against a small-to-midsize (few million max) set of strings using torch and GPU acceleration. This is meant to be a poor man's version of simstring, but does not scale up to anywhere near the DB sizes. On the other hand, it does not need compilation etc. All it needs is sklearn and torch.

Usage:

Here everywhere `strings` refers to a list of strings to index

Make an index and save it:

    import simstringcuda as ssc
    ssc_idx=ssc.build_index(strings)
    ssc.save_index(ssc_idx,filename)

Load a saved index:

    ssc_idx=ssc.load_index(filename)
    ssc_idx.cuda() #If you place the index onto GPU, all search will happen on GPU, but you don't have to if you only have a small number of strings in your DB, this method passes all of its arguments to torch .cuda() call

Lookup some strings:

    queries=["my","query","strings","there","can","be","many"]
    res=ssc.lookup(queries,ssc_idx,10)

"""



import torch
import sklearn.feature_extraction
import sys
import time
import pickle

class SSCModel:

    def __init__(self,sparse_t,vectorizer,strings):
        self.sparse_t=sparse_t
        self.vectorizer=vectorizer
        self.strings=strings

    def cuda(self,*args,**kwargs):
        """Place index on GPU, all arguments will be passed as-is to the `.cuda()` pytorch call"""
        self.sparse_t=self.sparse_t.cuda(*args,**kwargs)

    def dump(self,f):
        pickle.dump((self.sparse_t.cpu(),self.vectorizer,self.strings),f)

    @classmethod
    def load(cls,f):
        return cls(*pickle.load(f))


def scipy2torch_sparse(sparse_mat):
    nz_row,nz_col=sparse_mat.nonzero()
    indices=torch.LongTensor([nz_row,nz_col])
    sparse_t=torch.sparse.FloatTensor(indices,torch.FloatTensor(sparse_mat.data))
    return sparse_t

def build_index(strings):
    """ 
    Builds the index out of strings, which is an iterable, most likely a list

    Input:
      `strings`

    Returns: the index, instance of SSCModel
    """

    vectorizer=sklearn.feature_extraction.text.TfidfVectorizer(analyzer="char",ngram_range=(3,3),norm="l2",use_idf=False)
    vectorized=vectorizer.fit_transform(strings)
    sparse_t=scipy2torch_sparse(vectorized)
    return SSCModel(sparse_t, vectorizer, strings)

#we might mess with this at some later point
def get_lengths(strings):
    lengths={} #key: length, value: first index where this length is spotted
    for i,s in enumerate(strings):
        l=len(s)
        if l in lengths:
            continue
        lengths[l]=i
    return lengths

def save_index(ssc_idx,file_name):
    """ 
    Saves the index

    Input:
      `ssc_idx` the index to save
      `file_name`

    Returns nothing
    """

    with open(file_name,"wb") as f:
        ssc_idx.dump(f)

def load_index(file_name):
    """ 
    Returns loaded index, residing on CPU. If you want it to be on GPU, call .cuda() on the result of this function once.

    Input:
      `file_name`

    Returns loaded index. Instance of SSCModel
    """
    with open(file_name,"rb") as f:
        return SSCModel.load(f)

def lookup(queries,ssc_idx,topk=10):
    """
    Does the lookup. If you want it to happen on GPU as you probably do, remember to run ssc_idx.cuda() once after loading the index.

    Input:
      `queries` iterable over query strings
      `ssc_idx` whatever you get from load_index()
      `topk` the number of nearest hits you want

    Returns a [[(hit00,sim00),(hit01,sim01)],[(hit10,sim10),(hit11,sim11)],...] list of lists. There are as many lists 
    as there are query strings and in every list there are topk (hitword,simvalue) tuples.
    """
    #queries cannot be massively large, because a similarity matrix of queries x db will be created at some point
    queries_vectorized=ssc_idx.vectorizer.transform(queries) # string x dim, sparse
    queries_vectorized_torch=scipy2torch_sparse(queries_vectorized).to_dense().T # dim x string
    _,max_features=ssc_idx.sparse_t.size()
    q_features,q_count=queries_vectorized_torch.size() #dim x string
    if q_features<max_features: #almost always the case, since q_features only has max nonzero feature index in the queries
        zeros=torch.zeros((max_features-q_features,q_count),dtype=queries_vectorized_torch.dtype)
        queries_vectorized_torch=torch.vstack([queries_vectorized_torch,zeros]) #add enough many zeros to match dimensions
    #Now stick this to the same device
    queries_vectorized_torch=queries_vectorized_torch.to(ssc_idx.sparse_t.device)
    similarities=torch.sparse.mm(ssc_idx.sparse_t,queries_vectorized_torch).T #query string x database string
    topk=torch.topk(similarities,topk)
    result=[]
    for query,simvals in zip(topk.indices,topk.values):
        result.append([])
        for hit,simval in zip(query,simvals):
            result[-1].append( (ssc_idx.strings[int(hit)],float(simval)) )
    return result
    


