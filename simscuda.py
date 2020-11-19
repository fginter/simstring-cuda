#!/usr/bin/env python3

import argparse
import sys
import torch
import simstringcuda as ssc
import json

parser = argparse.ArgumentParser()
g=parser.add_argument_group("Common options")
g.add_argument("index",help="Filename of the index file used in lookup/indexing")
g=parser.add_argument_group("Index creation")
g.add_argument("-c","--create",default=False,action="store_true",help="Create the index using strings from stdin")
g=parser.add_argument_group("Lookup")
g.add_argument("-k",default=10,type=int,help="Retrieve top k hits for every query string on stdin. Default: %(default)d")
g.add_argument("--cpu",default=False,action="store_true",help="Force CPU, otherwise use GPU. Only relevant when doing lookup, not indexing.")
g.add_argument("--querybatch",default=100,type=int,help="Batch queries by this many. Decrease on cuda out of memory error. On CPU, larger batches make no difference and this parameter is ignored. Default: %(default)d")
g.add_argument("--jsonl",default=False,action="store_true",help="Output in json lines, otherwise human-readable.")
args=parser.parse_args()

if args.cpu:
    args.querybatch=1

if args.create: #We are asked to index stuff from stdin
    strings=[s.strip() for s in sys.stdin if s.strip()]
    ssc_model=ssc.build_index(strings)
    ssc.save_index(ssc_model,args.index)
    print(f"Indexed {len(strings)} strings.",file=sys.stderr)
else: #We are doing lookup
    ssc_model=ssc.load_index(args.index)
    if not args.cpu and torch.cuda.is_available():
        ssc_model.cuda()
    queries=[s.strip() for s in sys.stdin if s.strip()]
    for i in range(0,len(queries),args.querybatch):
        q_batch=queries[i:i+args.querybatch]
        result=ssc.lookup(q_batch,ssc_model,args.k)
        for q,nearest in zip(q_batch,result):
            if args.jsonl:
                res={"q":q,"r":nearest}
                print(json.dumps(res,ensure_ascii=False,sort_keys=True))
            print(q,", ".join(w for w,simval in nearest),sep="\t")

