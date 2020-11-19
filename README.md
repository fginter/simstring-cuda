A simple implementation of string search against a small-to-midsize (few million max) set of strings using torch and GPU acceleration. Cosine similarity of character 3-grams is the metric. This is meant to be a poor man's version of simstring, but does not scale up to anywhere near the DB sizes, and does not implement any of the fine tricks in simstring. On the other hand, it is easy to install. All it needs is sklearn and torch.

If the queries are batched by few hundred, the lookup against a DB of 1.4M strings from wikidata runs at 0.004sec per string on a relatively dated GPU.

# Installation

    python3 setup.py install

or

    python3 setup.py bdist_wheel

the wheel file is in `dist/SimString_cuda-0.1.0-py3-none-any.whl` and then you can install it anywhere you want with `pip3 install ____.whl`

# Usage

Here everywhere `strings` refers to a list of strings to index

## Make an index and save it:

    import simstringcuda as ssc
    ssc_idx=ssc.build_index(strings)
    ssc.save_index(ssc_idx,filename)

## Load a saved index:

    ssc_idx=ssc.load_index(filename)
    ssc_idx.cuda() #If you place the index onto GPU, all search will happen on GPU, but you don't have to if you only have a small number of strings in your DB, this method passes all of its arguments to torch .cuda() call

## Lookup some strings:

For GPU to make any sense, queries should preferably be batched into batches of few hundred or so, depending on your GPU memory. The limiting factor on memory is that a matrix of index x query is created. If your lookup runs out of GPU memory, make smaller query batches.

    queries=["my","query","strings","there","can","be","many"]
    res=ssc.lookup(queries,ssc_idx,10) #find top-10 hits for every query string


# Command-line usage:

The command simcuda gets installed for you via pip, so maybe best install the package this way.

    pip3 install path/to/builtwheel.whl
    simcuda -h

Create an index out of all strings in a file, store it as `index.fi` file

    bzcat strings.fi.bz2 | simscuda -c index.fi

Look up the first 1000 of these again

    bzcat strings.fi.bz2 | head -n 1000 | simcuda index.fi


And get the output in a jsonl format for easier processing later

    bzcat strings.fi.bz2 | head -n 1000 | simcuda --jsonl index.fi > out.jsonl

    
