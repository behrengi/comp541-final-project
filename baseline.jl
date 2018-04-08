for p in ("StatsBase","PyCall","Images", "Knet")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

include("Data.jl")
include("Read.jl")

using Read
using Data
using StatsBase: countmap  
using PyCall
using Knet
@pyimport nltk.translate.bleu_score as nl

"""
input: caption_length_dict
caption_length_dict: Dict(length1 => [filename, caption; filename; caption, ...])

output: Any[minibatch1(x, y), minibatch2(x, y), ...]

From the original paper (Xu et al., 2016):
Then, during training we randomly sample a length and retrieve a 
mini-batch of size 64 of that length. We found that this greatly 
improved convergence speed with no noticeable diminishment in 
performance. 
"""
function Minibatch(input, w2i, atype; batchsize=64)
    froms = Dict(k => 1 for k in keys(input))
    batched = Any[]
    while !isempty(froms)
        len, from = rand(froms)
        inps = input[len]

        to = min(from + batchsize - 1, length(inps))
        batch = inps[from:to]

        push!(batched, MakeBatch(batch, w2i, atype))

        froms[len] = to + 1
        if to + 1 > length(inps)
            delete!(froms, len)
        end
    end
    return(batched)
end

"""
"""
 function MakeBatch(samples, w2i, atype)
    captions = Any[]
    filenames = Any[]
    for (filename, sentences) in samples
        tokens = Any[]
        for s in sentences
            s in keys(w2i) ? push!(tokens, w2i[s]) : push!(tokens, w2i["_UNK_"])  
        end
        push!(filenames, filename)
        push!(captions, atype(tokens))
    end
    return (filenames, captions)
end

# 
function RandomModelTrain2(parameters, data)
    sum_caption_length, n_captions, frequency_dict, n = parameters
    images, captions = data
    sum_caption_length += sum(length.(captions))
    n_captions += length(captions)

    caption_ngrams = Read.Ngram.(captions, n)
    for caption in caption_ngrams
        for ngram in caption
            if ngram in keys(frequency_dict)
                frequency_dict[ngram] += 1 
            else
                frequency_dict[ngram] = 1
            end
        end
    end
    parameters = sum_caption_length, n_captions, frequency_dict, n
    return(parameters)
end

function RandomModelPredict2(parameters, input, i2w)
    sum_caption_length, n_captions, frequency_dict, n = parameters
    mean_caption_length = sum_caption_length / n_captions
    
    sentence = ["_BOS_"]
    while (length(sentence) < mean_caption_length) && (sentence[end] != "_EOS_")
        reldict = filter((k, v) -> i2w[Int(k[1])] == sentence[end], frequency_dict)
        mostfreqbi, _ = reduce((acc, n) -> n[2] > acc[2] ? n : acc, reldict)
        push!(sentence, i2w[Int.(mostfreqbi[2])])
    end

    predictions = Any[]
    for i in 1:length(input)
        push!(predictions, sentence)
    end
    return(predictions)
end

"""
Finds the parameters of the random model.
"""
function RandomModelTrain(parameters, data)
    sum_caption_length, n_captions, frequency_dict, n = parameters
    images, captions = data
    sum_caption_length += sum(length.(captions))
    n_captions += length(captions)

    caption_ngrams = Read.Ngram.(captions, n)
    for caption in caption_ngrams
        for ngram in caption
            if ngram in keys(frequency_dict)
                frequency_dict[ngram] += 1 
            else
                frequency_dict[ngram] = 1
            end
        end
    end
    parameters = sum_caption_length, n_captions, frequency_dict, n
    return(parameters)
end

"""
Predicts the captions given parameters and input.
"""
function RandomModelPredict(parameters, input, i2w)
    sum_caption_length, n_captions, frequency_dict, n = parameters

    reverse_dict = Dict(zip(collect(values(frequency_dict)), collect(keys(frequency_dict))))
    frequent_ngram = reverse_dict[maximum(collect(keys(reverse_dict)))]
    caption_length = sum_caption_length / n_captions
    ngram_count = round(Int, caption_length / n)

    pred = String[]
    for i in 1:ngram_count
        pred = vcat(pred, i2w[Int.(frequent_ngram)])
    end
    all_pred = Any[]
    for i in 1:length(input)
        push!(all_pred, pred)
    end
    return(all_pred)
end

atype = gpu() >= 0 ? KnetArray{Float32}  : Array{Float32}

data = Data.ImportFlickr8K()
i2w, w2i = data.i2w, data.w2i;
train_captions, test_captions = data.train_captions, data.test_captions
train_features, test_features = data.train_features, data.test_features

println("data importing end")

n = 4 # ngram n
ytrain = collect(values(train_captions))
ytest = collect(values(test_captions))
xtrain = collect(values(train_features))
xtest = collect(values(test_features))

EPOCHS = 1
test_pred = train_pred = []
parameters = (0, 0, Dict(), n) # parameter init
bleu_weights = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
batches = []

#to do the minibatching as described in the paper
train_count = Read.CaptionCountDict(train_captions) 
@time for epoch in 1:EPOCHS
    pred_data = Dict()
    batches = Minibatch(train_count, w2i, atype)
    # batches = Minibatch2(train_processed, w2i)
    # for (x, y, b) in batches
    for (x, y) in batches
        # x has the images, y has the captions. 

        # This baseline model finds the most frequent ngram in all the captions
        # and repeates the ngram until the predicted caption length more or less
        # matches the mean caption length of the training set. "More or less" because
        # mean caption length of the training set is a floating point number.
        parameters = RandomModelTrain(parameters, (x, y))
    end
    println("out")
    # println("BLEU score: ", nl.corpus_bleu(y, pred))
    test_pred = RandomModelPredict(parameters, xtest, i2w)
    train_pred = RandomModelPredict(parameters, xtrain, i2w)
    
end


println("train BLEU: ", 100 * nl.corpus_bleu(ytrain, train_pred),
     " test BLEU: ", 100 * nl.corpus_bleu(ytest, test_pred))

for b_w in bleu_weights
    println(" train BLEU: ", 100 * nl.corpus_bleu(ytrain, train_pred, b_w),
     " test BLEU: ", 100 * nl.corpus_bleu(ytest, test_pred, b_w))
end