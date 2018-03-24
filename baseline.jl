include("Flickr8K.jl")
include("Read.jl")

import Read

import Flickr8K
using StatsBase: countmap  
using PyCall
@pyimport nltk.translate.bleu_score as nl

"""
input: Dict(length1 => [index, [filename, caption; filename; caption, ...]])
output: Any[minibatch1(x, y), minibatch2(x, y), ...]
From the original paper (Xu et al., 2016):
Then, during training we randomly sample a length and retrieve a 
mini-batch of size 64 of that length. We found that this greatly 
improved convergence speed with no noticeable diminishment in 
performance. 
"""
function Minibatch(input, w2i; batchsize=64)
    input_copy = deepcopy(input)
    batched = Any[]
    while !isempty(input_copy)
        b_key = rand(collect(keys(input_copy)))
        b_input = input_copy[b_key]
        x, y = b_input[2][:, 1], b_input[2][:, 2]

        from = b_input[1]
        to = min(from + batchsize - 1, size(b_input[2], 1))
        y, b = MakeBatch(y[from:to], w2i)
        push!(batched, (x[from:to], y, b))

        b_input[1] = to + 1
        if to + 1 > size(b_input[2], 1); delete!(input_copy, b_key); end    
    end
    return(batched)
end

function Minibatch(input; batchsize=64)
    input_copy = deepcopy(input)
    batched = Any[]
    while !isempty(input_copy)
        b_key = rand(collect(keys(input_copy)))
        b_input = input_copy[b_key]
        x, y = b_input[2][:, 1], b_input[2][:, 2]

        from = b_input[1]
        to = min(from + batchsize - 1, size(b_input[2], 1))
        push!(batched, (x[from:to], y[from:to]))

        b_input[1] = to + 1
        if to + 1 > size(b_input[2], 1); delete!(input_copy, b_key); end    
    end
    return(batched)
end

 function MakeBatch(samples, w2i)
    input = Int[]
    longest = length(samples[1])
    batchsizes = zeros(Int, longest)
    
    for i in 1:longest
        batchsize = 0
        for s in samples
            if length(s) >= i 
                s[i] in keys(w2i) ? push!(input, w2i[s[i]]) : push!(input, w2i["_UNK_"])              
                batchsize += 1
            end
        end
        batchsizes[i] = batchsize
    end
    return input, batchsizes
end

# """
# Finds the most frequent_ngram given all the tokenized captions.
# """
# function FrequentNgram(ngrams)
#     ngram_freq = countmap(collect(Iterators.flatten(ngrams)))
#     ngram_freq = Dict(zip(values(ngram_freq), keys(ngram_freq)))
#     return(ngram_freq[maximum(collect(keys(ngram_freq)))])
# end

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
function RandomModelPredict(parameters, input)
    sum_caption_length, n_captions, frequency_dict, n = parameters

    reverse_dict = Dict(zip(collect(values(frequency_dict)), collect(keys(frequency_dict))))
    frequent_ngram = reverse_dict[maximum(collect(keys(reverse_dict)))]
    caption_length = sum_caption_length / n_captions
    ngram_count = round(Int, caption_length / n)

    pred = String[]
    for i in 1:ngram_count
        pred = vcat(pred, frequent_ngram)
    end
    all_pred = Any[]
    for i in 1:length(input)
        push!(all_pred, pred)
    end
    return(all_pred)
end

EPOCHS = 5
flickr8k = Flickr8K.Import()
i2w, w2i = flickr8k.i2w, flickr8k.w2i;
train_captions, test_captions = flickr8k.train_captions, flickr8k.test_captions
train_images, test_images = flickr8k.train_images, flickr8k.test_images


n = 4 # ngram n
ytrain = collect(values(test_captions))
ytest = collect(values(test_images))
xtrain = collect(values(train_captions))
xtest = collect(values(test_captions))

parameters = (0, 0, Dict(), n) # parameter init

#to do the minibatching as described in the paper
train_processed = Read.CaptionLengthDict(train_captions) 
@time for epoch in EPOCHS
    pred_data = Dict()
    batches = Minibatch(train_processed)
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
    # println("BLEU score: ", nl.corpus_bleu(y, pred))
    test_pred = RandomModelPredict(parameters, xtest)
    tr_pred = RandomModelPredict(parameters, xtrain)
    println("train BLEU: ", nl.corpus_bleu(tr_pred, ytrain), "test BLEU: ", nl.corpus_bleu(test_pred, ytest))
end



