include("Data.jl")
include("Read.jl")

import Read
import Data
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
function Minibatch(inputs; batchsize=64)
    froms = Dict(k => 1 for k in keys(inputs))
    batched = Any[]
    while !isempty(froms)
        len, from = rand(froms)
        inps = inputs[len]

        to = min(from + batchsize - 1, size(inps, 1))
        batch = inps[from:to, :]
        
        x, y = batch[:, 1], batch[:, 2]

        push!(batched, (x, y))

        froms[len] = to + 1
        if to + 1 > size(inps, 1)
            delete!(froms, len)
        end
    end
    return(batched)
end

# function Minibatch(input, w2i; batchsize=64)
#     input_copy = deepcopy(input)
#     batched = Any[]
#     while !isempty(input_copy)
#         b_key = rand(collect(keys(input_copy)))
#         b_input = input_copy[b_key]
#         x, y = b_input[2][:, 1], b_input[2][:, 2]

#         from = b_input[1]
#         to = min(from + batchsize - 1, size(b_input[2], 1))
#         y, b = MakeBatch(y[from:to], w2i)
#         push!(batched, (x[from:to], y, b))

#         b_input[1] = to + 1
#         if to + 1 > size(b_input[2], 1); delete!(input_copy, b_key); end    
#     end
#     return(batched)
# end

"""
"""
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
0
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


data = Data.ImportFlickr8K()
i2w, w2i = data.i2w, data.w2i;
train_captions, test_captions = data.train_captions, data.test_captions
train_images, test_images = data.train_images, data.test_images

println("data importing end")

n = 4 # ngram n
ytrain = collect(values(train_captions))
ytest = collect(values(test_captions))
xtrain = collect(values(train_images))
xtest = collect(values(test_images))

EPOCHS = 1
test_pred = train_pred = []
parameters = (0, 0, Dict(), n) # parameter init
bleu_weights = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

#to do the minibatching as described in the paper
train_processed = Read.CaptionCountDict(train_captions) 
@time for epoch in 1:EPOCHS
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
    train_pred = RandomModelPredict(parameters, xtrain)
    
end


println("train BLEU: ", 100 * nl.corpus_bleu(ytrain, train_pred),
     " test BLEU: ", 100 * nl.corpus_bleu(ytest, test_pred))

for b_w in bleu_weights
    println(" train BLEU: ", 100 * nl.corpus_bleu(ytrain, train_pred, b_w),
     " test BLEU: ", 100 * nl.corpus_bleu(ytest, test_pred, b_w))
end