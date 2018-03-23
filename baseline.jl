[include(script) for script in ["Flickr8K.jl", "Read.jl"]]

using Flickr8K
import Read
using StatsBase: countmap  

flickr8k = Flickr8K.Import()
i2w, w2i = flickr8k.i2w, flickr8k.w2i;
train_captions, test_captions = flickr8k.train_captions, flickr8k.test_captions
train_images, test_images = flickr8k.train_images, flickr8k.test_images


# Input: Dict(length1 => [index, [filename, caption; filename; caption, ...]])
# From the original paper:
# Then, during training we randomly sample a length and retrieve a 
# mini-batch of size 64 of that length. We found that this greatly 
# improved convergence speed with no noticeable diminishment in 
# performance. 
function minibatch(input; batchsize=64)
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

function frequent_ngram(ngrams)
    ngram_freq = countmap(Read.FlattenTokens(ngrams))
    ngram_freq = Dict(zip(values(ngram_freq), keys(ngram_freq)))
    return(ngram_freq[maximum(collect(keys(ngram_freq)))])
end

function mean_length_ngram(ngrams, captions)
    sum = 0
        i = 1
        for image in values(ngrams)
            for captions in image
                sum += length(captions)
                i = i + 1;
            end
        end
        return(round(Int, sum ./ i))
end

function random_model_params(captions)
    ngrams = Read.NgramDict(train_captions, 4)
    frequent = frequent_ngram(ngrams)
    length = mean_length_ngram(ngrams, captions) 
    
    return(frequent, length)
end

function random_model(parameters)
    frequent_ngram, max_ngram_count = parameters
    pred = Any[]
    for i in 1:max_ngram_count
        pred = vcat(pred, frequent_ngram)
    end
    return(pred)
end

pred = Dict()
random_output = random_model(random_model_params(train_captions))

for (k, v) in test_captions
    push!(pred, k => random_output)
end 

# sum_BLEU = 0
# i = 1
# for (k, v) in test_captions
#     if i > 10
#         break
#     end
#     target_captions = test_captions[k]
#     c_BLEU = 0
#     println(pred[k])
#     for caption in target_captions
#         println(caption)
#         c_BLEU += Read.BLEU(1, pred[k], caption)
#     end
#     sum_BLEU += c_BLEU / 5
#     i = i + 1
# end
# sum_BLEU /= length(test_captions)
