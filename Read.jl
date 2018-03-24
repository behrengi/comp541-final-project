module Read

using Images
using StatsBase: countmap  

"""
data: Image matrix
tokens: Tokens array, each entry has a String array with sentence tokens.
"""
struct CaptionedImage
    data
    tokens
end

"""

Creates a tokens dictionary where keys are the image file names and values are String arrays with sentence tokens.
raw_captions: Has image file names in the first column and corresponding captions in the second column.

TokensDict:
filename1 -> [Sentence1["token1", "token2", ...], Sentence2["token1", "token2", ...], ...]
.
.
.
"""
function TokensDict(raw_captions)
    tokens = Dict()
    for i in 1:size(raw_captions, 1)
        filename, _ = String.(split(raw_captions[i, 1], '#'))
        if !(filename in keys(tokens))
            tokens[filename] = Any[]
        end
        push!(tokens[filename], String.(split(lowercase(raw_captions[i, 2]))))
    end
    return(tokens)
end


"""
tokens: a dict where keys are the image files and values are the tokenized sentences. 
Returns an array of CaptionImages. 
"""
function ImageTokens(tokens, image_dir)
    images = CaptionedImage[]

    for filename in keys(tokens)
        ############################
        # Images need preprocessing?
        # Check the paper
        ############################
        push!(images, CaptionedImage(
            load(string(image_dir, filename)),
            tokens[filename])
            )
    end
    return(images)
end

"""
Returns i2w and w2i
i2w: Integer to words.
w2i: Words to integers.
minfreq: Optional, minimum frequency of a word to be included in the dictionary.
"""
function CreateDictionaries(tokens; minfreq = 5)
    tokencounts = countmap(collect(FlattenTokens(tokens)))
    tokencounts = filter((t, f) -> f>=minfreq, tokencounts)
    i2w = collect(keys(tokencounts))
    push!(i2w, "_UNK_")
    w2i = Dict(zip(i2w, 1:length(i2w)))
    return(i2w, w2i)
end

"""
Finds all the ngrams given the tokens and n.
"""
function Ngram(tokens, n)
    ngrams = Any[]
    if length(tokens) < n
        return(tokens)
    end
    for i in 1:(length(tokens) - n + 1)
        from, to = i, i + n - 1
        push!(ngrams, convert(Array{String}, tokens[from:to]))
    end
    return(ngrams)
end

"""
Returns an ngram dictionary (key: image filenames, values: array of sentence ngrams)
given tokensDict.

ngramDict:
filename1 -> [Sentence1[ngram1, ngram2, ...], Sentence2[ngram1, ngram2, ...], ...]
.
.
.
"""
function NgramDict(tokens_dict, n)
    ngrams = Any[]
    for (filename, tok_sentences) in tokens_dict
        ngrams[filename] =  map(x -> Read.Ngram(x, n), tok_sentences)
    end
    return(ngrams)
end

"""
caption_length: Dict(length1 => [index, [imagefilename, caption; imagefilename; caption, ...]])
index: index of the next minibatch item. 
"""
function CaptionLengthDict(captions)
    caption_length_dict = Dict()
    index = 1
    for (filename, sentences) in captions
        for sentence in sentences
            sentence_length = length(sentence)
            n_input = reshape([filename, sentence], 1, 2) # new input
            if sentence_length in keys(caption_length_dict)
                caption_length_dict[sentence_length][2] = [caption_length_dict[sentence_length][2]; n_input]
            else
                caption_length_dict[sentence_length] = [1, n_input]
            end
        end
    end 
    return(caption_length_dict)
end

# function BLEU(hypot_tokens, candid_tokens; n = 4)
#     h, c = length(hypot_tokens), length(candid_tokens)
#     bp_penalty = c >= h ? 1 : exp(1 - (h / c)) # brevity penalty
#     bleu = 1
#     for i in 1:n
#         bleu *= BLEU(i, hypot_tokens, candid_tokens)
#     end
#     println(isnan(bp_penalty * (bleu .^ (1 / n))))
#     return(bp_penalty * (bleu .^ (1 / n)))
# end

# function BLEU(n, hypot_tokens, candid_tokens)
#     hypot_ngrams = Ngram(hypot_tokens, n)
#     candid_ngrams = Ngram(candid_tokens, n)
#     precision = Precision(hypot_ngrams, candid_ngrams)
#     return(precision)
# end

function FlattenTokens(tokens_dict)
    tokens = values(tokens_dict)
    return(collect(Iterators.flatten(Iterators.flatten(tokens))))
end

end