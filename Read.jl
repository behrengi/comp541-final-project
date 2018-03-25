module Read

using Images
using StatsBase: countmap  

# """
# data: Image matrix
# tokens: Tokens array, each entry has a String array with sentence tokens.
# """
# struct CaptionedImage
#     data
#     tokens
# end

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
        if !haskey(tokens, filename)
            tokens[filename] = Any[]
        end
        push!(tokens[filename], String.(split(lowercase(raw_captions[i, 2]))))
    end
    return(tokens)
end


# """
# tokens: a dict where keys are the image files and values are the tokenized sentences. 
# Returns an array of CaptionImages. 
# """
# function ImageTokens(tokens, image_dir)
#     images = CaptionedImage[]

#     for filename in keys(tokens)
#         ############################
#         # Images need preprocessing?
#         # Check the paper
#         ############################
#         push!(images, CaptionedImage(
#             load(string(image_dir, filename)),
#             tokens[filename])
#             )
#     end
#     return(images)
# end

"""
Loads the images and returns a dictionary where keys are the filenames and the
values are the images.
"""
function ImageDict(image_dir, filenames)
    ImageDict = Dict()
    i = 1
    for file in filenames
        ImageDict[file] = Images.load(string(image_dir, file))
        i = i + 1
        if i % 200 == 0
            println("image loading process: %", 100 * (i / length(filenames)))
        end
    end
    return(ImageDict)
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
function CaptionCountDict(captions)
    caption_count = Dict()
    for (filename, sentences) in captions
        for sentence in sentences 
            sentence_length = length(sentence)
            n_input = reshape([filename, sentence], 1, 2) # new input
            if haskey(caption_count, sentence_length)
                caption_count[sentence_length] = [caption_count[sentence_length]; n_input]
            else
                caption_count[sentence_length] = n_input
            end
        end
    end 
    return(caption_count)
end

function FlattenTokens(tokens_dict)
    tokens = values(tokens_dict)
    return(collect(Iterators.flatten(Iterators.flatten(tokens))))
end

function JoinWithSpaces(tokens)
    tokens_with_spaces = String[]
    for token in tokens
        push!(tokens_with_spaces, token, " ")
    end
    return(join(tokens_with_spaces[1:end-1]))
end

end