for p in ("Images","StatsBase","MAT","JLD","Knet", "DataStructures", "Distances")
    Pkg.installed(p) == nothing && Pkg.add(p)
end

module Read

using Images
using StatsBase: countmap
using MAT
using JLD
using Knet
using DataStructures
using Distances

ftype = Float32
atype = gpu() >= 0 ? KnetArray{ftype} : Array{ftype}


"""

Creates a tokens dictionary where keys are the image file names and values are String arrays with sentence tokens.
raw_captions: Has image file names in the first column and corresponding captions in the second column.

TokensDict:
filename1 -> [Sentence1["token1", "token2", ...], Sentence2["token1", "token2", ...], ...]
.
.
.
"""
function TokensDict(raw_captions; start_token=false)
    tokens = Dict()
    for i in 1:size(raw_captions, 1)
        filename = String(split(raw_captions[i, 1], '#')[1])
        if !haskey(tokens, filename)
            tokens[filename] = Any[]
        end
        splitted = String.(split(lowercase(raw_captions[i, 2])))
        if start_token
            push!(tokens[filename], ["_BOS_"; splitted; "_EOS_"])
        else
            push!(tokens[filename], [splitted; "_EOS_"])
        end
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

function Covariance(A)
    A = atype(A')
    n = size(A, 2)
    C = sum(A' * A for i in 1:n)
    return Array{Float32}(C ./ n)
end

"""
Loads the images and returns a dictionary where keys are the filenames and the
values are the images.
"""
function FeatureDict(image_dir, featuredict_name, filenames)
    image_dict = OrderedDict()
    feature_file = joinpath(image_dir, string(featuredict_name, "features.jld"))
    println(feature_file)
    if !isfile(feature_file)
        println("creating file")
        feature_dict = Dict()
        vgg_layers = VggLoad()["layers"][1:35]
        i = 1
        for file in filenames
            feature_dict[file] = VggFeatures(Images.load(joinpath(image_dir, file)), vgg_layers)
            i += 1
            if i % 200 == 0
                println("image loading process: %", 100 * (i / length(filenames)))
            end
        end

        features = cat(3, values(feature_dict)...)
        features_mean = mean(features, 3)
        features_std = std(features, 3, mean=features_mean)
        for file in filenames
            feature_dict[file] = (feature_dict[file] .- features_mean) ./ features_std
        end
        JLD.save(feature_file, "feature_dict", feature_dict)
    else
        feature_dict = JLD.load(feature_file)["feature_dict"]
    end
    return(feature_dict)
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
    push!(i2w, "_BOS_")
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
        push!(ngrams, tokens[from:to])
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
caption_length: Dict(length1 => [(imagefilename, caption), (imagefilename, caption), ...])
index: index of the next minibatch item.
"""
function CaptionCountDict(captions)
    captions_by_length = Dict()
    for (filename, sentences) in captions
        for sentence in sentences
            sentence_length = length(sentence)
            n_input = (filename, sentence) # new input
            if haskey(captions_by_length, sentence_length)
                captions_by_length[sentence_length] = [captions_by_length[sentence_length]; n_input]
            else
                captions_by_length[sentence_length] = [n_input]
            end
        end
    end
    return(captions_by_length)
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

function VggLoad()
    file = joinpath(pwd(), "data", "vgg_19.mat")
    if !isdir("data")
        mkdir(joinpath(pwd(), "data"))
    end
    if !isfile(file)
        download("http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat", file)
    end
    return(matread(file))
end

function PreProcessImage(image; image_size=224)
    image = permutedims(float32.(rawview(channelview(image))), (2, 3, 1))
    h, w = size(image, 1), size(image, 2)
    if h < w
        resized = imresize(image, (256, floor(Int, w * (256 / h))))
    else
        resized = imresize(image, (floor(Int, h * (256 / w)), 256))
    end
    im_center = floor.(Int, center(resized))
    im_size_halved = Int(image_size / 2)
    cropped = resized[(im_center[1] - im_size_halved) : (im_center[1] + im_size_halved - 1),
                        (im_center[2] - im_size_halved) : (im_center[2] +  im_size_halved - 1), :]
    return(atype(cropped))

end

function VggFeatures(input_dir, vgg_layers)
    # use gpu to extract features if exists
    atype = atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}

    input = Read.PreProcessImage(input_dir)
    input = atype(reshape(input, (size(input)..., 1)))
    for layer in vgg_layers
        if layer["type"] == "conv" || layer["type"] == "pool"
            padding = Int.(layer["pad"][1])
            stride = Int.(layer["stride"][1])
            if layer["type"] == "conv"
                weights = atype(layer["weights"][1])
                input = conv4(weights, input; padding=padding, stride=stride)
            else
                input = pool(input; padding=padding, stride=stride, window = Int.(layer["pool"]))
            end
        elseif layer["type"] == "relu"
            input = relu.(input)
        end
    end
    return Array{Float32}(reshape(input, (size(input, 1) * size(input, 2), size(input, 3))))
end

function OneHot(i, size)
    onehot = spzeros(1, size)
    onehot[i] = 1
    return(onehot)
end

"""
Given the onehot vector and index to word dictionary, returns the word.
"""
function ToWord(onehot, i2w)
    return(i2w[findfirst(onehot .== 1)...])
end

end