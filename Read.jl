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
"""
function TokensDict(raw_captions)
    tokens = Dict()
    for i in 1:size(raw_captions, 1)
        filename, _ = split(raw_captions[i, 1], '#')
        if !(filename in keys(tokens))
            tokens[filename] = Any[]
        end
        push!(tokens[filename], split(lowercase(raw_captions[i, 2])))
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
            tokens[filename]
            ))
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
    tokencounts = countmap(collect(Iterators.flatten(Iterators.flatten(values(tokens)))))
    tokencounts = filter((t, f) -> f>=minfreq, tokencounts)
    i2w = collect(keys(tokencounts))
    w2i = Dict(zip(i2w, 1:length(i2w)))
    return(i2w, w2i)
end

end