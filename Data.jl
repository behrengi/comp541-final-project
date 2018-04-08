for p in ("JSON","PyCall")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
include("Read.jl")

module Data

import Read
import JSON
using PyCall
@pyimport nltk.tokenize as tokenizer

struct data
    i2w
    w2i
    train_features
    dev_features
    test_features
    train_captions
    dev_captions
    test_captions
end

function Import(captions_dir, train_dir, dev_dir, test_dir, image_dir)
    println("readdlm begin")
    raw_captions = readdlm(captions_dir, '\t', comments=false)
    raw_train = readdlm(train_dir)[1:5]
    raw_dev = readdlm(dev_dir)[1:5]
    raw_test = readdlm(test_dir)[1:5]
    println("readdlm end")

    println("tokens dict")
    captions_dict = Read.TokensDict(raw_captions)

    println("dictionaries")
    (i2w, w2i) = Read.CreateDictionaries(captions_dict)

    println("captions start")
    train_captions = Dict()
    dev_captions = Dict()
    test_captions = Dict()
    
    for fn in raw_train
        train_captions[fn] = captions_dict[fn]
    end
    for fn in raw_dev
        dev_captions[fn] = captions_dict[fn]
    end
    for fn in raw_test
        test_captions[fn] = captions_dict[fn]
    end 
    println("captions end")
    
    if length(image_dir) == 1 # Flickr8k and Flickr30k images
        train_features = Read.FeatureDict(image_dir[1], raw_train)
        println("training images features extracted")

        test_features = Read.FeatureDict(image_dir[1], raw_test)
        println("dev images features extracted")

        dev_features = Read.FeatureDict(image_dir[1], raw_dev)
        println("test images features extracted")
    else # COCO images are in two separate folders (train and val)
        train_features = Read.FeatureDict(image_dir[1], raw_train)
        println("training images features extracted")

        test_features = Read.FeatureDict(image_dir[2], raw_test)
        println("dev images features extracted")

        dev_features = Read.FeatureDict(image_dir[2], raw_dev)
        println("test images features extracted")
    end

    # train_features = rand(6000, 1)
    # test_features = rand(1000, 1)
    # dev_features = rand(1000, 1)

    return(data(i2w, w2i, train_features, dev_features, test_features, 
                train_captions, dev_captions, test_captions))
end

function ImportFlickr8K()
    #dir = joinpath(pwd, "data", "FLICKR8K")
    dir = "/Users/bihterakyol/Desktop/SPRING2018/COMP541/datasets/FLICKR8K"
    text_dir = joinpath(dir, "Flickr8k_text")
    image_dir = joinpath(dir, "Flickr8k_Dataset")

    captions_dir = joinpath(text_dir, "Flickr8k.lemma.token.txt")
    train_images_dir = joinpath(text_dir, "Flickr_8k.trainImages.txt")
    dev_images_dir = joinpath(text_dir, "Flickr_8k.devImages.txt")
    test_images_dir = joinpath(text_dir, "Flickr_8k.testImages.txt")

    Import(captions_dir, train_images_dir, dev_images_dir, test_images_dir, [image_dir])
end

function ImportFlickr30K()
    #dir = joinpath(pwd, "data", "FLICKR30K")
    dir = "/Users/bihterakyol/Desktop/SPRING2018/COMP541/datasets/FLICKR30K"
    image_dir = joinpath(dir, "flickr30k_images")
    json_dir = joinpath(dir, "dataset.json")    
    captions_dir, train_dir, dev_dir, test_dir = JSONtoTxt(json_dir, dir)

    Import(captions_dir, train_dir, dev_dir, test_dir, [image_dir])
end

function ImportCOCO()  
    #dir = joinpath(pwd, "data", "COCO")
    dir = "/Users/bihterakyol/Desktop/SPRING2018/COMP541/datasets/COCO"
    train_images_dir = joinpath(dir, "train2014") 
    test_images_dir = joinpath(dir, "val2014") 
    json_dir = joinpath(dir, "dataset.json")

    println("json to txt begin")
    captions_dir, train_dir, dev_dir, test_dir = JSONtoTxt(json_dir, dir)
    println("json to txt end")

    Import(captions_dir, train_dir, dev_dir, test_dir, [train_images_dir, test_images_dir])
end

"""
Function to turn Flickr30k and COCO dataset json files Flickr8k compatible.
This function wil create four different txt files, and return their filenames. 
1st will have the format: imagefilename#id caption
It will also parse tokens using nltk to match Flickr8k parsing.
2nd, 3rd, 4th will have the imagefilenames for train, dev and test datasets respectively.
"""
function JSONtoTxt(json_dir, save_dir)
    captions_dir = string(save_dir, "/", "tokens.txt")
    train_dir = string(save_dir, "/", "train.txt")
    dev_dir = string(save_dir, "/", "dev.txt")
    test_dir = string(save_dir, "/", "test.txt")
    
    if !(all([isfile(file) for file in [train_dir, dev_dir, test_dir]]))
        println("txt files does not exist, creating them")

        println("reading json file")
        json = JSON.parsefile(json_dir)
        println("finished reading json file")
        images = json["images"]
        
        captions = Array{String}(0, 2)
        train = Array{String}(0)
        dev = Array{String}(0)
        test = Array{String}(0)
        file_categories = Dict("test" => test, "val" => dev, "train" => train)
    
        j = 1
        for image in images
            if j % 100 == 0
                println("at image: ", j)
            end
            filename = image["filename"]
            # there is a restval split, check if you're going to use it
            if !(image["split"] == "restval") 
                push!(file_categories[image["split"]], filename)
                for (i, sentence) in enumerate(image["sentences"][1:5])
                    caption = Read.JoinWithSpaces(tokenizer.word_tokenize(sentence["raw"]))
                    filename_id = join([filename, "#", string(i - 1)])
                    captions = [captions; [filename_id, caption]]
                end
            end
            j = j + 1
        end
        writedlm(captions_dir, captions)
        writedlm(train_dir, train)
        writedlm(dev_dir, dev)
        writedlm(test_dir, test)   
    end   
    return(captions_dir, train_dir, dev_dir, test_dir)
end
end