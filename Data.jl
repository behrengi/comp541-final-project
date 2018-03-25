include("Read.jl")

module Data

import Read
import JSON
using JLD
using PyCall
@pyimport nltk.tokenize as tokenizer

struct data
    i2w
    w2i
    train_images
    dev_imaves
    test_images
    train_captions
    dev_captions
    test_captions
end

function Import(raw_captions, raw_train, raw_dev, raw_test, image_dir)
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
    
    println("train images start")
    #train_images = Read.ImageDict(image_dir, raw_train)
    # test_images = Read.ImageDict(image_dir, raw_test)
    # dev_images = Read.ImageDict(image_dir, raw_dev)

    train_images = Array{Any}(1, length(train_captions))
    dev_images = Array{Any}(1, length(dev_captions))
    test_images = Array{Any}(1, length(test_captions))
    println("train images end")

    return(data(i2w, w2i, train_images, dev_images, test_images, 
                train_captions, dev_captions, test_captions))
end


function ImportFlickr8K(dir="/Users/bihterakyol/Desktop/SPRING2018/COMP541/datasets/FLICKR8K/")
    text_dir = string(dir, "Flickr8k_text/")
    image_dir = string(dir, "Flickr8k_Dataset/")

    raw_captions = readdlm(string(text_dir, "Flickr8k.lemma.token.txt"), '\t', comments=false)
    raw_train = readdlm(string(text_dir, "Flickr_8k.trainImages.txt" ))
    raw_dev = readdlm(string(text_dir, "Flickr_8k.devImages.txt" ))
    raw_test = readdlm(string(text_dir, "Flickr_8k.testImages.txt" ))

    Import(raw_captions, raw_train, raw_dev, raw_test, image_dir)
end

function ImportFlickr30K(dir="/Users/bihterakyol/Desktop/SPRING2018/COMP541/datasets/FLICKR30K/")
    image_dir = string(dir, "flickr30k_images/")
    json_dir = string(dir, "dataset.json") 
    
    raw_captions_fn, train_fn, dev_fn, test_fn = JSONtoTxt(json_dir, dir)
    raw_captions = readdlm(raw_captions_fn, '\t', comments=false)
    raw_train = readdlm(train_fn)
    raw_dev = readdlm(dev_fn)
    raw_test = readdlm(test_fn)

    Import(raw_captions, raw_train, raw_dev, raw_test, image_dir)
end

function ImportCOCO(dir="/Users/bihterakyol/Desktop/SPRING2018/COMP541/datasets/COCO/")  
    text_dir = string(dir, "annotations/")
    image_dir = ""
    train_image_dir = string(dir, "train2014/") 
    test_image_dir = string(dir, "test2014/") # val2014???? downloaded wrong image data
    json_dir = string(dir, "dataset.json")

    println("json to txt begin")
    raw_captions_fn, train_fn, dev_fn, test_fn = JSONtoTxt(json_dir, dir)
    println("json to txt end")

    println("readdlm begin")
    raw_captions = readdlm(raw_captions_fn, '\t', comments=false)
    raw_train = readdlm(train_fn)
    raw_dev = readdlm(dev_fn)
    raw_test = readdlm(test_fn)
    println("readdlm end")

    Import(raw_captions, raw_train, raw_dev, raw_test, image_dir)
end

"""
Function to turn Flickr30k and COCO dataset json files Flickr8k compatible.
This function wil create four different txt files, and return their filenames. 
1st will have the format: imagefilename#id caption
It will also parse tokens using nltk to match Flickr8k parsing.
2nd, 3rd, 4th will have the imagefilenames for train, dev and test datasets respectively.
"""
function JSONtoTxt(json_dir, save_dir)
    rawcaptions_filename = string(save_dir, "/", "tokens.txt")
    train_filename = string(save_dir, "/", "train.txt")
    dev_filename = string(save_dir, "/", "dev.txt")
    test_filename = string(save_dir, "/", "test.txt")
    
    if !(all([isfile(file) for file in [train_filename, dev_filename, test_filename]]))
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
                    captions = vcat(captions, reshape([filename_id, caption], 1, 2))
                end
            end
            j = j + 1
        end
        writedlm(rawcaptions_filename, captions)
        writedlm(train_filename, train)
        writedlm(dev_filename, dev)
        writedlm(test_filename, test)   
    end   
    return(rawcaptions_filename, train_filename, dev_filename, test_filename)
end

end