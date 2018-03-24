include("Read.jl")

module Flickr8K
import Read

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

function Import()

    dir = "/Users/bihterakyol/Desktop/SPRING2018/COMP541/datasets/FLICKR8K/"

    text_dir = string(dir, "Flickr8k_text/")
    image_dir = string(dir, "Flickr8k_Dataset/")

    raw_captions = readdlm(string(text_dir, "Flickr8k.lemma.token.txt"), '\t', comments=false)
    raw_train = readdlm(string(text_dir, "Flickr_8k.trainImages.txt" ))
    raw_dev = readdlm(string(text_dir, "Flickr_8k.devImages.txt" ))
    raw_test = readdlm(string(text_dir, "Flickr_8k.testImages.txt" ))

    captions_dict = Read.TokensDict(raw_captions)
    (i2w, w2i) = Read.CreateDictionaries(captions_dict)
    
    # train = Read.ImageTokens(image_dir, raw_train, tokens)
    # dev = Read.ImageTokens(image_dir, raw_dev, tokens)
    # test = Read.ImageTokens(tokens, image_dir, raw_test)
    
    ######################################
    # This is just a temporary thing, will 
    # be updated with the code above
    ######################################
    train_images = raw_train
    dev_images = raw_dev
    test_images = raw_test

    train_captions, dev_captions, test_captions = (filter((k, v) -> k in dataset, captions_dict)
                                                    for dataset in [raw_train, raw_dev, raw_test])

    return(data(i2w, w2i, train_images, dev_images, test_images, 
                train_captions, dev_captions, test_captions))
end

end