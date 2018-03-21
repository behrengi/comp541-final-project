include("Read.jl")

import Read
using JSON
using PyCall

@pyimport nltk
dir = "/Users/bihterakyol/Desktop/SPRING2018/COMP541/datasets/COCO/"
text_dir = string(dir, "annotations/")
train_image_dir = string(dir, "train2014")
val_image_dir = string(dir, "test2014") # val2014???? downloaded wrong image data

######################################
# Annotations needs some preprocessing
# ready to be nltk-split with simple
# space-split.
######################################
raw_JSON_train = JSON.parsefile(string(text_dir, "captions_train2014.json"))
raw_JSON_val = JSON.parsefile(string(text_dir, "captions_val2014.json"))
raw_JSON = merge(raw_JSON_train, raw_JSON_val)

raw_captions = []
tokens = Dict()
for image in values(raw_JSON["images"])
    tokens[image["id"]] = (image["file_name"], Any[])
end

for annotation in values(raw_JSON["annotations"])
    push!(tokens[annotation["image_id"]][2], split(lowercase(annotation["caption"])))
end
tokens = Dict(collect(values(tokens)))

i2w, wi2 = Read.CreateDictionaries(tokens)
# train = Read.ImageTokens(train_image_dir, raw_training, tokens)
# val = Read.ImageTokens(val_image_dir, raw_dev, tokens)