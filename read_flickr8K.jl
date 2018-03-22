include("Read.jl")
import Read

dir = "/Users/bihterakyol/Desktop/SPRING2018/COMP541/datasets/FLICKR8K/"

text_dir = string(dir, "Flickr8k_text/")
image_dir = string(dir, "Flickr8k_Dataset/")

raw_captions = readdlm(string(text_dir, "Flickr8k.lemma.token.txt"), '\t', comments=false)
raw_training = readdlm(string(text_dir, "Flickr_8k.trainImages.txt" ))
raw_dev = readdlm(string(text_dir, "Flickr_8k.devImages.txt" ))
raw_test = readdlm(string(text_dir, "Flickr_8k.testImages.txt" ))

tokens = Read.TokensDict(raw_captions)
(i2w, wi2) = Read.CreateDictionaries(tokens)

# train = Read.ImageTokens(image_dir, raw_training, tokens)
# dev = Read.ImageTokens(image_dir, raw_dev, tokens)
# test = Read.ImageTokens(tokens, image_dir, raw_test)
