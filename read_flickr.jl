using Images
dir = "/Users/bihterakyol/Desktop/SPRING2018/COMP541/datasets/FLICKR8K/"

text_dir = string(dir, "Flickr8k_text/")
image_dir = string(dir, "Flickr8k_Dataset/")

raw_captions = readdlm(string(text_dir, "Flickr8k.lemma.token.txt"), '\t', comments=false)
raw_training = readdlm(string(text_dir, "Flickr_8k.trainImages.txt" ))
raw_dev = readdlm(string(text_dir, "Flickr_8k.devImages.txt" ))
raw_test = readdlm(string(text_dir, "Flickr_8k.testImages.txt" ))

struct FlickrImage
    data
    tokens
end

train_images = FlickrImage[]
dev_images = FlickrImage[]
test_images = FlickrImage[]

tokens = Dict()
for i in 1:size(raw_captions, 1)
    filename, _ = split(raw_captions[i, 1], '#')
    if !(filename in keys(tokens))
        tokens[filename] = Any[]
    end
    push!(tokens[filename], split(raw_captions[i, 2], ' '))
end

for filename in [raw_training; raw_dev; raw_test]
    if filename in raw_training
        images = train_images
    elseif filename in raw_dev
        images = test_images
    else
        images = dev_images
    end
    push!(images, FlickrImage(
        load(string(image_dir, filename)),
        tokens[filename]
        ))
end
