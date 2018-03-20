using JSON

dir = "/Users/bihterakyol/Desktop/SPRING2018/COMP541/datasets/COCO/"

raw_captions = JSON.parsefile(string(dir, "annotations/captions_train2014.json"))
