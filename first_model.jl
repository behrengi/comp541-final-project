for p in ("Knet", "PyCall", "Knet")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
include("Data.jl")
include("Read.jl")

using Data
using Read
using PyCall
using Knet
@pyimport nltk.translate.bleu_score as nl


"""
input: caption_length_dict
caption_length_dict: Dict(length1 => [filename, caption; filename; caption, ...])

output: Any[minibatch1(x, y), minibatch2(x, y), ...]

From the original paper (Xu et al., 2016):
Then, during training we randomly sample a length and retrieve a 
mini-batch of size 64 of that length. We found that this greatly 
improved convergence speed with no noticeable diminishment in 
performance. 
"""
function Minibatch(input, w2i, atype; batchsize=64)
    froms = Dict(k => 1 for k in keys(input))
    batched = Any[]
    while !isempty(froms)
        len, from = rand(froms)
        inps = input[len]

        to = min(from + batchsize - 1, length(inps))
        batch = inps[from:to]

        push!(batched, MakeBatch(batch, w2i, atype))
        froms[len] = to + 1
        if to + 1 > length(inps)
            delete!(froms, len)
        end
    end
    return(batched)
end

"""
Makes a single batch.
"""
 function MakeBatch(batch, w2i, atype)
    caption_len = length(batch[1][2])

    captions = [atype(spzeros(length(batch), length(w2i))) for i=1:caption_len]
    for (index, val) in enumerate(batch)
        caption = val[2]
        for t = 1:caption_len
            tok = caption[t] in keys(w2i) ? w2i[caption[t]] : w2i["_UNK_"]
            captions[t][index, tok] = 1
        end
    end
    filenames = first.(batch)
    return (filenames, captions)
end

# image: L x D
# y, 1 of K encoded: 1 x K
# a: L x D
# E: m x K
# L_0: K x m
# L_h: m x n
# L_z: m x D
# z: 1 x D


"""
LSTM weights and biases 

"""
function WInit(atype, embed, dict, feature, place, hidden, winit=0.01)
    w = Array{Any}(13)
    w[1] = xavier(dict, embed) # embedding
    w[2] = xavier(embed + hidden + feature , 4*hidden) # lstm
    w[3] = zeros(1, 4*hidden) # lstm bias
    w[4] = xavier(feature, hidden) # c_init
    w[5] = zeros(1, hidden) # c_init bias
    w[6] = xavier(feature, hidden) # h_init
    w[7] = zeros(1, hidden) # h_init bias
    w[8] = xavier(embed, dict) # l_0 Eq.7
    w[9] = xavier(hidden, embed) # l_h Eq.7
    w[10] = xavier(embed, feature) # l_z Eq.7
    w[11] = xavier(hidden, feature) # attention w1
    w[12] = zeros(feature, place) # attention bias
    w[13] = xavier(feature, 1) # attention w2
    return(map(x -> atype(x), w))
end

"""
param: weights and biases
state: hidden, cell and context
"""
function LSTM(param, state, input)
    w, bias, embed = param[2], param[3], param[1]
    hidden, cell, context = state

    h = size(hidden, 2)
    gates = hcat(input * embed, hidden, context) * w .+ bias
    ingate = sigm.(gates[:, 1:h])
    forget = sigm.(gates[:, h+1:2h])
    outgate = sigm.(gates[:, 2*h+1:3h])
    change = tanh.(gates[:, 3*h+1:4h])
    cell = forget .* cell .+ ingate .* change
    hidden = outgate .* tanh.(cell)
    
    return(hidden, cell)
end



"""
A simple MLP init function.
"""
function Init(weight, bias, input)
    return(relu.(input * weight .+ bias))
end

# hidden * w .+ features .+ bias
# check if dimensions work
"""
Calculates attention weights and returns the context vector. 
"""
function Attention(param, features, hidden)
    w, bias, attention_out = param[11], param[12], param[13]

    combined = relu.(hidden * w .+ features .+ reshape(bias, 1, size(bias)...))
    combined = permutedims(combined, (1, 3, 2))

    alpha = reshape(combined, size(combined, 1) * size(combined, 2), size(combined, 3)) * attention_out
    alpha = reshape(alpha, size(combined, 1), size(combined, 2))
    exp_alpha = exp.(alpha)
    alpha = exp_alpha ./ sum(exp_alpha)

    context = mean(alpha .* permutedims(features, (1, 3, 2)), 2)
    context = reshape(context, size(context, 1), size(context, 3))

    return(context)
end

"""
Probability of each output word. 
"""
function Norm_P(param, input, state)
    l_0, l_h, l_z, embed = param[8], param[9], param[10], param[1]
    hidden, _, context = state
    p = exp.((input * embed .+ hidden * l_h .+ context * l_z) * l_0)
    return(p ./ sum(p, 2))
end

function Loss(param, features, outputs, input, atype)
    c_init, c_bias = param[4], param[5]
    h_init, h_bias = param[6], param[7]
    mean_features = reshape(mean(features, 3), size(features, 1), size(features, 2))

    hidden = Init(h_init, h_bias, mean_features)
    cell = Init(c_init, c_bias, mean_features)
    context = Attention(param, features, hidden)
    state = (hidden, cell, context)

    loss = 0
    for output in outputs
        hidden, cell = LSTM(param, state, input)
        context = Attention(param, features, hidden)
        state = (hidden, cell, context) 
        p_output = Norm_P(param, input, state)
        input = atype(spzeros(size(input)...))
        input[p_output .== maximum(p_output, 2)] = 1

        loss -= sum(output .* log.(p_output)) / size(output, 1)

        if isnan(getval(loss))
            println("output:", output)
            println("p_output:", p_output)
            println("state:", state)
            # println(":", output)
        end
        # println(">", getval(loss), "<")
    end
    # return(input, loss)
    return(loss / length(outputs))
end

gradient = gradloss(Loss)


atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}

data = Data.ImportFlickr8K()
i2w, w2i = data.i2w, data.w2i;
train_captions, test_captions = data.train_captions, data.test_captions
train_features, test_features = data.train_features, data.test_features
EPOCH = 10

println("data importing end")

ytrain = collect(values(train_captions))
ytest = collect(values(test_captions))
xtrain = collect(values(train_features))
xtest = collect(values(test_features))

train_count = Read.CaptionCountDict(train_captions) 
batches = Minibatch(train_count, w2i, atype)
bos = Read.OneHot(w2i["_BOS_"], length(w2i))

w = WInit(atype, 512, length(w2i), 512, 196, 1024)
opt = map(x -> Adam(lr = 0.001), w)
grads = []
loss = []

for i = 1:EPOCH
    sum_loss = 0
    
    for (x, y) in batches
        init_input = atype(repmat(bos, size(x, 1), 1))
        features = permutedims(cat(3, [atype(train_features[x]) for x in x]...), (3, 2, 1))
        typeof(features)
        grads, loss = gradient(w, features, y, init_input, atype)
        sum_loss += loss
        update!(w, grads, opt)
    end
    println("EPOCH: ", i, " " ,sum_loss)
end