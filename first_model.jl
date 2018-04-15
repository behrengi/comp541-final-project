for p in ("PyCall", "Knet", "JLD")
    Pkg.installed(p) == nothing && Pkg.add(p)
end
include("Data.jl")
include("Read.jl")

using Data
using Read
using PyCall
using Knet
using JLD
@pyimport nltk.translate.bleu_score as nl


"""
Makes a single batch.
"""
function MakeBatch(batch, w2i)
    caption_len = length(batch[1][2])

    captions = [spzeros(length(batch), length(w2i)) for i in 1:caption_len]
    for (index, val) in enumerate(batch)
        caption = val[2]
        for t in 1:caption_len
            tok = w2i[haskey(w2i, caption[t]) ? caption[t] : "_UNK_"]
            captions[t][index, tok] = 1
        end
    end
    filenames = first.(batch)
    return filenames, captions
end

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
function Minibatch(input, w2i, batchsize=64)
    froms = Dict(k => 1 for k in keys(input))
    for (len, data) in input
        input[len] = data[randperm(length(data)), :]
    end

    batched = Any[]
    while !isempty(froms)
        len, from = rand(froms)
        inps = input[len]

        to = min(from + batchsize - 1, length(inps))
        batch = inps[from:to]

        push!(batched, MakeBatch(batch, w2i))

        froms[len] = to + 1
        if to + 1 > length(inps)
            delete!(froms, len)
        end
    end
    return batched
end

# function Seq(caption, w2i)

# end

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
function WInit(atype, embed, dict, feature, place, hidden)
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
    w[10] = xavier(feature, embed) # l_z Eq.7
    w[11] = xavier(hidden, feature) # attention w1
    w[12] = zeros(feature, place) # attention bias
    w[13] = xavier(feature, 1) # attention w2
    return map(x -> atype(x), w)
end

"""
param: weights and biases
state: hidden, cell and context
"""
function LSTM(param, state, input, drop)
    w, bias, embed = param[2], param[3], param[1]
    hidden, cell, context = state

    h = size(hidden, 2)
    gates = hcat(embed[input, :], hidden, context) * w .+ bias
    ingate = sigm.(gates[:, 1:h])
    forget = sigm.(gates[:, h+1:2h])
    outgate = sigm.(gates[:, 2*h+1:3h])
    change = tanh.(gates[:, 3*h+1:4h])

    ingate, forget, outgate = (dropout(gate, drop) for gate in (ingate, forget, outgate))

    cell = forget .* cell .+ ingate .* change
    hidden = outgate .* tanh.(cell)

    return hidden, cell
end

# function CaptionSeq(captions, w2i)
#     seq = Any[]
#     for i in 1:length(x)
#         push!(batched, MakeBatch((x[i], ))
#     end
# end

"""
A simple MLP init function.
"""
function Init(weight, bias, input)
    return relu.(input * weight .+ bias)
end

# hidden * w .+ features .+ bias
# check if dimensions work
"""
Calculates attention weights and returns the context vector.
"""
function Attention(param, features, hidden)
    wei, bias, attention_out = param[11], param[12], param[13];

    combined = features .+ hidden * wei .+ reshape(bias, 1, size(bias)...);
    combined = relu.(combined);
    combined = KnetArray{Float32}(permutedims(Array{Float32}(getval(combined)), [1, 3, 2]));

    s1, s2 = size(combined);

    alpha = mat(combined) * attention_out
    alpha = reshape(alpha, s1, s2);
    exp_alpha = exp.(alpha .- maximum(alpha, 2))
    alpha = exp_alpha ./ sum(exp_alpha, 2)

    context = mean(alpha .* KnetArray{Float32}(permutedims(Array{Float32}(getval(features)), (1, 3, 2))), 2)
    context = reshape(context, size(context, 1), size(context, 3))

    return context
end

"""
Probability of each output word.
"""
function P(param, input, state)
    l_0, l_h, l_z, embed = param[8], param[9], param[10], param[1]
    hidden, _, context = state
    p = (embed[input, :] .+ hidden * l_h .+ context * l_z) * l_0
    p = exp.(p .- maximum(p, 2))

    return p ./ sum(p, 2)
end

function RowMax(x)
    return Int.(ceil.(vec(findmax(x, 2)[2]) ./ size(x, 1)))
end

function Loss(param, features, outputs, input, drop_lstm, drop, atype, i2w)
    c_init, c_bias = param[4], param[5]
    h_init, h_bias = param[6], param[7]

    mean_features = reshape(mean(features, 3), size(features, 1), size(features, 2))
    hidden = Init(h_init, h_bias, mean_features)
    cell = Init(c_init, c_bias, mean_features)

    state = (hidden, cell, Attention(param, features, hidden))

    eps_array = atype([eps(Float32)])
    loss = 0
    for output in outputs
        hidden, cell = LSTM(param, state, input, drop_lstm)
        hidden = dropout(hidden, drop)
        context = Attention(param, features, hidden)
        context = dropout(context, drop)
        state = (hidden, cell, context)

        p_output = P(param, input, state)

        p_output_nonknet = Array{Float32}(getval(p_output))
        input = RowMax(p_output_nonknet)

        # for i in 1:size(output, 1)
        #     println(i, ": ", i2w[output[i, :] .== 1][1], " - ")
        #     for s in sortperm(p_output_nonknet[i, :], rev=true)[1:4]
        #         println("\t", p_output_nonknet[i, s], ": ", i2w[s])
        #     end
        # end
        # readline(STDIN)

        loss += sum(atype(output) .* log.(p_output .+ eps_array))
    end

    return -loss
end

function Predict(param, features, i2w, eos, bos)
    c_init, c_bias = param[4], param[5]
    h_init, h_bias = param[6], param[7]

    mean_features = reshape(mean(features, 3), size(features, 1), size(features, 2))
    hidden = Init(h_init, h_bias, mean_features)
    cell = Init(c_init, c_bias, mean_features)

    state = (hidden, cell, Attention(param, features, hidden))

    n, dict_size = size(features, 1), length(i2w)
    input = repmat(bos, n)

    captions = [[] for _ in 1:n]
    is_eos = BitArray(n)
    while !all(is_eos)
        hidden, cell = LSTM(param, state, input, 0)
        state = (hidden, cell, Attention(param, features, hidden))

        p_output = P(param, input, state)
        input = RowMax(Array{Float32}(getval(p_output)))
        for i in 1:n
            if input[i] == eos
                is_eos[i] = true
                push!(captions[i], "_EOS_")
            elseif !is_eos[i]
                push!(captions[i], i2w[input[i]])
                if length(captions[i]) > 50
                    is_eos[i] = true
                end
            end
        end
    end
    return captions
end

function Predict(param, features, i2w, eos, bos, splitsize, atype)
    captions = []
    for from in 1:splitsize:size(features, 1)
        to = min(from + splitsize - 1, size(features, 1))
        _features = features[from:to, :, :]
        captions = [captions; Predict(param, atype(_features), i2w, eos, bos)]
    end
    return captions
end

function ROUTINE()
    setseed(1)
    atype = gpu() >= 0 ? KnetArray{Float32} : Array{Float32}

    data = Data.ImportFlickr8K()
    i2w, w2i = data.i2w, data.w2i

    train_captions = data.train_captions
    test_captions = data.test_captions
    dev_captions = data.dev_captions

    train_features = data.train_features
    test_features = data.test_features
    dev_features = data.dev_features

    xtrain = permutedims(cat(3, values(train_features)...), [3, 2, 1])
    xtest = permutedims(cat(3, values(test_features)...), [3, 2, 1])
    xdev = permutedims(cat(3, values(dev_features)...), [3, 2, 1])

    ytrain = collect(values(train_captions))
    ytest = collect(values(test_captions))
    ydev = collect(values(dev_captions))

    println("End of data importing")

    length_train = Read.CaptionCountDict(train_captions) # captions by length
    dict_length = length(w2i)

    # end of sentence and beginning of sentence tokens
    bos = [w2i["_BOS_"]]
    eos = w2i["_EOS_"]

    dict = length(w2i) # dictionary size
    feature = 512 # first dim of features
    embed = feature # word embedding size
    place = 196 # second dim of features
    hidden = 1800 # lstm size
    drop_lstm = 0 # dropout to lstm (i, f, o)
    drop = 0 # dropout to the context and hidden vectors
    lr = 0.001 # learning rate
    bs = 64 # batch-size
    EPOCHS = 3
    train_bleus = []
    test_bleus = []
    train_loss = []
    final_test_bleu = []
    final_train_bleu = []
    weights = []
    bleu_type = eye(4, 4)

    w = WInit(atype, embed, dict, feature, place, hidden)
    opt = map(x -> Rmsprop(lr = lr), w)

    test_predict = nothing
    train_predict = nothing

    gradient = gradloss(Loss)

    println("training starts")
    for i in 1:EPOCHS
        test_predict = nothing
        train_predict = nothing
        sum_loss = 0


        batches = Minibatch(length_train, w2i, bs)
        b = 0
        @time for (x, y) in batches
            caption_length = length(y)
            if caption_length > 5
                continue
            end
            b += 1

            n_caption = length(x)

            features = atype(permutedims(cat(3, [train_features[name] for name in x]...), [3, 2, 1]))
            input = Int.(repmat(bos, n_caption))

            grads, loss = gradient(w, features, y, input, drop_lstm, drop, atype, i2w)
            sum_loss += loss / n_caption / caption_length
            update!(w, grads, opt)
        end

        epoch_loss = sum_loss / b
        println("EPOCH: ", i, " " , epoch_loss)



        # train_loss = push!(train_loss, Loss(w, xtrain, ytrain, bos, 0, 0, Array{Float32}) /
        #                                length(xtrain))
        # test_loss = push!(test_loss,
        #                  Loss(w, xtest, ytest, bos, 0, 0, Array{Float32}) /
        #                  length(ytest))

        # println("predictions are being made")
        train_predict = Predict(w, xtrain, i2w, eos, bos, 1000, atype)
        # println("train done, doing test")
        test_predict = Predict(w, xtest, i2w, eos, bos, 1000, atype)
        # println("predictions are done")

        # println("bleu score")
        train_bleu = nl.corpus_bleu(ytrain, train_predict)
        test_bleu = nl.corpus_bleu(ytest, test_predict)
        # println(train_bleu, " ", test_bleu)
        push!(train_bleus, train_bleu)
        push!(test_bleus, test_bleu)
        push!(weights, map(x -> Array{Float32}(x), w))
        # println("bleu scores are saved")

        println("train bleu: ", train_bleu)
        println("test bleu: ", test_bleu)

        push!(train_loss, Float32(epoch_loss))
        # println("loss pushed")
    end

    println("saving model")
    JLD.save(joinpath(pwd(), "loss.jld"), "train_loss", train_loss, "weights", weights)
    println("loss saved")


    for i in 1:size(bleu_type, 1)
        push!(final_train_bleu, nl.corpus_bleu(ytrain, train_predict, bleu_type[i, :]))
        push!(final_test_bleu, nl.corpus_bleu(ytest, test_predict, bleu_type[i, :]))
    end

    JLD.save(joinpath(pwd(), "bleu.jld"), "train_bleus", train_bleus, "test_bleus", test_bleus,
                                        "final_train_bleu", final_train_bleu, "final_test_bleu", final_test_bleu)
end

ROUTINE()
