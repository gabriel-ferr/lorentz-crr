#
#               By Gabriel Ferreira
#                   Orientation: Prof. Dr. Thiago de Lima Prado
#                                Prof. Dr. Sergio Roberto Lopes
#
# =================================================================================================
#
using JLD2
using Flux
using Random
using Statistics
using Microstates
using LinearAlgebra
using DifferentialEquations
#
#       Some random settings...
rng = MersenneTwister()
Random.seed!()
#
#   ===================================    LORENZ SETTINGS   ======================================
#       Paramenters ....
const σ = 10.0
const β = (8.0 / 3.0)
const ρ_values = [26.0, 26.5, 27.0, 27.5, 28.0, 28.5]
#       Time interval that DifferentialEquations will use.
#   - 0.0 - 3000.0 give us 37.743 values with default resolution.
const tspan = (0.0, 3000.0)
#
#   =================================     MICROSTATES SETTINGS     ================================
#       The size of our microstates (needs be lower than 8!)
const n = 2
#       How I am going to use the Microstates.jl in some loops, define a power vector before it is important xD
const pvec = power_vector(n)
#       Threshold value.
const ε_range = range(0.0, 40.0, 120)
#
#   =================================       FLUX AND MLP        ===================================
#       Initial values for our Lorenz =3
const init_values = rand(Float64, 3, 1200)
const init_values_test = rand(Float64, 3, 360)
#       Learning rate
const learning_rate = 0.001
#       Epochs of training
const epochs = 500
#
# =================================================================================================
#       Lorenz system
function lorenz!(dr, r, ρ, t)
    x, y, z = r
    dr[1] = σ * (y - x)
    dr[2] = x * (ρ - z) - y
    dr[3] = x * y - β * z
end
# =================================================================================================
#       Calculate the data previous... because if I don't do it this will use much time =<
function get_serie(init_r, ρ)
    problem = ODEProblem(lorenz!, init_r, tspan, ρ)
    return (solve(problem)[:, :])[:, 1001:end]
end
# =================================================================================================
#       I make here a function to generate our data from a initial value.
function get_probs(serie, ε)
    probs, _ = microstates(serie, ε, n; power_aux=pvec, recurrence=Microstates.crd_recurrence)
    return probs
end
# =================================================================================================
#       Calculates the accuracy ...
function calc_accuracy(predict, trusty)
    conf = zeros(Int, length(ρ_values), length(ρ_values))
    sz = size(predict, 2)

    for i = 1:sz
        mx_prd = findmax(predict[:, i])
        mx_trt = findmax(trusty[:, i])
        conf[mx_prd[2], mx_trt[2]] += 1
    end

    return tr(conf) / sum(conf)
end
# =================================================================================================
#       Main function
function main()
    #
    #       Number of samples that we are taking.
    sz = (size(init_values, 2), size(init_values_test, 2))
    #
    #       Checks if we have a saved progress...
    if (!isfile("status.dat"))
        #
        #       If does not exist a previous progress, we create a new xD
        save_object("accuracy.dat", zeros(Float64, length(ε_range), length(ε_range), epochs))
        save_object("loss.dat", zeros(Float64, length(ε_range), length(ε_range), epochs))

        #       Create a new MLP network too =3
        mlp = Chain(
            Dense(2^(n * n) => 128),
            Dense(128 => 64, selu),
            Dense(64 => 32, selu),
            Dense(32 => length(ρ_values)),
            softmax
        )
        mlp = f64(mlp)
        save_object("network.mlp", mlp)

        data_serie = zeros(Float64, 3, 1000, sz[1], length(ρ_values))
        data_serie_test = zeros(Float64, 3, 1000, sz[2], length(ρ_values))

        for p in eachindex(ρ_values)
            Threads.@threads for v in 1:sz[1]
                data_serie[:, :, v, p] .= get_serie(init_values[:, v], ρ_values[p])[:, 35501:36500]
                if (v <= sz[2])
                    data_serie_test[:, :, v, p] .= get_serie(init_values_test[:, v], ρ_values[p])[:, 35501:36500]
                end
            end
        end

        save_object("serie.dat", data_serie)
        save_object("test.dat", data_serie_test)

        save_object("status.dat", [1, 1])
    end

    #       Load the data and previous progress...
    status = load_object("status.dat")
    accuracy = load_object("accuracy.dat")
    loss = load_object("loss.dat")

    dataserie = load_object("serie.dat")
    testserie = load_object("test.dat")
    #
    #       I need to make labels too ...
    labels = ones(sz[1] * length(ρ_values))
    labels_test = ones(sz[2] * length(ρ_values))

    for i in eachindex(ρ_values)
        labels[1+(i-1)*sz[1]:i*sz[1]] .*= ρ_values[i]
        labels_test[1+(i-1)*sz[2]:i*sz[2]] .*= ρ_values[i]
    end

    labels = Flux.onehotbatch(labels, ρ_values)
    labels_test = Flux.onehotbatch(labels_test, ρ_values)
    #
    for thres_max in status[2]:length(ε_range)
        for thres_min in status[1]:thres_max-1
            save_object("status.dat", [thres_min, thres_max])

            data = zeros(Float64, 2^(n * n), sz[1], length(ρ_values))
            data_test = zeros(Float64, 2^(n * n), sz[2], length(ρ_values))

            for p in eachindex(ρ_values)
                Threads.@threads for v in 1:sz[1]
                    data[:, v, p] .= get_probs(dataserie[:, :, v, p], (ε_range[thres_min], ε_range[thres_max]))
                    if (v <= sz[2])
                        data_test[:, v, p] .= get_probs(testserie[:, :, v, p], (ε_range[thres_min], ε_range[thres_max]))
                    end
                end
            end

            data_temp = reshape(data, 2^(n * n), length(ρ_values) * sz[1])
            data_test_temp = reshape(data_test, 2^(n * n), length(ρ_values) * sz[2])

            #
            #       Loads the network...
            model = load_object("network.mlp")
            opt = Flux.setup(Flux.Adam(learning_rate), model)
            #
            loader = Flux.DataLoader((data_temp, labels), batchsize=32, shuffle=true)

            for epc = 1:epochs
                losses = []
                for (x, y) in loader
                    ld_loss, grads = Flux.withgradient(model) do m
                        y_hat = m(x)
                        Flux.logitcrossentropy(y_hat, y)
                    end
                    push!(losses, ld_loss)
                    Flux.update!(opt, model, grads[1])
                end
                accuracy[thres_min, thres_max, epc] = calc_accuracy(model(data_test_temp), labels_test)
                loss[thres_min, thres_max, epc] = mean(losses)
            end

            save_object("accuracy.dat", accuracy)
            save_object("loss.dat", loss)
        end
    end
end
# =================================================================================================
main()