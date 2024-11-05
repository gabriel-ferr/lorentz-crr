#
#               By Gabriel Ferreira
#                   Orientation: Prof. Dr. Thiago de Lima Prado
#                                Prof. Dr. Sergio Roberto Lopes
#
# =================================================================================================
#
#       I'll use Microstates.jl to work with microstates of recurrence. If you want 
#   read about: https://github.com/gabriel-ferr/Microstates.jl
#
# =================================================================================================
include("Microstates.jl/microstates.jl")
# =================================================================================================
using .Microstates
using Colors
using CairoMakie
using Statistics
using ProgressMeter
# =================================================================================================
const ε_range = range(0, 40, 1000)
# =================================================================================================
#       Lorentz function =3
#           Okay, I try use DynamicalSystems but I really don't understood how I use it >.<
#   So, I will use the Numerical Computation that Strogatz teaches in Nonlinear Dynamics and Chaos
#   and make it by myself =3
function lorentz_step(σ, β, ρ, entry, Δt)
    function f_dx(vector)
        return σ * (vector[2] - vector[1])
    end
    function f_dy(vector)
        return vector[1] * (ρ - vector[3]) - vector[2]
    end
    function f_dz(vector)
        return vector[1] * vector[2] - β * vector[3]
    end

    x = entry[1] + integrator(f_dx, entry, Δt)
    y = entry[2] + integrator(f_dy, entry, Δt)
    z = entry[3] + integrator(f_dz, entry, Δt)

    return [x, y, z]
end

function integrator(func, vector, Δt)
    k1 = func(vector) * Δt
    k2 = func(vector .+ ((1 / 2) * k1)) * Δt
    k3 = func(vector .+ ((1 / 2) * k2)) * Δt
    k4 = func(vector .+ k3) * Δt
    return (1 / 6) * (k1 .+ (2 * k2) .+ (2 * k3) .+ k4)
end

function lorentz(start, σ, β, ρ, times)
    serie = [start]
    for t = 1:times
        push!(serie, lorentz_step(σ, β, ρ, serie[t], 0.01))
    end

    result = zeros(Float64, length(serie), 3)
    for i in eachindex(serie)
        result[i, :] .= serie[i]
    end

    return result
end
# =================================================================================================
data = lorentz([0.1, 0.2, 0.3], 12, 8 / 4, 22, 100000)
#fig = Figure()
#ax = Axis3(fig[1, 1], aspect=:equal, azimuth=-0.1 * pi)
#lines!(data)
#fig
# =================================================================================================
function color_scheme()
    colors = ones(RGB{Float64}, 10000)
    red_interval_1 = range(66, 255, 4999)
    green_interval_1 = range(0, 55, 4999)
    for i = 2:5000
        #       From RGB(68, 0, 0)
        #       To RGB(255, 55, 0)
        colors[i] = RGB{Float64}(red_interval_1[i-1] / 255, green_interval_1[i-1] / 255, 0)
    end

    green_interval_2 = range(55, 255, 3000)
    for i = 5001:8000
        colors[i] = RGB{Float64}(1, green_interval_2[i-5000] / 255, 0)
    end

    red_interval_2 = range(255, 0, 1000)
    for i = 8001:9000
        colors[i] = RGB{Float64}(red_interval_2[i-8000] / 255, 1, 0)
    end

    blue_interval_1 = range(0, 255, 900)
    for i = 9001:9900
        colors[i] = RGB{Float64}(0, 1, blue_interval_1[i-9000] / 255)
    end

    red_interval_3 = range(0, 255, 100)
    green_interval_3 = range(255, 0, 100)

    for i = 9901:10000
        colors[i] = RGB{Float64}(red_interval_3[i-9900] / 255, green_interval_3[i-9900] / 255, 1)
    end

    return colors
end
# =================================================================================================
function main()
    etr = zeros(Float64, length(ε_range), length(ε_range))
    @showprogress for i in eachindex(ε_range)
        Threads.@threads for j in 1:i-1
            probs, _ = microstates(data, (ε_range[j], ε_range[i]), 2; recurrence=Microstates.corridor_recurrence)
            etr[i, j] = entropy(probs)
        end
    end
    return etr
end
# =================================================================================================
data = main()

fig = Figure()
ax = Axis(fig[1, 1], title="Lorentz System", xlabel="ε max", ylabel="ε min")
hm = heatmap!(ε_range, ε_range, data, colormap=color_scheme(), colorrange=(0.0, 1))
Colorbar(fig[1, 2], hm)

save("fig-lor-etr.png", fig)
fig