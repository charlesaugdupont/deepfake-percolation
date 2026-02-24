# ================================================
# deepfake_percolation.jl
# ================================================
using Graphs
using DataStructures
using StatsBase
using Random
using ProgressMeter
using DataFrames
using Arrow
using Statistics
using Dates

# ================================================
# CONFIGURATION
# ================================================
Base.@kwdef struct SimConfig
    random_seed::Int = 42
    num_agents::Int = 1000
    num_connections::Int = 2
    global_fake_prob::Float64 = 0.5
    verification_cost::Float64 = 1.0 
    rep_gain::Float64 = 1.0
    rep_loss::Float64 = 2.0
    verification_tpr::Float64 = 0.90
    verification_fpr::Float64 = 0.10
    prior_min::Float64 = 1.0
    prior_max::Float64 = 5.0
    enable_rewiring::Bool = true
    rewiring_interval::Int = 10
end

mutable struct BetaDist
    alpha::Float64
    beta::Float64
end

# ================================================
# AGENT LOGIC
# ================================================
module AgentLogic
    using ..Main: SimConfig, BetaDist
    compute_beta_mean(b::BetaDist) = b.alpha / (b.alpha + b.beta)
    compute_beta_mean(alpha::Float64, beta::Float64) = alpha / (alpha + beta)
    function calculate_posterior_truth_prob(reliability::Float64, global_true_belief::Float64)
        p_false_global = 1.0 - global_true_belief
        numerator = reliability * global_true_belief
        denominator = (reliability * global_true_belief) + ((1.0 - reliability) * p_false_global)
        return denominator == 0.0 ? 0.0 : numerator / denominator
    end
    function decide(p_true::Float64, reach::Int, est_tpr::Float64, est_fpr::Float64, cfg::SimConfig)
        k = Float64(reach)
        u_share_if_true = cfg.rep_gain * k
        u_share_if_fake = -cfg.rep_loss * k
        u_share = (p_true * u_share_if_true) + ((1.0 - p_true) * u_share_if_fake)
        eu_given_real = (1.0 - est_fpr) * (cfg.rep_gain * k)
        eu_given_fake = (1.0 - est_tpr) * (-cfg.rep_loss * k)
        u_verify = -cfg.verification_cost + (p_true * eu_given_real) + ((1.0 - p_true) * eu_given_fake)
        if u_share > 0.0
            return u_share > u_verify ? :SHARE : :VERIFY
        else
            return 0.0 >= u_verify ? :DISCARD : :VERIFY
        end
    end
end

# ================================================
# ENVIRONMENT
# ================================================
mutable struct SocialNetwork
    cfg::SimConfig
    rng::AbstractRNG
    G::SimpleGraph{Int}
    reputation::Vector{Float64}
    true_belief::Vector{BetaDist}
    tech_tpr_belief::Vector{BetaDist}
    tech_fpr_belief::Vector{BetaDist}
    neighbor_trust::Vector{Dict{Int, BetaDist}}
    base_neighbor_alpha::Vector{Float64}
    base_neighbor_beta::Vector{Float64}
end

function SocialNetwork(cfg::SimConfig, rng::AbstractRNG)
    G = barabasi_albert(cfg.num_agents, cfg.num_connections; rng=rng)
    rand_prior() = cfg.prior_min + rand(rng) * (cfg.prior_max - cfg.prior_min)
    reputation = zeros(Float64, cfg.num_agents)
    true_belief = [BetaDist(rand_prior(), rand_prior()) for _ in 1:cfg.num_agents]
    tech_tpr_belief = [BetaDist(rand_prior(), rand_prior()) for _ in 1:cfg.num_agents]
    tech_fpr_belief = [BetaDist(rand_prior(), rand_prior()) for _ in 1:cfg.num_agents]
    base_neighbor_alpha = [rand_prior() for _ in 1:cfg.num_agents]
    base_neighbor_beta = [rand_prior() for _ in 1:cfg.num_agents]
    neighbor_trust = [Dict{Int, BetaDist}() for _ in 1:cfg.num_agents]
    net = SocialNetwork(cfg, rng, G, reputation, true_belief, tech_tpr_belief, tech_fpr_belief, neighbor_trust, base_neighbor_alpha, base_neighbor_beta)
    for e in edges(G)
        u, v = src(e), dst(e)
        _init_trust_one_way!(net, u, v)
        _init_trust_one_way!(net, v, u)
    end
    return net
end

function _init_trust_one_way!(net::SocialNetwork, observer::Int, target::Int)
    net.neighbor_trust[observer][target] = BetaDist(net.base_neighbor_alpha[observer], net.base_neighbor_beta[observer])
end

function update_trust!(net::SocialNetwork, observer::Int, sender::Int, sender_was_truthful::Bool)
    if haskey(net.neighbor_trust[observer], sender)
        stats = net.neighbor_trust[observer][sender]
        sender_was_truthful ? (stats.alpha += 1.0) : (stats.beta += 1.0)
    end
end

function update_global_belief!(net::SocialNetwork, agent_id::Int, item_was_fake::Bool)
    stats = net.true_belief[agent_id]
    item_was_fake ? (stats.beta += 1.0) : (stats.alpha += 1.0)
end

function rewire_network!(net::SocialNetwork)
    edges_to_check = collect(edges(net.G))
    if isempty(edges_to_check) return end
    min_rep, max_rep = minimum(net.reputation), maximum(net.reputation)
    weights = 1e-6 .+ (net.reputation .- min_rep) ./ (max_rep - min_rep + 1e-6)
    wv = Weights(weights)
    all_nodes = 1:nv(net.G)

    for e in edges_to_check
        u, v = src(e), dst(e)
        if !has_edge(net.G, u, v) continue end
        trust_u_on_v = get(net.neighbor_trust[u], v, nothing)
        trust_v_on_u = get(net.neighbor_trust[v], u, nothing)
        if trust_u_on_v === nothing || trust_v_on_u === nothing continue end

        rel_u_v = AgentLogic.compute_beta_mean(trust_u_on_v)
        rel_v_u = AgentLogic.compute_beta_mean(trust_v_on_u)
        weakest_trust_val, rewiring_node, passive_node = rel_u_v < rel_v_u ? (rel_u_v, u, v) : (rel_v_u, v, u)

        if rand(net.rng) < (1.0 - weakest_trust_val)
            rem_edge!(net.G, rewiring_node, passive_node)
            delete!(net.neighbor_trust[rewiring_node], passive_node)
            delete!(net.neighbor_trust[passive_node], rewiring_node)
            target = degree(net.G, passive_node) == 0 ? passive_node : rewiring_node
            avoid = degree(net.G, passive_node) == 0 ? rewiring_node : passive_node
            _rewire_node!(net, target, all_nodes, wv, avoid)
        end
    end
end

function _rewire_node!(net::SocialNetwork, u::Int, all_nodes, wv::Weights, node_to_avoid::Int)
    invalid_set = Set(neighbors(net.G, u))
    push!(invalid_set, u, node_to_avoid)
    candidate_indices = sample(net.rng, all_nodes, wv, 20, replace=true)
    for candidate in candidate_indices
        if !(candidate in invalid_set)
            add_edge!(net.G, u, candidate)
            _init_trust_one_way!(net, u, candidate)
            _init_trust_one_way!(net, candidate, u)
            return
        end
    end
end

# ================================================
# SIMULATION
# ================================================
mutable struct Simulation
    cfg::SimConfig
    rng::AbstractRNG
    net::SocialNetwork
    cascade_count::Int
    all_nodes::Vector{Int}
end

function Simulation(cfg::SimConfig)
    rng = Xoshiro(cfg.random_seed)
    net = SocialNetwork(cfg, rng)
    return Simulation(cfg, rng, net, 0, collect(1:cfg.num_agents))
end

function run_single_cascade!(sim::Simulation)
    sim.cascade_count += 1
    item_is_fake = rand(sim.rng) < sim.cfg.global_fake_prob
    seed_node = rand(sim.rng, sim.all_nodes)
    seed_degree = degree(sim.net.G, seed_node)
    
    # Queue now stores: (current_node, sender_node, depth_in_tree)
    queue = Queue{Tuple{Int, Int, Int}}()
    enqueue!(queue, (seed_node, 0, 0))
    visited = Set{Int}([seed_node])
    parent_map = Dict{Int, Int}()
    
    actions_log = Dict{Int, Tuple{Symbol, Int}}()
    verification_log = Vector{Tuple{Int, Bool}}()
    
    n_shares, n_verifications, n_receivers = 0, 0, 0
    total_payoff = 0.0
    max_depth = 0

    actual_tpr, actual_fpr = sim.cfg.verification_tpr, sim.cfg.verification_fpr

    while !isempty(queue)
        current, sender, depth = dequeue!(queue)
        n_receivers += 1
        max_depth = max(max_depth, depth)
        
        if sender == 0
            reliability = 0.5
        else
            if haskey(sim.net.neighbor_trust[current], sender)
                reliability = AgentLogic.compute_beta_mean(sim.net.neighbor_trust[current][sender])
            else
                reliability = AgentLogic.compute_beta_mean(sim.net.base_neighbor_alpha[current], sim.net.base_neighbor_beta[current])
            end
        end

        global_true_est = AgentLogic.compute_beta_mean(sim.net.true_belief[current])
        p_true = AgentLogic.calculate_posterior_truth_prob(reliability, global_true_est)
        est_tpr = AgentLogic.compute_beta_mean(sim.net.tech_tpr_belief[current])
        est_fpr = AgentLogic.compute_beta_mean(sim.net.tech_fpr_belief[current])
        reach = degree(sim.net.G, current)
        
        action = AgentLogic.decide(p_true, reach, est_tpr, est_fpr, sim.cfg)
        
        if action != :DISCARD
            is_verifying = (action == :VERIFY)
            tool_says_fake = false
            
            if is_verifying
                n_verifications += 1
                total_payoff -= sim.cfg.verification_cost
                tool_says_fake = item_is_fake ? (rand(sim.rng) < actual_tpr) : (rand(sim.rng) < actual_fpr)
                push!(verification_log, (current, tool_says_fake))
            end

            if !tool_says_fake
                n_shares += 1
                outcome_type = item_is_fake ? :shared_fake : :shared_true
                actions_log[current] = (outcome_type, reach)
                total_payoff += item_is_fake ? -(sim.cfg.rep_loss * reach) : (sim.cfg.rep_gain * reach)

                for nbr in neighbors(sim.net.G, current)
                    if !(nbr in visited)
                        push!(visited, nbr)
                        enqueue!(queue, (nbr, current, depth + 1))
                        parent_map[nbr] = current
                    end
                end
            end
        end
    end

    _apply_updates!(sim, actions_log, verification_log, visited, parent_map, item_is_fake)

    if sim.cfg.enable_rewiring && sim.cascade_count % sim.cfg.rewiring_interval == 0
        rewire_network!(sim.net)
    end

    return (is_fake = item_is_fake, size = length(visited), shares = n_shares, 
            verifications = n_verifications, receivers = n_receivers, 
            utility = total_payoff, seed_node = seed_node, seed_degree = seed_degree, max_depth = max_depth)
end

function _apply_updates!(sim::Simulation, actions_log, verification_log, visited, parent_map, item_is_fake)
    for (agent, (action, marginal_reach)) in actions_log
        if action == :shared_true
            sim.net.reputation[agent] += (sim.cfg.rep_gain * marginal_reach)
        elseif action == :shared_fake
            sim.net.reputation[agent] -= (sim.cfg.rep_loss * marginal_reach)
        end
    end
    for agent in visited
        update_global_belief!(sim.net, agent, item_is_fake)
    end
    for (receiver, sender) in parent_map
        update_trust!(sim.net, receiver, sender, !item_is_fake)
    end
    for (agent_id, tool_said_fake) in verification_log
        tpr_stats, fpr_stats = sim.net.tech_tpr_belief[agent_id], sim.net.tech_fpr_belief[agent_id]
        if item_is_fake
            tool_said_fake ? (tpr_stats.alpha += 1.0) : (tpr_stats.beta += 1.0)
        else
            tool_said_fake ? (fpr_stats.alpha += 1.0) : (fpr_stats.beta += 1.0)
        end
    end
end

# ================================================
# SWEEP & EXPORT LOGIC
# ================================================
function latin_hypercube_sample(n_samples, bounds; rng=Random.default_rng())
    n_vars = length(bounds)
    samples = zeros(Float64, n_samples, n_vars)
    for i in 1:n_vars
        lb, ub = bounds[i]
        pts = (collect(0:(n_samples-1)) .+ rand(rng, n_samples)) ./ n_samples
        pts = lb .+ pts .* (ub - lb)
        shuffle!(rng, pts)
        samples[:, i] = pts
    end
    return samples
end

function run_lhs_sweep()
    # Ensure data directory exists
    timestamp = Dates.format(now(), "yyyy_mm_dd_HHMM")
    out_dir = joinpath("data", "sweep_$timestamp")
    mkpath(out_dir)

    bounds = [
        (0.10, 5.00), # 1: verification cost
        (0.10, 5.00), # 2: rep_gain
        (0.10, 5.00), # 3: rep_loss
        (0.60, 0.95), # 4: verification_tpr
        (0.05, 0.40), # 5: verification_fpr
        (0.10, 0.90), # 6: global_fake_prob
        (2.00, 6.99)  # 7: num_connections
    ]
    
    NUM_LHS_POINTS = 1000
    NUM_REPS = 30
    CASCADES_PER_SIM = 100
    N_AGENTS = 2000

    TOTAL_SIMS = NUM_LHS_POINTS * NUM_REPS

    # Save Experiment Metadata
    open(joinpath(out_dir, "experiment_info.txt"), "w") do io
        println(io, "Deepfake Percolation ABM Sweep")
        println(io, "==============================")
        println(io, "Date: ", now())
        println(io, "Julia Threads: ", Threads.nthreads())
        println(io, "Total LHS Points: ", NUM_LHS_POINTS)
        println(io, "Reps per point: ", NUM_REPS)
        println(io, "Agents (N): ", N_AGENTS)
        println(io, "Cascades per Sim: ", CASCADES_PER_SIM)
        println(io, "\nParameter Bounds:")
        println(io, bounds)
    end

    master_rng = Xoshiro(42)
    lhs_points = latin_hypercube_sample(NUM_LHS_POINTS, bounds; rng=master_rng)

    # Arrays to hold thread outputs
    sim_dfs = Vector{DataFrame}(undef, TOTAL_SIMS)
    cascades_dfs = Vector{DataFrame}(undef, TOTAL_SIMS)
    node_dfs = Vector{DataFrame}(undef, TOTAL_SIMS)
    edge_dfs = Vector{DataFrame}(undef, TOTAL_SIMS)

    println("Saving experiment to: $out_dir/")
    println("Generating $NUM_LHS_POINTS LHS combinations...")
    println("Running $NUM_REPS stochastic reps per combination -> $TOTAL_SIMS total simulations.")
    println("Using $(Threads.nthreads()) threads...")

    prog = Progress(TOTAL_SIMS, 1, "Sweeping... ")
    lk = ReentrantLock()

    Threads.@threads for idx in 1:TOTAL_SIMS
        lhs_id = div(idx - 1, NUM_REPS) + 1
        rep_id = mod(idx - 1, NUM_REPS) + 1

        cost, gain, loss = lhs_points[lhs_id, 1], lhs_points[lhs_id, 2], lhs_points[lhs_id, 3]
        tpr, fpr   = lhs_points[lhs_id, 4], lhs_points[lhs_id, 5]
        prev       = lhs_points[lhs_id, 6]
        m          = Int(floor(lhs_points[lhs_id, 7]))
        
        thread_seed = rand(master_rng, 1:Int(1e9)) + idx
        
        cfg = SimConfig(
            num_agents=N_AGENTS, num_connections=m, global_fake_prob=prev, verification_cost=cost,
            rep_gain=gain, rep_loss=loss, verification_tpr=tpr, verification_fpr=fpr,
            random_seed=thread_seed
        )
        
        sim = Simulation(cfg)
        
        cascade_records = []
        point_utilities, point_fake_sizes, point_true_sizes, point_verify_rates = Vector{Float64}(), Vector{Float64}(), Vector{Float64}(), Vector{Float64}()
        
        for c_id in 1:CASCADES_PER_SIM
            stats = run_single_cascade!(sim)
            
            # Store cascade metrics
            push!(cascade_records, (
                global_sim_id = idx,
                cascade_id = c_id,
                is_fake = stats.is_fake,
                seed_node = stats.seed_node,
                seed_degree = stats.seed_degree,
                max_depth = stats.max_depth,
                cascade_size = stats.size,
                shares = stats.shares,
                verifications = stats.verifications,
                receivers = stats.receivers,
                utility = stats.utility
            ))

            push!(point_utilities, stats.utility)
            vr = stats.receivers > 0 ? (stats.verifications / stats.receivers) : 0.0
            push!(point_verify_rates, vr)
            stats.is_fake ? push!(point_fake_sizes, stats.shares) : push!(point_true_sizes, stats.shares)
        end

        cascades_dfs[idx] = DataFrame(cascade_records)

        sim_dfs[idx] = DataFrame(
            global_sim_id = idx, lhs_id = lhs_id, rep_id = rep_id,
            cost = cost, rep_gain = gain, rep_loss = loss, verif_tpr = tpr, verif_fpr = fpr, 
            fake_prob = prev, num_conn = m,
            avg_welfare = mean(point_utilities),
            avg_fake_cascade = isempty(point_fake_sizes) ? 0.0 : mean(point_fake_sizes)/N_AGENTS,
            avg_true_cascade = isempty(point_true_sizes) ? 0.0 : mean(point_true_sizes)/N_AGENTS,
            avg_verify_rate = mean(point_verify_rates)
        )

        node_dfs[idx] = DataFrame(
            global_sim_id = fill(idx, N_AGENTS), node_id = 1:N_AGENTS,
            reputation = sim.net.reputation, base_alpha = sim.net.base_neighbor_alpha, base_beta  = sim.net.base_neighbor_beta,
            true_alpha = [b.alpha for b in sim.net.true_belief], true_beta  = [b.beta for b in sim.net.true_belief],
            tpr_alpha  = [b.alpha for b in sim.net.tech_tpr_belief], tpr_beta   = [b.beta for b in sim.net.tech_tpr_belief],
            fpr_alpha  = [b.alpha for b in sim.net.tech_fpr_belief], fpr_beta   = [b.beta for b in sim.net.tech_fpr_belief],
        )

        srcs, dsts, alphas, betas = Int[], Int[], Float64[], Float64[]
        for u in 1:N_AGENTS
            for (v, trust) in sim.net.neighbor_trust[u]
                push!(srcs, u), push!(dsts, v), push!(alphas, trust.alpha), push!(betas, trust.beta)
            end
        end
        edge_dfs[idx] = DataFrame(global_sim_id = fill(idx, length(srcs)), src = srcs, dst = dsts, trust_alpha = alphas, trust_beta = betas)
        
        lock(lk) do
            next!(prog)
        end
    end

    println("\nConcatenating DataFrames (this might take a few seconds)...")
    all_sims = vcat(sim_dfs...)
    all_cascades = vcat(cascades_dfs...)
    all_nodes = vcat(node_dfs...)
    all_edges = vcat(edge_dfs...)

    println("Saving to disk as Arrow files...")
    Arrow.write(joinpath(out_dir, "simulations.arrow"), all_sims)
    Arrow.write(joinpath(out_dir, "cascades.arrow"), all_cascades)
    Arrow.write(joinpath(out_dir, "nodes.arrow"), all_nodes)
    Arrow.write(joinpath(out_dir, "edges.arrow"), all_edges)

    println("\nSuccessfully saved into $(out_dir)/")
end

run_lhs_sweep()