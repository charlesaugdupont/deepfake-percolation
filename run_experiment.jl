# ================================================
# deepfake_percolation.jl
# ================================================
using Graphs
using DataStructures
using StatsBase
using Random
using ProgressMeter
using Plots

# ================================================
# CONFIGURATION
# ================================================
Base.@kwdef struct SimConfig
    random_seed::Int = 42

    # Network
    num_agents::Int = 50
    num_connections::Int = 1

    # Simulation
    global_fake_prob::Float64 = 0.5

    # Payoffs & Costs
    verification_cost::Float64 = 0.5
    rep_gain::Float64 = 1.0
    rep_loss::Float64 = 2.0

    # Verification Technology Confusion Matrix
    verification_tpr::Float64 = 0.90
    verification_fpr::Float64 = 0.10

    # Belief Priors
    neighbor_truth_prior_alpha::Float64 = 1.0
    neighbor_false_prior_beta::Float64 = 0.2
    global_truth_prior_alpha::Float64 = 3.0
    global_false_prior_beta::Float64 = 2.0
    verification_detected_fakes_prior::Float64 = 5.0
    verification_missed_fakes_prior::Float64 = 1.0
    verification_false_alarms_prior::Float64 = 1.0
    verification_correct_rejections_prior::Float64 = 5.0

    # Rewiring
    enable_rewiring::Bool = true
    rewiring_interval::Int = 10
end

# Mutable struct for Beta Distributions to avoid creating new objects constantly
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
        return numerator / denominator
    end

    function decide(p_true::Float64, reach::Int, est_tpr::Float64, est_fpr::Float64, cfg::SimConfig)
        k = Float64(reach)
        
        u_share_if_true = cfg.rep_gain * k
        u_share_if_fake = -cfg.rep_loss * k
        u_discard = 0.0
        
        u_share = (p_true * u_share_if_true) + ((1.0 - p_true) * u_share_if_fake)
        
        eu_given_real = (1.0 - est_fpr) * (cfg.rep_gain * k)
        eu_given_fake = (1.0 - est_tpr) * (-cfg.rep_loss * k)
        
        u_verify = -cfg.verification_cost + (p_true * eu_given_real) + ((1.0 - p_true) * eu_given_fake)

        if u_share > u_discard
            return u_share > u_verify ? :SHARE : :VERIFY
        else
            return u_discard >= u_verify ? :DISCARD : :VERIFY
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
    
    # State vectors indexed by node ID (1 to num_agents)
    reputation::Vector{Float64}
    true_belief::Vector{BetaDist}
    tech_tpr_belief::Vector{BetaDist}
    tech_fpr_belief::Vector{BetaDist}
    neighbor_trust::Vector{Dict{Int, BetaDist}}
end

function SocialNetwork(cfg::SimConfig, rng::AbstractRNG)
    G = barabasi_albert(cfg.num_agents, cfg.num_connections; rng=rng)    
    reputation = zeros(Float64, cfg.num_agents)
    true_belief = [BetaDist(cfg.global_truth_prior_alpha, cfg.global_false_prior_beta) for _ in 1:cfg.num_agents]
    tech_tpr_belief = [BetaDist(cfg.verification_detected_fakes_prior, cfg.verification_missed_fakes_prior) for _ in 1:cfg.num_agents]
    tech_fpr_belief = [BetaDist(cfg.verification_false_alarms_prior, cfg.verification_correct_rejections_prior) for _ in 1:cfg.num_agents]
    neighbor_trust = [Dict{Int, BetaDist}() for _ in 1:cfg.num_agents]
    
    net = SocialNetwork(cfg, rng, G, reputation, true_belief, tech_tpr_belief, tech_fpr_belief, neighbor_trust)
    
    for e in edges(G)
        u, v = src(e), dst(e)
        _init_trust_one_way!(net, u, v)
        _init_trust_one_way!(net, v, u)
    end
    
    return net
end

function _init_trust_one_way!(net::SocialNetwork, observer::Int, target::Int)
    net.neighbor_trust[observer][target] = BetaDist(net.cfg.neighbor_truth_prior_alpha, net.cfg.neighbor_false_prior_beta)
end

function update_trust!(net::SocialNetwork, observer::Int, sender::Int, sender_was_truthful::Bool)
    if haskey(net.neighbor_trust[observer], sender)
        stats = net.neighbor_trust[observer][sender]
        if sender_was_truthful
            stats.alpha += 1.0
        else
            stats.beta += 1.0
        end
    end
end

function update_global_belief!(net::SocialNetwork, agent_id::Int, item_was_fake::Bool)
    stats = net.true_belief[agent_id]
    if item_was_fake
        stats.beta += 1.0
    else
        stats.alpha += 1.0
    end
end

function rewire_network!(net::SocialNetwork)
    edges_to_check = collect(edges(net.G))
    if isempty(edges_to_check) return end

    min_rep = minimum(net.reputation)
    max_rep = maximum(net.reputation)
    
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

        if rel_u_v < rel_v_u
            weakest_trust_val = rel_u_v
            rewiring_node, passive_node = u, v
        else
            weakest_trust_val = rel_v_u
            rewiring_node, passive_node = v, u
        end

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
    push!(invalid_set, u)
    push!(invalid_set, node_to_avoid)

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
    
    # Int tuple: (Current Node, Sender Node). 0 means None.
    queue = Queue{Tuple{Int, Int}}()
    enqueue!(queue, (seed_node, 0))
    visited = Set{Int}([seed_node])
    parent_map = Dict{Int, Int}()
    
    actions_log = Dict{Int, Tuple{Symbol, Int}}()
    verification_log = Vector{Tuple{Int, Bool}}()
    
    n_shares = 0
    n_verifications = 0
    n_receivers = 0
    total_payoff = 0.0

    actual_tpr = sim.cfg.verification_tpr
    actual_fpr = sim.cfg.verification_fpr

    while !isempty(queue)
        current, sender = dequeue!(queue)
        n_receivers += 1
        
        if sender == 0
            reliability = 0.5
        else
            if haskey(sim.net.neighbor_trust[current], sender)
                reliability = AgentLogic.compute_beta_mean(sim.net.neighbor_trust[current][sender])
            else
                reliability = AgentLogic.compute_beta_mean(sim.cfg.neighbor_truth_prior_alpha, sim.cfg.neighbor_false_prior_beta)
            end
        end

        global_true_est = AgentLogic.compute_beta_mean(sim.net.true_belief[current])
        p_true = AgentLogic.calculate_posterior_truth_prob(reliability, global_true_est)

        est_tpr = AgentLogic.compute_beta_mean(sim.net.tech_tpr_belief[current])
        est_fpr = AgentLogic.compute_beta_mean(sim.net.tech_fpr_belief[current])

        reach = degree(sim.net.G, current)
        action = AgentLogic.decide(p_true, reach, est_tpr, est_fpr, sim.cfg)
        
        if action == :DISCARD
            # Do nothing
        elseif action == :SHARE
            n_shares += 1
            outcome_type = item_is_fake ? :shared_fake : :shared_true
            actions_log[current] = (outcome_type, reach)
            
            total_payoff += item_is_fake ? -(sim.cfg.rep_loss * reach) : (sim.cfg.rep_gain * reach)

            for nbr in neighbors(sim.net.G, current)
                if !(nbr in visited)
                    push!(visited, nbr)
                    enqueue!(queue, (nbr, current))
                    parent_map[nbr] = current
                end
            end
        elseif action == :VERIFY
            n_verifications += 1
            total_payoff -= sim.cfg.verification_cost

            tool_says_fake = false
            if item_is_fake
                tool_says_fake = rand(sim.rng) < actual_tpr
            else
                tool_says_fake = rand(sim.rng) < actual_fpr
            end

            push!(verification_log, (current, tool_says_fake))

            if !tool_says_fake
                n_shares += 1
                outcome_type = item_is_fake ? :shared_fake : :shared_true
                actions_log[current] = (outcome_type, reach)
                
                total_payoff += item_is_fake ? -(sim.cfg.rep_loss * reach) : (sim.cfg.rep_gain * reach)
                
                for nbr in neighbors(sim.net.G, current)
                    if !(nbr in visited)
                        push!(visited, nbr)
                        enqueue!(queue, (nbr, current))
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

    return (
        is_fake = item_is_fake,
        size = length(visited),
        shares = n_shares,
        verifications = n_verifications,
        receivers = n_receivers,
        utility = total_payoff
    )
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
        tpr_stats = sim.net.tech_tpr_belief[agent_id]
        fpr_stats = sim.net.tech_fpr_belief[agent_id]
        if item_is_fake
            tool_said_fake ? (tpr_stats.alpha += 1.0) : (tpr_stats.beta += 1.0)
        else
            tool_said_fake ? (fpr_stats.alpha += 1.0) : (fpr_stats.beta += 1.0)
        end
    end
end

# ================================================
# EXPERIMENT & PLOTTING
# ================================================
function run_heatmap_experiment()
    RESOLUTION = 15
    NUM_TRIALS = 15
    CASCADES_PER_TRIAL = 100
    MASTER_SEED = 12345
    master_rng = Xoshiro(MASTER_SEED)
    
    costs = range(0.1, 5.0, length=RESOLUTION)
    prevalences = range(0.05, 0.80, length=RESOLUTION)
    
    grid_fake_size = zeros(RESOLUTION, RESOLUTION)
    grid_true_size = zeros(RESOLUTION, RESOLUTION)
    grid_verify_rate = zeros(RESOLUTION, RESOLUTION)
    grid_welfare = zeros(RESOLUTION, RESOLUTION)

    total_sims = RESOLUTION * RESOLUTION * NUM_TRIALS * CASCADES_PER_TRIAL
    println("Starting sweep. Total cascades: ", total_sims)
    
    # Progress bar using ProgressMeter
    prog = Progress(RESOLUTION * RESOLUTION, 1, "Sweeping: ")

    for (i, prev) in enumerate(prevalences)
        for (j, cost) in enumerate(costs)
            
            cell_seed = rand(master_rng, 1:1_000_000_000)
            cell_rng = Xoshiro(cell_seed)
            
            point_fake_sizes = Float64[]
            point_true_sizes = Float64[]
            point_verify_rates = Float64[]
            point_utilities = Float64[]
            
            for _ in 1:NUM_TRIALS
                cfg = SimConfig(
                    num_agents = 300,
                    num_connections = 5,
                    verification_cost = cost,
                    global_fake_prob = prev,
                    rep_gain = 1.0,
                    rep_loss = 2.0,
                    enable_rewiring = true,
                    rewiring_interval = 10,
                    random_seed = rand(cell_rng, 1:1_000_000_000)
                )
                
                sim = Simulation(cfg)
                
                for _ in 1:CASCADES_PER_TRIAL
                    stats = run_single_cascade!(sim)
                    
                    push!(point_utilities, stats.utility)
                    
                    vr = stats.receivers > 0 ? (stats.verifications / stats.receivers) : 0.0
                    push!(point_verify_rates, vr)
                    
                    if stats.is_fake
                        push!(point_fake_sizes, stats.shares)
                    else
                        push!(point_true_sizes, stats.shares)
                    end
                end
            end
            
            grid_fake_size[i, j] = isempty(point_fake_sizes) ? 0.0 : mean(point_fake_sizes) / 300.0
            grid_true_size[i, j] = isempty(point_true_sizes) ? 0.0 : mean(point_true_sizes) / 300.0
            grid_verify_rate[i, j] = isempty(point_verify_rates) ? 0.0 : mean(point_verify_rates)
            grid_welfare[i, j] = mean(point_utilities)

            next!(prog)
        end
    end

    plot_results(costs, prevalences, grid_fake_size, grid_true_size, grid_verify_rate, grid_welfare)
end

function make_twoslope_cmap(vmin, vmax; cmap=:RdBu, n_steps=256)
    orig_cg = cgrad(cmap)
    # Get the color type dynamically to initialize our array
    colors = typeof(get(orig_cg, 0.0))[] 
    
    for i in 1:n_steps
        f = (i - 1) / (n_steps - 1)
        val = vmin + f * (vmax - vmin)
        
        if val < 0
            # Map [vmin, 0] linearly to the bottom half of the colormap [0.0, 0.5]
            c_f = vmin < 0 ? 0.5 * (1.0 - val / vmin) : 0.5
        else
            # Map [0, vmax] linearly to the top half of the colormap [0.5, 1.0]
            c_f = vmax > 0 ? 0.5 + 0.5 * (val / vmax) : 0.5
        end
        push!(colors, get(orig_cg, c_f))
    end
    return cgrad(colors)
end

function plot_results(costs, prevalences, fake, true_size, verify, welfare)
    p1 = heatmap(costs, prevalences, fake,
        title="Avg Fake Cascade Size", xlabel="Verification Cost", ylabel="Fake Prevalence",
        color=:Reds)

    p2 = heatmap(costs, prevalences, true_size,
        title="Avg True Cascade Size", xlabel="Verification Cost", ylabel="Fake Prevalence",
        color=:Greens)

    p3 = heatmap(costs, prevalences, verify,
        title="Verification Rate", xlabel="Verification Cost", ylabel="Fake Prevalence",
        color=:Blues)

    vmin = minimum(welfare)
    vmax = maximum(welfare)
    
    # Create our custom piecewise linear colormap
    custom_cmap = make_twoslope_cmap(vmin, vmax, cmap=:RdBu)

    p4 = heatmap(costs, prevalences, welfare,
        title="Avg Social Welfare", xlabel="Verification Cost", ylabel="Fake Prevalence",
        color=custom_cmap, clims=(vmin, vmax))

    fig = plot(p1, p2, p3, p4, layout=(2, 2), size=(1000, 800), margin=5Plots.mm)
    
    savefig(fig, "results_julia.png")
    display(fig)
end

# Run the experiment
run_heatmap_experiment()