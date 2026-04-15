# ================================================
# run_burnin_diagnostic.jl
#
# Runs burn-in diagnostics at a single, hand-picked
# parameter point with many stochastic replications.
# Edit DIAG_CFG to probe different regions of
# parameter space.
# ================================================

# ── guard so run_experiment.jl doesn't fire run_lhs_sweep() when included ──
const _INCLUDED_AS_MODULE = true
include("run_experiment.jl")

using Arrow, DataFrames, Statistics, Dates, ProgressMeter, Random

# ════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC CONFIG  — edit these to probe different parameter points
# ════════════════════════════════════════════════════════════════════════════
const DIAG = (
    # ── parameter point to probe ─────────────────────────────────────────────
    v_cost   = 0.01, # [0.01, 1.00]
    rep_loss = 3.00, # [1.00, 3.00]
    tpr      = 0.85, # [0.50, 0.99]
    fpr      = 0.10, # [0.01, 0.50]
    p_fake   = 0.2, # [0.10, 0.90]

    # ── network / payoff (must match main sweep) ──────────────────────────────
    n_agents          = 300,
    num_connections   = 3,
    p_triadic_closure = 0.5,
    rep_gain          = 1.0,

    # ── run length ───────────────────────────────────────────────────────────
    burn_in_length = 2000,   # must match the main sweep value
    post_burnin    = 200,    # extra cascades recorded after burn-in ends

    # ── replications & snapshot cadence ──────────────────────────────────────
    n_reps         = 50,
    snapshot_every = 1,
    rng_seed       = 42,
)

# ════════════════════════════════════════════════════════════════════════════
# SNAPSHOT HELPERS
# ════════════════════════════════════════════════════════════════════════════

function agent_snapshot(sim::Simulation, centrality, rep_id::Int, cascade_id::Int, phase::String)
    net = sim.net
    G   = net.G
    n   = sim.cfg.num_agents

    # — existing belief/trust computations —
    true_mean  = [AgentLogic.compute_beta_mean(b) for b in net.true_belief]
    tpr_mean   = [AgentLogic.compute_beta_mean(b) for b in net.tech_tpr_belief]
    fpr_mean   = [AgentLogic.compute_beta_mean(b) for b in net.tech_fpr_belief]
    true_n     = [b.alpha + b.beta for b in net.true_belief]
    tpr_n      = [b.alpha + b.beta for b in net.tech_tpr_belief]
    fpr_n      = [b.alpha + b.beta for b in net.tech_fpr_belief]

    degree_vec = degree(G)

    mean_trust = map(1:n) do i
        ts = net.neighbor_trust[i]
        isempty(ts) && return NaN
        mean(AgentLogic.compute_beta_mean(v) for v in values(ts))
    end
    mean_trust_n = map(1:n) do i
        ts = net.neighbor_trust[i]
        isempty(ts) && return NaN
        mean(v.alpha + v.beta for v in values(ts))
    end

    DataFrame(
        rep_id = fill(rep_id,     n),
        cascade_id = fill(cascade_id, n),
        phase = fill(phase,      n),
        node_id = 1:n,
        true_mean = true_mean,
        tpr_mean = tpr_mean,
        fpr_mean = fpr_mean,
        true_n = true_n,
        tpr_n = tpr_n,
        fpr_n = fpr_n,
        mean_trust = mean_trust,
        mean_trust_n = mean_trust_n,
        node_degree = degree_vec,
        betweenness = centrality.betweenness,
        closeness = centrality.closeness,
        eigenvector = centrality.eigenvector,
        verify_count = copy(sim.net.verify_count),
        exposure_count = copy(sim.net.exposure_count)
    )
end

function cascade_summary(stats, rep_id::Int, cascade_id::Int, phase::String)
    vr = stats.size > 0 ? stats.verifications / stats.size : 0.0
    (
        rep_id        = rep_id,
        cascade_id    = cascade_id,
        phase         = phase,
        is_fake       = stats.is_fake,
        cascade_size  = stats.size,
        shares        = stats.shares,
        verifications = stats.verifications,
        verify_rate   = vr,
        utility       = stats.utility,
        max_depth     = stats.max_depth,
        seed_degree   = stats.seed_degree,
    )
end

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

function run_burnin_diagnostic()
    out_dir   = joinpath("data", "burnin_diagnostic")
    mkpath(out_dir)

    # Save config so Python can read ground-truth values
    open(joinpath(out_dir, "diagnostic_config.txt"), "w") do io
        for k in fieldnames(typeof(DIAG))
            println(io, "$k = $(getfield(DIAG, k))")
        end
    end

    total_cascades = DIAG.burn_in_length + DIAG.post_burnin
    all_snapshots  = Vector{DataFrame}(undef, DIAG.n_reps)
    all_cascades   = Vector{DataFrame}(undef, DIAG.n_reps)

    println("Saving diagnostic to: $out_dir/")
    println("Parameter point: v_cost=$(DIAG.v_cost)  loss=$(DIAG.rep_loss)  " *
            "tpr=$(DIAG.tpr)  fpr=$(DIAG.fpr)  p_fake=$(DIAG.p_fake)")
    println("Running $(DIAG.n_reps) reps × $total_cascades cascades " *
            "using $(Threads.nthreads()) threads...")

    prog = Progress(DIAG.n_reps, 1, "Running burn-in diagnostics... ")
    lk   = ReentrantLock()

    Threads.@threads for rep in 1:DIAG.n_reps
        rep_seed = Int(mod(hash((rep, DIAG.rng_seed)), typemax(Int)))

        cfg = SimConfig(
            num_agents          = DIAG.n_agents,
            num_connections     = DIAG.num_connections,
            p_triadic_closure   = DIAG.p_triadic_closure,
            global_fake_prob    = DIAG.p_fake,
            verification_cost   = DIAG.v_cost,
            rep_gain            = DIAG.rep_gain,
            rep_loss            = DIAG.rep_loss,
            verification_tpr    = DIAG.tpr,
            verification_fpr    = DIAG.fpr,
            random_seed         = rep_seed,
        )

        sim = Simulation(cfg)

        # compute centrality measures
        centrality = (
            betweenness = betweenness_centrality(sim.net.G),
            closeness = closeness_centrality(sim.net.G),
            eigenvector = eigenvector_centrality(sim.net.G),
        )

        snap_dfs = DataFrame[]
        cascade_rows = NamedTuple[]

        for c_id in 1:total_cascades
            phase        = c_id <= DIAG.burn_in_length ? "burnin" : "post"
            after_burnin = phase == "post"

            stats = run_single_cascade!(sim, after_burnin)

            push!(cascade_rows, cascade_summary(stats, rep, c_id, phase))

            is_boundary = (c_id == 1) || (c_id == DIAG.burn_in_length) || (c_id == total_cascades)
            if is_boundary || c_id % DIAG.snapshot_every == 0
                push!(snap_dfs, agent_snapshot(sim, centrality, rep, c_id, phase))
            end
        end

        all_snapshots[rep] = vcat(snap_dfs...)
        all_cascades[rep]  = DataFrame(cascade_rows)

        lock(lk) do; next!(prog) end
    end

    println("\nConcatenating and saving...")
    Arrow.write(joinpath(out_dir, "burnin_snapshots.arrow"), vcat(all_snapshots...))
    Arrow.write(joinpath(out_dir, "burnin_cascades.arrow"),  vcat(all_cascades...))
    println("Done → $out_dir/")
end

run_burnin_diagnostic()
