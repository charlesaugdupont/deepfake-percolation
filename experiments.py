import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from config import SimConfig
from simulation import Simulation

def run_heatmap_experiment():
    RESOLUTION = 20           
    NUM_TRIALS = 15          
    CASCADES_PER_TRIAL = 50   
    
    costs = np.linspace(0.5, 6.0, RESOLUTION) 
    prevalences = np.linspace(0.1, 0.9, RESOLUTION) 
    
    grid_fake_size = np.zeros((RESOLUTION, RESOLUTION))
    grid_true_size = np.zeros((RESOLUTION, RESOLUTION))
    grid_verify_rate = np.zeros((RESOLUTION, RESOLUTION))
    grid_welfare = np.zeros((RESOLUTION, RESOLUTION))

    total_sims = RESOLUTION * RESOLUTION * NUM_TRIALS * CASCADES_PER_TRIAL
    print(f"Starting sweep. Total cascades: {total_sims}")
    
    for i, prev in enumerate(tqdm(prevalences, desc="Sweeping")):
        for j, cost in enumerate(costs):
            
            point_fake_sizes = []
            point_true_sizes = []
            point_verify_rates = []
            point_utilities = []
            
            for _ in range(NUM_TRIALS):
                cfg = SimConfig(
                    num_agents=250,
                    num_connections=2,
                    verification_cost=cost,
                    global_fake_prob=prev,
                    rep_gain=1.0,
                    rep_loss=2.0,
                    rep_debunk=0.5, # Note: Maybe increase this to 1.0 later?
                    self_interest_weight=0.7
                )
                
                sim = Simulation(cfg)
                
                for cascade_idx in range(CASCADES_PER_TRIAL):
                    stats = sim.run_single_cascade(cascade_idx)
                    
                    point_utilities.append(stats['utility'])
                    
                    if stats['receivers'] > 0:
                        point_verify_rates.append(stats['verifications'] / stats['receivers'])
                    else:
                        point_verify_rates.append(0.0)
                    
                    if stats['is_fake']:
                        point_fake_sizes.append(stats['shares']) 
                    else:
                        point_true_sizes.append(stats['shares'])
            
            grid_fake_size[i, j] = np.mean(point_fake_sizes) / cfg.num_agents if point_fake_sizes else 0
            grid_true_size[i, j] = np.mean(point_true_sizes) / cfg.num_agents if point_true_sizes else 0
            grid_verify_rate[i, j] = np.mean(point_verify_rates) if point_verify_rates else 0
            grid_welfare[i, j] = np.mean(point_utilities)

    plot_results(costs, prevalences, grid_fake_size, grid_true_size, grid_verify_rate, grid_welfare)

def plot_results(costs, prevalences, fake, true, verify, welfare):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    extent = [costs.min(), costs.max(), prevalences.min(), prevalences.max()]
    origin = 'lower'

    # 1. Fake Cascade Size
    ax1 = axes[0, 0]
    im1 = ax1.imshow(fake, extent=extent, origin=origin, aspect='auto', cmap='Reds')
    ax1.set_title(r'Avg Fake Cascade Size (% of Net)')
    ax1.set_xlabel(r'Verification Cost ($C_v$)')
    ax1.set_ylabel(r'Fake Prevalence ($\pi_f$)')
    fig.colorbar(im1, ax=ax1)

    # 2. True Cascade Size
    ax2 = axes[0, 1]
    im2 = ax2.imshow(true, extent=extent, origin=origin, aspect='auto', cmap='Greens')
    ax2.set_title(r'Avg True Cascade Size (% of Net)')
    ax2.set_xlabel(r'Verification Cost ($C_v$)')
    ax2.set_ylabel(r'Fake Prevalence ($\pi_f$)')
    fig.colorbar(im2, ax=ax2)

    # 3. Verification Rate
    ax3 = axes[1, 0]
    im3 = ax3.imshow(verify, extent=extent, origin=origin, aspect='auto', cmap='Blues') 
    ax3.set_title(r'Verification Rate (Verifies / Receptions)')
    ax3.set_xlabel(r'Verification Cost ($C_v$)')
    ax3.set_ylabel(r'Fake Prevalence ($\pi_f$)')
    fig.colorbar(im3, ax=ax3)

    # 4. Social Welfare (FIXED COLOR SCALE)
    ax4 = axes[1, 1]
    
    # Calculate min and max
    vmin, vmax = welfare.min(), welfare.max()
    
    # Create the Norm: Center=0, Vmin=min_data, Vmax=max_data
    # Use TwoSlopeNorm (available in matplotlib 3.2+)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
    
    im4 = ax4.imshow(welfare, extent=extent, origin=origin, aspect='auto', 
                     cmap='RdBu', norm=norm)
                     
    ax4.set_title(r'Avg Social Welfare (Utility)')
    ax4.set_xlabel(r'Verification Cost ($C_v$)')
    ax4.set_ylabel(r'Fake Prevalence ($\pi_f$)')
    fig.colorbar(im4, ax=ax4)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_heatmap_experiment()