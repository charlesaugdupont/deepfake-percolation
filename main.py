import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import SimConfig
from model import DeepfakePercolationModel

def run_resilience_experiment():
    
    # 1. Sweep Parameters
    costs = np.linspace(0, 10, 15)
    prevalences = np.linspace(0.1, 0.9, 15)
    
    # Storage for 4 Heatmaps
    map_fake_size = np.zeros((len(prevalences), len(costs)))
    map_true_size = np.zeros((len(prevalences), len(costs)))
    map_utility   = np.zeros((len(prevalences), len(costs)))
    map_verification_rate = np.zeros((len(prevalences), len(costs)))
    
    # Simulation Parameters
    NUM_STEPS = 50 
    
    for i, p_val in enumerate(tqdm(prevalences, desc="Prevalence")):
        for j, c_val in enumerate(costs):
            
            # Setup Model
            cfg = SimConfig(
                num_agents=200, 
                global_prevalence=p_val,
                eta_w_update_rate=0.05,
                network_type='watts_strogatz'
            )
            model = DeepfakePercolationModel(cfg)
            
            # Trackers for this specific run
            fake_sizes = []
            true_sizes = []
            verification_ratios = []
            
            # --- SEQUENTIAL LOOP ---
            for _ in range(NUM_STEPS):
                is_fake = np.random.random() < p_val
                seed = np.random.randint(0, cfg.num_agents)
                stats = model.run_cascade(seed, is_ground_truth_fake=is_fake)
                
                norm_size = stats['total_shared'] / cfg.num_agents
                
                if is_fake:
                    fake_sizes.append(norm_size)
                else:
                    true_sizes.append(norm_size)
                
                # --- NEW: Track Verification Rate ---
                if stats['total_reached'] > 1: # Avoid counting only the source
                    ratio = stats['total_verified'] / (stats['total_reached'] - 1)
                    verification_ratios.append(ratio)
                else:
                    verification_ratios.append(0)

            # --- AGGREGATE RESULTS ---
            map_fake_size[i, j] = np.mean(fake_sizes) if fake_sizes else 0
            map_true_size[i, j] = np.mean(true_sizes) if true_sizes else 0
            map_verification_rate[i, j] = np.mean(verification_ratios) # NEW
            
            total_util = sum(d.get('utility', 0) for n, d in model.G.nodes(data=True))
            avg_util = total_util / (cfg.num_agents * NUM_STEPS)
            map_utility[i, j] = avg_util

    # --- PLOTTING (2x2 Grid) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Phase Diagram of a Social Network with Dynamic Trust', fontsize=16)
    
    extent = [0, 10, 0.1, 0.9]
    
    # Plot 1: Fake News Spread (Top-Left)
    im1 = axes[0, 0].imshow(map_fake_size, origin='lower', aspect='auto', extent=extent, cmap='Reds')
    axes[0, 0].set_title('Avg Fake Cascade Size')
    axes[0, 0].set_ylabel('Prevalence (P_global)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: True News Spread (Top-Right)
    im2 = axes[0, 1].imshow(map_true_size, origin='lower', aspect='auto', extent=extent, cmap='Greens')
    axes[0, 1].set_title('Avg True Cascade Size')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Average Utility (Bottom-Left)
    im3 = axes[1, 0].imshow(map_utility, origin='lower', aspect='auto', extent=extent, cmap='RdBu', vmin=-5, vmax=5)
    axes[1, 0].set_title('Avg Agent Utility per Step')
    axes[1, 0].set_xlabel('Cost (C)')
    axes[1, 0].set_ylabel('Prevalence (P_global)')
    plt.colorbar(im3, ax=axes[1, 0])

    # Plot 4: Verification Rate (Bottom-Right) - NEW
    im4 = axes[1, 1].imshow(map_verification_rate, origin='lower', aspect='auto', extent=extent, cmap='viridis', vmin=0, vmax=1)
    axes[1, 1].set_title('Verification Rate')
    axes[1, 1].set_xlabel('Cost (C)')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.show()

if __name__ == "__main__":
    run_resilience_experiment()