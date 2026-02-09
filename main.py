import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import SimConfig
from model import DeepfakePercolationModel

def run_resilience_experiment():
    print("Running Sequential Mixed-Stream Experiment...")
    
    # 1. Sweep Parameters
    costs = np.linspace(0, 10, 15)
    prevalences = np.linspace(0.1, 0.9, 15)
    
    # Storage for 3 Heatmaps
    map_fake_size = np.zeros((len(prevalences), len(costs)))
    map_true_size = np.zeros((len(prevalences), len(costs)))
    map_utility   = np.zeros((len(prevalences), len(costs)))
    
    # Simulation Parameters
    NUM_STEPS = 50 # Number of cascades per run (Time evolution)
    
    for i, p_val in enumerate(tqdm(prevalences, desc="Prevalence")):
        for j, c_val in enumerate(costs):
            
            # Setup Model
            cfg = SimConfig(
                num_agents=200, 
                C=c_val, 
                global_prevalence=p_val, # This now controls the INJECTION ratio
                network_type='watts_strogatz'
            )
            model = DeepfakePercolationModel(cfg)
            
            # Trackers for this specific run
            fake_sizes = []
            true_sizes = []
            
            # --- SEQUENTIAL LOOP ---
            for _ in range(NUM_STEPS):
                # 1. Decide: Is this injection True or Fake?
                # We use the environment's prevalence to decide what hits the network
                is_fake = np.random.random() < p_val
                
                seed = np.random.randint(0, cfg.num_agents)
                stats = model.run_cascade(seed, is_ground_truth_fake=is_fake)
                
                norm_size = stats['total_shared'] / cfg.num_agents
                
                if is_fake:
                    fake_sizes.append(norm_size)
                else:
                    true_sizes.append(norm_size)
            
            # --- AGGREGATE RESULTS ---
            # We take the mean of the sizes. 
            # Note: If no true news was injected (rare), handle div/0 or empty
            map_fake_size[i, j] = np.mean(fake_sizes) if fake_sizes else 0
            map_true_size[i, j] = np.mean(true_sizes) if true_sizes else 0
            
            # Calculate Average Utility per Agent
            total_util = sum(d['utility'] for n, d in model.G.nodes(data=True))
            avg_util = total_util / (cfg.num_agents * NUM_STEPS)
            map_utility[i, j] = avg_util

    # --- PLOTTING (3 Plots) ---
    _, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    extent = [0, 10, 0.1, 0.9]
    
    # Plot 1: Fake News Spread
    im1 = axes[0].imshow(map_fake_size, origin='lower', aspect='auto', extent=extent, cmap='Reds')
    axes[0].set_title('Avg Fake Cascade Size')
    axes[0].set_xlabel('Cost (C)')
    axes[0].set_ylabel('Prevalence (P_global)')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot 2: True News Spread
    im2 = axes[1].imshow(map_true_size, origin='lower', aspect='auto', extent=extent, cmap='Greens')
    axes[1].set_title('Avg True Cascade Size')
    axes[1].set_xlabel('Cost (C)')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot 3: Average Utility
    # Use a diverging colormap (Red = Negative, White = Zero, Blue = Positive)
    im3 = axes[2].imshow(map_utility, origin='lower', aspect='auto', extent=extent, cmap='RdBu', vmin=-5, vmax=5)
    axes[2].set_title('Avg Agent Utility')
    axes[2].set_xlabel('Cost (C)')
    plt.colorbar(im3, ax=axes[2], label='Utility per Step')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_resilience_experiment()