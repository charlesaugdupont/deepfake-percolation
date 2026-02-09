import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import SimConfig
from model import DeepfakePercolationModel

def run_experiment():
    print("Initializing Deepfake Percolation Experiment...")
    
    # 1. Define Sweep Range for Cost (C)
    # We expect a transition: Low Cost = High Verification = Low Spread
    # High Cost = Low Verification = High Spread
    costs = np.linspace(0.0, 10.0, 20)
    
    avg_cascade_sizes = []
    verification_ratios = []
    
    # Run simulation for each Cost value
    for cost in tqdm(costs, desc="Simulating Cost Regimes"):
        
        # Create Config
        cfg = SimConfig(
            num_agents=500,
            C=cost,
            global_prevalence=0.5, # High danger environment
            network_type='watts_strogatz',
            ws_k=6,
            ws_p=0.1
        )
        
        # Rebuild model fresh for each parameter point
        # (Or you could keep the same network and just reset agents)
        model = DeepfakePercolationModel(cfg)
        
        # Run M cascades (Monte Carlo)
        num_cascades = 50
        sizes = []
        verifications = []
        
        for _ in range(num_cascades):
            # Pick a random seed
            seed = np.random.randint(0, cfg.num_agents)
            
            # Inject a FAKE news item
            # We want to see how far a FAKE item travels
            stats = model.run_cascade(source_node=seed, is_ground_truth_fake=True)
            
            # Record normalized cascade size
            sizes.append(stats['total_shared'] / cfg.num_agents)
            
            # Record verification ratio (verifications / reach)
            if stats['total_reached'] > 0:
                verifications.append(stats['total_verified'] / stats['total_reached'])
            else:
                verifications.append(0)
                
        avg_cascade_sizes.append(np.mean(sizes))
        verification_ratios.append(np.mean(verifications))
        
    # --- Plotting Results ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Verification Cost (C)')
    ax1.set_ylabel('Avg. Fake Cascade Size (% of Network)', color=color)
    ax1.plot(costs, avg_cascade_sizes, color=color, marker='o', label='Cascade Size')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Verification Ratio', color=color)  # we already handled the x-label with ax1
    ax2.plot(costs, verification_ratios, color=color, linestyle='--', label='Verification Rate')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Phase Transition: Effect of Verification Cost on Deepfake Spread')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_experiment()