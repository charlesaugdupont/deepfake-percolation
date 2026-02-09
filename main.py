import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import SimConfig
from model import DeepfakePercolationModel

def run_resilience_experiment():
    costs = np.linspace(0, 10, 30)
    prevalences = np.linspace(0.1, 0.9, 30)

    map_fake = np.zeros((len(prevalences), len(costs)))
    map_true = np.zeros((len(prevalences), len(costs)))
    map_util = np.zeros((len(prevalences), len(costs)))
    map_verify = np.zeros((len(prevalences), len(costs)))

    NUM_STEPS = 100

    for i, p in enumerate(tqdm(prevalences, desc="Prevalence")):
        for j, c in enumerate(costs):
            cfg = SimConfig(num_agents=200, C=c, global_prevalence=p)
            model = DeepfakePercolationModel(cfg)

            fake_sizes, true_sizes = [], []
            utilities = []
            verify_rates = []

            for _ in range(NUM_STEPS):
                is_fake = np.random.random() < p
                seed = np.random.randint(cfg.num_agents)
                stats = model.run_cascade(seed, is_fake)

                size = stats['total_shared'] / cfg.num_agents
                utilities.append(stats['utility'] / cfg.num_agents)

                if stats['num_received'] > 0:
                    verify_rates.append(
                        stats['num_verified'] / stats['num_received']
                    )

                (fake_sizes if is_fake else true_sizes).append(size)

            map_fake[i, j] = np.mean(fake_sizes) if fake_sizes else 0.0
            map_true[i, j] = np.mean(true_sizes) if true_sizes else 0.0
            map_util[i, j] = np.mean(utilities)
            map_verify[i, j] = np.mean(verify_rates) if verify_rates else 0.0

    # ---- PLOTTING ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    extent = [0, 10, 0.1, 0.9]

    im1 = axes[0, 0].imshow(
        map_fake, origin='lower', extent=extent, aspect='auto', cmap='Reds'
    )
    axes[0, 0].set_title('Avg Fake Cascade Size')
    axes[0, 0].set_xlabel('Verification Cost (C)')
    axes[0, 0].set_ylabel('Global Fake Prevalence')
    plt.colorbar(im1, ax=axes[0, 0], label='Fraction of Network')

    im2 = axes[0, 1].imshow(
        map_true, origin='lower', extent=extent, aspect='auto', cmap='Greens'
    )
    axes[0, 1].set_title('Avg True Cascade Size')
    axes[0, 1].set_xlabel('Verification Cost (C)')
    plt.colorbar(im2, ax=axes[0, 1], label='Fraction of Network')

    im3 = axes[1, 0].imshow(
        map_util, origin='lower', extent=extent, aspect='auto',
        cmap='RdBu', vmin=-5, vmax=5
    )
    axes[1, 0].set_title('Avg Agent Utility')
    axes[1, 0].set_xlabel('Verification Cost (C)')
    axes[1, 0].set_ylabel('Global Fake Prevalence')
    plt.colorbar(im3, ax=axes[1, 0], label='Utility per Agent')

    im4 = axes[1, 1].imshow(
        map_verify, origin='lower', extent=extent, aspect='auto', cmap='Blues'
    )
    axes[1, 1].set_title('Verification Rate')
    axes[1, 1].set_xlabel('Verification Cost (C)')
    plt.colorbar(im4, ax=axes[1, 1], label='P(VERIFY | Received)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_resilience_experiment()