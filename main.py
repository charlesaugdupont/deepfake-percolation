import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import SimConfig
from model import DeepfakePercolationModel

def run_resilience_experiment():
    costs = np.linspace(0, 10, 15)
    prevalences = np.linspace(0.1, 0.9, 15)

    map_fake = np.zeros((len(prevalences), len(costs)))
    map_true = np.zeros((len(prevalences), len(costs)))
    map_util = np.zeros((len(prevalences), len(costs)))

    NUM_STEPS = 50

    for i, p in enumerate(tqdm(prevalences)):
        for j, c in enumerate(costs):
            cfg = SimConfig(num_agents=200, C=c, global_prevalence=p)
            model = DeepfakePercolationModel(cfg)

            fake_sizes, true_sizes, utilities = [], [], []

            for _ in range(NUM_STEPS):
                is_fake = np.random.random() < p
                seed = np.random.randint(cfg.num_agents)
                stats = model.run_cascade(seed, is_fake)

                size = stats['total_shared'] / cfg.num_agents
                utilities.append(stats['utility'] / cfg.num_agents)

                (fake_sizes if is_fake else true_sizes).append(size)

            map_fake[i, j] = np.mean(fake_sizes) if fake_sizes else 0
            map_true[i, j] = np.mean(true_sizes) if true_sizes else 0
            map_util[i, j] = np.mean(utilities)

    _, ax = plt.subplots(1, 3, figsize=(18, 5))
    extent = [0, 10, 0.1, 0.9]

    ax[0].imshow(map_fake, origin='lower', extent=extent, aspect='auto', cmap='Reds')
    ax[0].set_title('Avg Fake Cascade Size')

    ax[1].imshow(map_true, origin='lower', extent=extent, aspect='auto', cmap='Greens')
    ax[1].set_title('Avg True Cascade Size')

    ax[2].imshow(map_util, origin='lower', extent=extent, aspect='auto',
                 cmap='RdBu', vmin=-5, vmax=5)
    ax[2].set_title('Avg Utility')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_resilience_experiment()