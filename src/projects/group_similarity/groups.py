from typing import List, Union
import torch
import numpy as np
import matplotlib.pyplot as plt 
import os

from src.datamodule.datacuration import hierarchical_kmeans_gpu as hkmg
from src.datamodule.datacuration.utils import group_number

from src.projects.group_similarity import config


class CreateGroups:
    def __init__(self, embeddings: np.ndarray, n_groups: Union[List, int] = 2,
                  n_clusters_base: List = [100], log: str = None, name: str = None):
        
        if isinstance(embeddings, str):
            self.embeddings = torch.load(embeddings)
            base_name = os.path.basename(embeddings)
            self.name = os.path.splitext(base_name)[0]
            self.log = os.path.dirname(embeddings)
        else:
            self.embeddings = embeddings
            self.name = name
            self.log = log
        
        if 'tsne' in self.name:
            self.embeddings = self.embeddings['tsne_embeddings']
        elif 'umap' in self.name:
            self.embeddings = self.embeddings['umap_embeddings']
        else:
            self.embeddings = self.embeddings['embeddings']

        if isinstance(n_groups, int):
            self.n_groups = [n_groups]
        else:
            self.n_groups = n_groups

        self.n_clusters_base = n_clusters_base
        self.n_levels = len(n_clusters_base) + 1


        self.clusters = {}
        self.groups = {}
        
        self._calculate_groups()

    def _calculate_groups(self):
        for group in self.n_groups:

            data = self.embeddings

            self.clusters[group] = hkmg.hierarchical_kmeans(
                data=torch.tensor(data, device="cuda", dtype=torch.float32),
                n_clusters=self.n_clusters_base + [group],
                n_levels=self.n_levels,
                verbose=False,
            )

            self.groups[group] = group_number(self.clusters[group])

    def save_groups(self):
        if self.groups:
            np.save(f"{self.log}/groups_{self.name}.npy", self.groups)
        else:
            print("No groups to save. Run _calculate_groups first.")

    def plot_and_save_groups(self):
        if self.groups is None or self.embeddings is None:
            print("No groups or embeddings to plot. Run _calculate_groups first.")
            return
        elif self.embeddings.shape[1] != 2:
            print("Embeddings are not 2D. Cannot plot.")
            return

        data = self.embeddings
        
        num_groups_to_plot = len(self.groups)
        if num_groups_to_plot == 0:
            print("No groups to plot.")
            return

        figh = num_groups_to_plot
        figw = 2
        
        fig, axs = plt.subplots(figh, figw, figsize=(6 * figw, 5 * figh), squeeze=False)

        for i, (n_group_key, group_assignments) in enumerate(self.groups.items()):
            # Plot original data in the first column
            axs[i, 0].scatter(data[:, 0], data[:, 1], alpha=0.2)
            axs[i, 0].set_title("original data" if i == 0 else "", fontsize=20)
            axs[i, 0].tick_params(labelsize=16)
            if i > 0: # Remove x-axis labels for subplots that are not the bottom one
                axs[i, 0].set_xticks([])
            if i < num_groups_to_plot - 1: # Remove y-axis labels for subplots that are not the leftmost one
                axs[i, 0].set_yticks([])

            # Plot data colored by group in the second column
            scatter = axs[i, 1].scatter(data[:, 0], data[:, 1], c=group_assignments, cmap='tab20', alpha=0.7)
            axs[i, 1].set_title(f"data colored by group (n_groups={n_group_key})" if i == 0 else f"n_groups={n_group_key}", fontsize=20)
            axs[i, 1].tick_params(labelsize=16)
            plt.colorbar(scatter, ax=axs[i, 1], label="group")
            if i > 0: # Remove x-axis labels for subplots that are not the bottom one
                axs[i, 1].set_xticks([])
            if i < num_groups_to_plot - 1: # Remove y-axis labels for subplots that are not the leftmost one
                axs[i, 1].set_yticks([])

        plt.tight_layout()
        plt.savefig(f"{self.log}/groups_{self.name}_combined_plot.png") # Save a single combined figure
        plt.close(fig) # Close the figure to free memory

if __name__ == "__main__":
    groups_creator = CreateGroups(embeddings=config['groups']['embeddings_path'], n_groups=config['groups']['n_groups'],
                                   n_clusters_base=config['groups']['n_clusters_base'], log=config['groups']['log'])
    groups_creator.save_groups()
    groups_creator.plot_and_save_groups()
