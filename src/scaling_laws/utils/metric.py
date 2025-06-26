import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Literal, Optional
from pathlib import Path
from matplotlib.ticker import ScalarFormatter

class MetricAnalyzer:
    """Analyzes and visualizes model performance metrics with confidence intervals."""

    def __init__(self, results_df: pd.DataFrame, num_groups: int, save_path: str):
        """
        Initialize the MetricAnalyzer.

        Args:
            results_df: DataFrame with experiment results
            num_groups: Number of demographic groups
            save_path: Directory path for saving plots
        """
        self.results_df = results_df.copy()
        self.num_groups = num_groups
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        model_size_map = {
            'hiera_tiny_224.mae': 27.1,
            'hiera_small_224.mae': 34.2,
            'hiera_base_224.mae': 50.8,
            'hiera_large_224.mae': 213.0,
            'hiera_huge_224.mae': 671.0
        }

        self.results_df['model_size_million'] = self.results_df['pretrain'].map(model_size_map)

    def create_demographic_frac(self, metric='auroc') -> None:
        """Create demographic fraction columns for each group."""
        for group in ['gender', 'age']:
            frac_column = self._calculate_frac(group, metric)
            self.results_df[f'{group}_frac_{metric}'] = frac_column

    def _create_line_plot(self, x_column: str, y_column: str, y_label: str,
                           x_label: str, title: str, filename: str, hue='pretrain',
                            y_ref=None, scatter=False) -> None:
        """Create line plot with automatic confidence intervals using seaborn."""
        plt.figure(figsize=(10, 6))

        # Seaborn calcula automaticamente média e CI
        if scatter:
            mean_df = self.results_df.groupby([x_column, hue])[y_column].mean().reset_index()
            
            sns.scatterplot(
                data=mean_df,
                x=x_column,
                y=y_column,
                hue=hue,
                marker='o',
                s=50
                )
        else:
            sns.lineplot(
                data=self.results_df, 
                x=x_column, 
                y=y_column,
                hue=hue, 
                marker='o', 
                linewidth=2,
                errorbar='ci'  # Calcula CI 95% automaticamente
            )

        # Set logarithmic scale for x-axis
        

        if x_column == 'model_size_million':
            plt.xscale('log')
            xticks = [27.1, 34.2, 50.8, 213.0, 671.0]
            plt.xticks(xticks, labels=['27M', '34M', '51M', '213M', '671M'])
        else:
            plt.xscale('log')
            xticks = sorted(self.results_df[x_column].unique())
            plt.xticks(xticks, labels=[str(int(x)) for x in xticks])
            
            if y_ref is not None:

                D = np.array(xticks)
                y_ref = y_ref(D)
                plt.plot(D, y_ref, linestyle='--', color='red', linewidth=2, 
                        label='L(D)', alpha=0.8)
        
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plt.savefig(self.save_path / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _calculate_max_min(self, group: str, metric: str = 'auroc') -> tuple[float, float]:
        """Calculate max and min values for a group."""
        if group == 'gender':
            gender_columns = [f'test_gender_{metric}_{i}' for i in range(2)]
            max_column = self.results_df[gender_columns].max(axis=1)
            min_column = self.results_df[gender_columns].min(axis=1)
            return max_column, min_column
        elif group == 'age':
            age_columns = [f'test_age_{metric}_{i}' for i in range(self.num_groups)]
            max_column = self.results_df[age_columns].max(axis=1)
            min_column = self.results_df[age_columns].min(axis=1)
            return max_column, min_column
        else:
            raise ValueError("group must be 'gender' or 'age'")

    def _calculate_frac(self, group: Literal['gender', 'age'], metric: str = 'auroc') -> str:
        max_vals, min_vals = self._calculate_max_min(group, metric)
        return min_vals / max_vals

    def plot_model_efficiency(self, metric: str = 'auroc') -> None:
        """Plot model performance vs model size."""
        self._create_line_plot(
            x_column='model_size_million',
            y_column=f'test_{metric}',
            x_label='Model Size (Million Parameters)',
            y_label=f'Test {metric.lower()}',
            title=f'{metric.upper()} vs Model Size',
            filename=f'model_efficiency_{metric}.png',
            hue='fraction'
        )

    def plot_data_efficiency(self, metric: str = 'auroc') -> None:
        """Plot model performance vs training data fraction."""
        self._create_line_plot(
            x_column='fraction',
            y_column=f'test_{metric}',
            y_label=metric.upper(),
            x_label='Examples per class',
            title=f'{metric.upper()} vs Training Data Fraction',
            filename=f'data_efficiency_{metric}.png',
            # y_ref=lambda D: 0.3839 * (D ** -0.6998) + 0.5463
        )

    def plot_frac_efficiency(self, group: Literal['gender', 'age'], metric: str = 'auroc') -> None:
        """Plot demographic frac vs training data fraction."""
        self._create_line_plot(
            x_column='fraction',
            y_column=f'{group}_frac_{metric}',
            y_label=f'{group.title()} Frac % ({metric.upper()})',
            x_label='Examples per class',
            title=f'{group.title()} Performance Frac vs Training Data',
            filename=f'frac_efficiency_{group}.png'
        )

    def plot_model_size(self, group: Literal['gender', 'age'], metric: str = 'auroc') -> None:
        """Plot model size vs fairness fraction metric with lines for each fraction dataset."""
        self._create_line_plot(
            x_column='model_size_million',
            y_column=f'{group}_frac_{metric}',
            y_label=f'{group.title()} Frac % ({metric.upper()})',
            x_label='Model Size (Million Parameters)',
            title=f'{group.title()} Performance Frac vs Model Size',
            filename=f'frac_model_{group}.png',
            hue='fraction'
        )

    def generate_all_plots(self, metric: str = 'auroc') -> None:
        """Generate all analysis plots."""
        print("Generating plots...")
        
        self.create_demographic_frac(metric)
        
        self.plot_data_efficiency(metric)
        self.plot_data_efficiency('loss')
        self.plot_model_efficiency('loss')
        print("✓ Data efficiency plot created")

        for group in ['age', 'gender']:
            self.plot_frac_efficiency(group, metric)
            self.plot_model_size(group, metric)
            print(f"✓ {group.title()} analysis plots created")

        print(f"All plots saved to: {self.save_path}")
