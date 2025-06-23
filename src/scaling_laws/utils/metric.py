import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Literal, Optional
from pathlib import Path

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

        self.create_demographic_frac()

    def create_demographic_frac(self, metric='auroc') -> None:
        """Create demographic fraction columns for each group."""
        for group in ['gender', 'age']:
            frac_column = self._calculate_frac(group, metric)
            self.results_df[f'{group}_frac_{metric}'] = frac_column

    def _create_line_plot(self, column: str, y_label: str, title: str, filename: str) -> None:
        """Create line plot with automatic confidence intervals using seaborn."""
        plt.figure(figsize=(10, 6))

        # Seaborn calcula automaticamente média e CI
        sns.lineplot(
            data=self.results_df, 
            x='fraction', 
            y=column,
            hue='pretrain', 
            marker='o', 
            linewidth=2,
            errorbar='ci'  # Calcula CI 95% automaticamente
        )

        # # Formatting - adjust xticks for log scale
        plt.xticks([0.0001, 0.0005, 0.001, 0.01], 
                  ['23', '115', '230', '2300'])
        # Set logarithmic scale for x-axis
        plt.xscale('log')
        
        
        plt.xlabel('Images', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(title, fontsize=14)
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

    def plot_data_efficiency(self, metric: str = 'auroc') -> None:
        """Plot model performance vs training data fraction."""
        self._create_line_plot(
            column=f'test_{metric}',
            y_label=metric.upper(),
            title=f'{metric.upper()} vs Training Data Fraction',
            filename='data_efficiency.png'
        )

    def plot_frac_efficiency(self, group: Literal['gender', 'age'], metric: str = 'auroc') -> None:
        """Plot demographic frac vs training data fraction."""
        frac_column = f'{group}_frac_{metric}'
        self._create_line_plot(
            column=frac_column,
            y_label=f'{group.title()} Frac % ({metric.upper()})',
            title=f'{group.title()} Performance Frac vs Training Data',
            filename=f'frac_efficiency_{group}.png'
        )

    def plot_frac_vs_performance(self, group: Literal['gender', 'age'], metric: str = 'auroc') -> None:
        """Plot Pareto front: performance vs demographic fairness."""
        frac_column_name = f'{group}_frac_{metric}'

        # Calculate statistics for both metrics
        stats = (self.results_df
                .groupby(['version', 'pretrain', 'fraction'])
                [[f'test_{metric}', frac_column_name]]
                .agg(['mean', 'std'])
                .reset_index())

        # Flatten column names
        stats.columns = ['_'.join(col).strip('_') if col[1] else col[0]
                        for col in stats.columns.values]

        # Correct Pareto front calculation
        def calculate_pareto_front(df, x_col, y_col, maximize_x=True, maximize_y=True):
            """Calculate Pareto front points."""
            pareto_points = []

            for i, point in df.iterrows():
                is_dominated = False

                for j, other in df.iterrows():
                    if i == j:
                        continue

                    # Check if current point is dominated by other point
                    x_better = (other[x_col] >= point[x_col]) if maximize_x else (other[x_col] <= point[x_col])
                    y_better = (other[y_col] >= point[y_col]) if maximize_y else (other[y_col] <= point[y_col])

                    # At least one dimension must be strictly better
                    x_strictly_better = (other[x_col] > point[x_col]) if maximize_x else (other[x_col] < point[x_col])
                    y_strictly_better = (other[y_col] > point[y_col]) if maximize_y else (other[y_col] < point[y_col])

                    if x_better and y_better and (x_strictly_better or y_strictly_better):
                        is_dominated = True
                        break

                if not is_dominated:
                    pareto_points.append(point)

            return pd.DataFrame(pareto_points)

        plt.figure(figsize=(10, 8))

        # Create scatter plot
        sns.scatterplot(
            data=stats,
            x=f'{frac_column_name}_mean',
            y=f'test_{metric}_mean',
            hue='fraction',
            style='pretrain',
            s=120,
            alpha=0.8,
            palette='Set1',
            zorder=2
        )

        # Calculate and plot Pareto front
        pareto_df = calculate_pareto_front(
            stats,
            f'{frac_column_name}_mean',
            f'test_{metric}_mean',
            maximize_x=True,
            maximize_y=True
        )

        if len(pareto_df) > 1:
            pareto_df = pareto_df.sort_values(f'{frac_column_name}_mean')
            plt.plot(
                pareto_df[f'{frac_column_name}_mean'],
                pareto_df[f'test_{metric}_mean'],
                'r--',
                alpha=0.8,
                linewidth=3,
                label='Pareto Front',
                zorder=1
            )

        # Add error bars
        plt.errorbar(
            stats[f'{frac_column_name}_mean'],
            stats[f'test_{metric}_mean'],
            xerr=stats[f'{frac_column_name}_std'],
            yerr=stats[f'test_{metric}_std'],
            fmt='none',
            alpha=0.3,
            color='gray'
        )

        plt.xlabel(f'{group.title()} Fairness Fraction', fontsize=12)
        plt.ylabel(f'{metric.upper()} Performance', fontsize=12)
        plt.title(f'Pareto Front: {metric.upper()} vs {group.title()} Fairness', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.save_path / f'pareto_front_{group}_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_group_comparison(self, group: Literal['gender', 'age'], metric: str = 'auroc') -> None:
        """Plot max vs min group performance."""
        max_vals, min_vals = self._calculate_max_min(group, metric)

        if group == 'age':
            x_label, y_label = f'Max Age {metric.upper()}', f'Min Age {metric.upper()}'
        elif group == 'gender':
            x_label, y_label = f'Female {metric.upper()}', f'Male {metric.upper()}'
        else:
            raise ValueError("group must be 'gender' or 'age'")

        # Create temporary DataFrame for seaborn
        temp_df = self.results_df.copy()
        temp_df['max_vals'] = max_vals
        temp_df['min_vals'] = min_vals

        plt.figure(figsize=(10, 6))
        
        # Use seaborn with automatic statistical calculations
        sns.scatterplot(
            data=temp_df,
            x='max_vals',
            y='min_vals',
            hue='fraction',
            style='pretrain',
            s=150
        )

        # Add identity line
        all_vals = pd.concat([temp_df['max_vals'], temp_df['min_vals']])
        lims = [all_vals.min(), all_vals.max()]
        plt.plot(lims, lims, 'k--', alpha=0.7, linewidth=2, label='Perfect Equality')

        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(f'{group.title()} Group Performance Comparison', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(self.save_path / f'group_comparison_{group}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_model_size(self, group: Literal['gender', 'age'], metric: str = 'auroc') -> None:
        """Plot model size vs fairness fraction metric with lines for each fraction dataset."""
        model_size_map = {
            'hiera_tiny_224.mae': 27.1,
            'hiera_small_224.mae': 34.2,
            'hiera_base_224.mae': 50.8,
            'hiera_large_224.mae': 213.0,
            'hiera_huge_224.mae': 671.0
        }

        self.results_df['model_size_million'] = self.results_df['pretrain'].map(model_size_map)
        frac_column = f'{group}_frac_{metric}'
        
        plot_data = self.results_df[['model_size_million', frac_column, 'fraction']].dropna()

        plt.figure(figsize=(10, 6))
        
        # Use seaborn with automatic confidence intervals
        sns.lineplot(
            data=plot_data, 
            x='model_size_million', 
            y=frac_column, 
            hue='fraction',
            marker='o',
            linewidth=2,
            errorbar='ci',
            palette='Set1'
        )

        plt.xscale('log')
        plt.xlabel('Model Size (Million Parameters)', fontsize=12)
        plt.ylabel(f'{group.title()} Frac % ({metric.upper()})', fontsize=12)
        plt.title(f'Model Size vs {group.title()} {metric.upper()} Fairness Fraction', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Training Data Fraction', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(self.save_path / f'model_size_vs_{group}_frac_{metric}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_all_plots(self, metric: str = 'auroc') -> None:
        """Generate all analysis plots."""
        print("Generating plots...")

        self.plot_data_efficiency(metric)
        print("✓ Data efficiency plot created")

        for group in ['age', 'gender']:
            self.plot_frac_efficiency(group, metric)
            self.plot_frac_vs_performance(group, metric)
            self.plot_group_comparison(group, metric)
            self.plot_model_size(group, metric)
            print(f"✓ {group.title()} analysis plots created")

        print(f"All plots saved to: {self.save_path}")
