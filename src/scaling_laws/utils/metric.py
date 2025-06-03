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
    
    def _calculate_statistics(self, column: str, n: int = 3) -> pd.DataFrame:
        """Calculate mean, std, and 95% CI for a column grouped by experimental conditions."""
        stats = (self.results_df
                .groupby(['version', 'fraction', 'pretrain'])[column]
                .agg(['mean', 'std'])
                .reset_index()
                .sort_values('fraction'))
        
        # Calculate 95% confidence interval
        margin_error = 1.96 * (stats['std'] / np.sqrt(n))
        stats['ci95_hi'] = stats['mean'] + margin_error
        stats['ci95_lo'] = stats['mean'] - margin_error
        
        return stats
    
    def _create_line_plot(self, stats: pd.DataFrame, y_label: str, 
                         title: str, filename: str) -> None:
        """Create line plot with confidence intervals."""
        plt.figure(figsize=(10, 6))
        
        # Main line plot
        sns.lineplot(data=stats, x='fraction', y='mean', 
                    hue='pretrain', marker='o', linewidth=2)
        
        # Add confidence intervals
        for pretrain in stats['pretrain'].unique():
            subset = stats[stats['pretrain'] == pretrain]
            plt.fill_between(subset['fraction'], subset['ci95_lo'], 
                           subset['ci95_hi'], alpha=0.3)
        
        # Formatting
        plt.xticks([0.25, 0.5, 0.75, 1.0])
        plt.xlabel('Percentage of Training Data', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)

        # Optimize axis limits
        y_max = stats['mean'].max() + (stats['mean'].max() * 0.1)
        x_max = stats['fraction'].max() + (stats['fraction'].max() * 0.1)
        plt.ylim(0, y_max)
        plt.xlim(0, x_max)

        plt.tight_layout()
        
        # Save plot
        plt.savefig(self.save_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_max_min(self, group_type: str, metric: str = 'auroc') -> tuple[float, float]:
        """Calculate max and min values for a group."""
        if group_type == 'gender':
            max_val = max(
                self.results_df[f'test_gender_{metric}_0'],
                self.results_df[f'test_gender_{metric}_1']
            )
            min_val = min(
                self.results_df[f'test_gender_{metric}_0'],
                self.results_df[f'test_gender_{metric}_1']
            )
            return max_val, min_val
        elif group_type == 'age':
            age_columns = [f'test_age_{metric}_{i}' for i in range(self.num_groups)]
            max_val = self.results_df[age_columns].max(axis=1)
            min_val = self.results_df[age_columns].min(axis=1)
            return max_val.max(), min_val.min()
        else:
            raise ValueError("group_type must be 'gender' or 'age'")

    def _calculate_gap(self, gap_type: Literal['gender', 'age'],
                      metric: str = 'auroc', as_percentage: bool = True) -> str:
        """
        Calculate performance gap between demographic groups.

        Args:
            gap_type: The type of gap to calculate ('gender' or 'age').
            metric: The metric to use for gap calculation. Default is 'auroc'.
            as_percentage: Whether to return the gap as a percentage. Default is True.

        Returns:
            The name of the column containing the gap values.
        """
        gap_column = f'gap_{gap_type}'
        fraction_column = f'frac_{gap_type}'

        # Calculate max and min values using the _calculate_max_min function
        max_val, min_val = self._calculate_max_min(gap_type, metric)

        # Calculate gap and fraction
        if as_percentage:
            gap = abs(max_val - min_val) * 100
        else:
            gap = abs(max_val - min_val)

        fraction = min_val / max_val

        # Store results in the DataFrame
        self.results_df[gap_column] = gap
        self.results_df[fraction_column] = fraction

        return gap_column
    
    def plot_data_efficiency(self, metric: str = 'auroc') -> None:
        """Plot model performance vs training data fraction."""
        stats = self._calculate_statistics(f'test_{metric}')
        self._create_line_plot(
            stats=stats,
            y_label=metric.upper(),
            title=f'{metric.upper()} vs Training Data Fraction',
            filename='data_efficiency.png'
        )
    
    def plot_gap_efficiency(self, gap_type: Literal['gender', 'age'], 
                           metric: str = 'auroc') -> None:
        """Plot demographic gap vs training data fraction."""
        gap_column = self._calculate_gap(gap_type, metric)
        stats = self._calculate_statistics(gap_column)
        
        self._create_line_plot(
            stats=stats,
            y_label=f'{gap_type.title()} Gap % ({metric.upper()})',
            title=f'{gap_type.title()} Performance Gap vs Training Data',
            filename=f'gap_efficiency_{gap_type}.png'
        )
    
    def plot_gap_vs_performance(self, gap_type: Literal['gender', 'age'], 
                               metric: str = 'auroc') -> None:
        """Plot performance vs demographic gap scatter plot."""
        gap_column = self._calculate_gap(gap_type, metric)
        
        # Calculate statistics for both metrics
        stats = (self.results_df
                .groupby(['version', 'pretrain', 'fraction'])
                [[f'test_{metric}', gap_column]]
                .agg(['mean', 'std'])
                .reset_index())
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=stats,
            x=(f'test_{metric}', 'mean'),
            y=(gap_column, 'mean'),
            hue='fraction',
            style='pretrain',
            s=100,
            palette='viridis'
        )
        
        plt.xlabel(f'{metric.upper()}', fontsize=12)
        plt.ylabel(f'{gap_type.title()} Gap % ({metric.upper()})', fontsize=12)
        plt.title(f'Performance vs {gap_type.title()} Gap', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(self.save_path / f'gap_vs_performance_{gap_type}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_group_comparison(self, group_type: Literal['gender', 'age'],
                             metric: str = 'auroc') -> None:
        """Plot max vs min group performance."""
        # Calculate max and min values using the _calculate_max_min function
        max_vals, min_vals = self._calculate_max_min(group_type, metric)

        # Set labels based on group type
        if group_type == 'age':
            x_label, y_label = f'Max Age {metric.upper()}', f'Min Age {metric.upper()}'
        elif group_type == 'gender':
            x_label, y_label = f'Female {metric.upper()}', f'Male {metric.upper()}'
        else:
            raise ValueError("group_type must be 'gender' or 'age'")
        
        # Create temporary DataFrame for grouping
        temp_df = self.results_df.copy()
        temp_df['max_vals'] = max_vals
        temp_df['min_vals'] = min_vals
        
        stats = (temp_df
                .groupby(['version', 'fraction', 'pretrain'])
                [['max_vals', 'min_vals']]
                .agg(['mean', 'std'])
                .reset_index())
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=stats,
            x=('max_vals', 'mean'),
            y=('min_vals', 'mean'),
            hue='fraction',
            style='pretrain',
            s=150,
            palette='viridis'
        )
        
        # Add identity line
        lims = [
            min(stats[('max_vals', 'mean')].min(), stats[('min_vals', 'mean')].min()),
            max(stats[('max_vals', 'mean')].max(), stats[('min_vals', 'mean')].max())
        ]
        plt.plot(lims, lims, 'k--', alpha=0.7, linewidth=2, label='Perfect Equality')
        
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.title(f'{group_type.title()} Group Performance Comparison', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        plt.savefig(self.save_path / f'group_comparison_{group_type}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_all_plots(self, metric: str = 'auroc') -> None:
        """Generate all analysis plots."""
        print("Generating plots...")
        
        self.plot_data_efficiency(metric)
        print("✓ Data efficiency plot created")
        
        for gap_type in ['age', 'gender']:
            self.plot_gap_efficiency(gap_type, metric)
            self.plot_gap_vs_performance(gap_type, metric)
            self.plot_group_comparison(gap_type, metric)
            print(f"✓ {gap_type.title()} analysis plots created")
        
        print(f"All plots saved to: {self.save_path}")
