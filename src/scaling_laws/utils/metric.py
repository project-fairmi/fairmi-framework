import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Metric:
    def __init__(self, results_df: pd.DataFrame, num_groups: int, path: str):
        """
        Initialize the MetricCalculator class.
        
        Args:
            results_df (pd.DataFrame): DataFrame with the results from the experiments.
            num_groups (int): Number of groups (e.g., age or gender groups).
            path (str): Path to save the generated plots.
        """
        self.results_df = results_df
        self.num_groups = num_groups
        self.path = path

    def calculate_statistics(self, column, n=3):
        """
        Calculate the mean, standard deviation, and 95% confidence interval for the specified column.
        
        Parameters:
        - column: The name of the column to calculate statistics for.
        - n: The number of observations per group (default is 3).
        
        Returns:
        - stats: DataFrame with calculated statistics.
        """
        stats = self.results_df.groupby(['version', 'fraction', 'pretrain'])[column].agg(['mean', 'std']).reset_index().sort_values(by='fraction')
        stats['ci95_hi'] = stats['mean'] + 1.96 * (stats['std'] / np.sqrt(n))
        stats['ci95_lo'] = stats['mean'] - 1.96 * (stats['std'] / np.sqrt(n))
        return stats

    def plot_with_confidence_interval(self, stats, y_label, title, filename):
        """
        Create a line plot with confidence intervals.
        
        Parameters:
        - stats: DataFrame containing statistics to plot.
        - y_label: Label for the y-axis.
        - title: Title of the plot.
        - filename: Name of the file to save the plot.
        """
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=stats, x='fraction', y='mean', hue='pretrain', marker='o')

        for pretrain in stats['pretrain'].unique():
            subset = stats[stats['pretrain'] == pretrain]
            plt.fill_between(subset['fraction'], subset['ci95_lo'], subset['ci95_hi'], alpha=0.3)

        plt.xticks([0.25, 0.5, 0.75, 1])
        plt.xlabel('Percentage of training data')
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(f"{self.path}/{filename}", dpi=300)

    def plot_data_efficiency(self, metric='auroc'):
        """
        Plots the AUROC with standard deviation as a line plot for each version of the model.
        
        Parameters:
        - metric: The performance metric to use, such as 'auroc'.
        """
        stats = self.calculate_statistics(f'test_{metric}')
        self.plot_with_confidence_interval(
            stats,
            y_label='AUROC',
            title='AUROC by Fraction with 95% Confidence Interval',
            filename='data_efficiency.png'
        )

    def plot_gap_efficiency(self, gap_type, metric='auroc'):
        """
        Plots the gap for gender or age AUROC with a 95% confidence interval.
        
        Parameters:
        - gap_type: 'gender' or 'age'. Determines which gap to calculate and plot.
        - metric: The performance metric to use, such as 'auroc'.
        """
        gap_column = self.calculate_gap(gap_type, metric)
        stats = self.calculate_statistics(gap_column)
        self.plot_with_confidence_interval(
            stats,
            y_label=f'Gap {gap_type.capitalize()} % ({metric})',
            title=f'Gap {gap_type.capitalize()} % ({metric}) by Fraction with 95% Confidence Interval',
            filename=f'gap_efficiency_{gap_type}.png'
        )

    def calculate_max_age(self, metric):
        """
        Calculate the maximum value for age groups based on the provided metric.
        
        Parameters:
        - metric: The performance metric to use, such as 'auroc'.
        
        Returns:
        - max_age: The maximum value across age groups.
        """
        columns = [f'test_age_{metric}_{i}' for i in range(self.num_groups)]
        max_age = self.results_df[columns].max(axis=1)
        return max_age

    def calculate_min_age(self, metric):
        """
        Calculate the minimum value for age groups based on the provided metric.
        
        Parameters:
        - metric: The performance metric to use, such as 'auroc'.
        
        Returns:
        - min_age: The minimum value across age groups.
        """
        columns = [f'test_age_{metric}_{i}' for i in range(self.num_groups)]
        min_age = self.results_df[columns].min(axis=1)
        return min_age

    def calculate_gap(self, gap_type, metric):
        """
        Calculate the gap for gender or age based on the provided metric.
        
        Parameters:
        - gap_type: 'gender' or 'age'. Determines which gap to calculate.
        - metric: The performance metric to use, such as 'auroc'.
        
        Returns:
        - gap_column: The name of the new column added to results_df.
        """
        if gap_type == 'gender':
            gap_column = 'gap_gender'
            self.results_df[gap_column] = abs(self.results_df[f'test_gender_{metric}_0'] - self.results_df[f'test_gender_{metric}_1']) * 100
        elif gap_type == 'age':
            gap_column = 'gap_age'
            columns = [f'test_age_{metric}_{i}' for i in range(self.num_groups)]
            self.results_df[gap_column] = (
                self.results_df[columns].max(axis=1) -
                self.results_df[columns].min(axis=1)
            ) * 100
        else:
            raise ValueError("gap_type must be either 'gender' or 'age'")
        
        return gap_column

    def plot_gap_vs_metric(self, gap_type, metric='auroc'):
        """
        Plots the test AUROC against the gap for gender or age with a 95% confidence interval.
        
        Parameters:
        - gap_type: 'gender' or 'age'. Determines which gap to calculate and plot.
        - metric: The performance metric to use, such as 'auroc'.
        """
        gap_column = self.calculate_gap(gap_type, metric)

        # Group by 'version', 'pretrain', and 'fraction' and calculate the mean and std for the metric and gap
        stats = self.results_df.groupby(['version', 'pretrain', 'fraction'])[[f'test_{metric}', gap_column]].agg(['mean', 'std']).reset_index()

        # Create the plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=stats,
            y=(gap_column, 'mean'),
            x=(f'test_{metric}', 'mean'),
            hue='fraction',
            style='pretrain',
            marker='o',
            palette='spring'
        )

        plt.xlabel(f'{metric.upper()}')
        plt.ylabel(f'Gap {gap_type.capitalize()} % ({metric.upper()})')
        plt.title(f'{metric.upper()} by Gap {gap_type.capitalize()} {metric.upper()} with 95% Confidence Interval')
        plt.legend(loc='lower left')
        plt.savefig(f"{self.path}/gap_vs_metric_{gap_type}.png", dpi=300)

    def plot_max_min(self, type, metric='auroc'):
        """
        Plots the maximum and minimum age group performance based on the provided metric.
        
        Parameters:
        - metric: The performance metric to use, such as 'auroc'.
        """
        if type == 'age':
            max = self.calculate_max_age(metric)
            min = self.calculate_min_age(metric)
            x_label = 'Max Age AUROC'
            y_label = 'Min Age AUROC'
        elif type == 'gender':
            max = 'test_gender_auroc_0'
            min = 'test_gender_auroc_1'
            x_label = 'Female (AUROC)'
            y_label = 'Male (AUROC)'
        stats = self.results_df.groupby(['version', 'fraction', 'pretrain'])[[max, min]].agg(['mean', 'std']).reset_index().sort_values(by='fraction')

        plt.figure(figsize=(10, 6))

        # Create the scatter plot with max AUROC values on the X-axis and min on the Y-axis
        sns.scatterplot(data=stats, x=(max, 'mean'), y=(min, 'mean'),
                                    style='pretrain',
                                    hue='fraction', palette='spring',
                                    s=200)

        # Add identity line
        limits = [min(stats[(max, 'mean')].min(), stats[(min, 'mean')].min()),
                max(stats[(max, 'mean')].max(), stats[(min, 'mean')].max())]

        plt.plot(limits, limits, 'k--', lw=2)  # Identity line

        # Set grid and labels
        plt.grid(True, linestyle=':', color='gray')
        plt.xlabel(x_label, fontsize=16)
        plt.ylabel(y_label, fontsize=16)
        plt.legend(loc='lower right', fontsize=14, title_fontsize=12)
        plt.savefig(f"{self.path}/max_min_{type}.png", dpi=300)
        
    def run_metrics(self):
        """
        Runs all the metric calculations and plots.
        """
        self.plot_data_efficiency()
        self.plot_gap_efficiency('age')
        self.plot_gap_efficiency('gender')
        self.plot_gap_vs_metric('age')
        self.plot_gap_vs_metric('gender')