import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_summary(
    column, title, ax=None, plot_type="hist", show_stats=True  # options: 'hist', 'box'
):
    """
    Plot a histogram or boxplot of the given column, optionally with mean/median lines.

    Parameters:
        column (pd.Series): Column to plot
        title (str): Title for the plot
        ax (matplotlib.axes._subplots.AxesSubplot): Axis to plot on. If None, creates one.
        plot_type (str): 'hist' for histogram or 'box' for boxplot
        show_stats (bool): Whether to show mean/median lines
    """
    if ax is None:
        ax = plt.gca()

    if plot_type == "hist":
        ax.hist(
            column,
            bins=30,
            edgecolor="white",
            color="steelblue",
            alpha=0.8,
            density=True,
        )

        if show_stats:
            mean_val = column.mean()
            median_val = column.median()
            std_dev = column.std()

            ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean = ${mean_val:,.0f}",
            )
            ax.axvline(
                median_val,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Median = ${median_val:,.0f}",
            )
            ax.axvline(
                mean_val - std_dev, color="black", linestyle=":", linewidth=2, alpha=0.7
            )
            ax.axvline(
                mean_val + std_dev,
                color="black",
                linestyle=":",
                linewidth=2,
                alpha=0.7,
                label=f"±1 Std = ${std_dev:,.0f}",
            )

        ax.set_ylabel("Density")

    elif plot_type == "box":
        ax.boxplot(
            column.dropna(),
            vert=False,
            patch_artist=True,
            boxprops=dict(facecolor="lightcoral", color="darkred"),
            medianprops=dict(color="white"),
            whiskerprops=dict(color="darkred"),
            capprops=dict(color="darkred"),
            flierprops=dict(markerfacecolor="gray", marker="o", alpha=0.3),
        )
        ax.set_yticks([])  # optional: remove y-axis ticks

        if show_stats:
            mean_val = column.mean()
            ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean = ${mean_val:,.0f}",
            )

        ax.set_ylabel("")

    # Common formatting
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{column.name} ($)", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(axis="x", alpha=0.3)

    if column.name == "price":
        ax.xaxis.set_major_formatter("${x:,.0f}")

    if show_stats:
        ax.legend(fontsize=10, loc="upper right")


def custom_plot(
    df: pd.DataFrame, feature_col: str, target_col: str, x_label: str
) -> None:
    """
    Creates a dual-plot visualization showing:
    1. Count plot of feature values
    2. Bar plot of average and median of prices by feature category

    :param df: Current Dataframe
    :param feature_col: Column name for the feature to analyze
    :param target_col: Column name for the target to analyze
    :param x_label: Label of the feature column
    """
    avg_group = df.groupby(feature_col)[target_col].mean().sort_values()
    value_counts = (
        df[feature_col].value_counts().reindex(avg_group.index)
    )  # align with bar order

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Price Analysis by {x_label}", y=1.02, fontsize=22)

    # Plot 1: Count plot of feature_col
    cmap = plt.get_cmap("Oranges")
    colors = cmap(np.linspace(0.3, 0.8, len(avg_group)))
    bars = ax1.bar(value_counts.index.astype(str), value_counts.values, color=colors)

    ax1.set_title(f"Count of {x_label} Category", pad=15, fontsize=18)
    ax1.set_ylabel("Count", labelpad=10, fontsize=15)
    ax1.set_xlabel(x_label, labelpad=10, fontsize=15)

    # Add count and percent labels
    total = value_counts.sum()
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count_val = value_counts.iloc[i]
        pct_val = value_counts.iloc[i] / total

        # Percent in middle
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2,
            f"{count_val:,.0f}\n{pct_val:.1%}",
            ha="center",
            va="center",
            fontsize=13,
            color="white",
            bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"),
        )

    # Plot 2: Bar plot of average prices by category
    # Compute both average and median per category
    grouped = df.groupby(feature_col)[target_col]
    avg_group = grouped.mean().sort_values()
    med_group = grouped.median().reindex(avg_group.index)

    # Bar width and positions
    x = np.arange(len(avg_group))
    width = 0.35

    # Bar plot: Average and Median side-by-side
    bars1 = ax2.bar(
        x - width / 2,
        avg_group.values,
        width=width,
        label="Mean",
        color="mediumseagreen",
    )
    bars2 = ax2.bar(
        x + width / 2, med_group.values, width=width, label="Median", color="coral"
    )

    ax2.set_title(
        f"Average and Median of Price by {x_label} Category", pad=15, fontsize=18
    )
    ax2.set_ylabel("Average Price", labelpad=10, fontsize=15)
    ax2.set_xlabel(x_label, labelpad=10, fontsize=15)
    ax2.set_xticks(range(len(value_counts)))
    ax2.set_xticklabels(value_counts.index.astype(str))
    ax2.legend(fontsize=15)

    # Add value labels and percentages to bars
    for bars, values in [(bars1, avg_group), (bars2, med_group)]:
        for bar, val in zip(bars, values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() / 2,
                f"{val:,.0f}",
                ha="center",
                va="center",
                color="white",
                fontsize=13,
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"),
            )

    plt.tight_layout()
    plt.show()


def plot_price_violin(
    column, title="Price Distribution (Violin Plot)", ax=None, show_mean=True
):
    """
    Plots a violin plot of the given column, optionally overlaying the mean.

    Parameters:
        column (pd.Series): The column to plot
        title (str): Plot title
        ax (matplotlib.axes._subplots.AxesSubplot): Axis to plot on (optional)
        show_mean (bool): Whether to show the mean as a vertical line
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Seaborn violin plot
    sns.violinplot(
        x=column, ax=ax, inner="box", color="skyblue"  # show embedded boxplot
    )

    if show_mean:
        mean_val = column.mean()
        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean = ${mean_val:,.0f}",
        )
        ax.legend(fontsize=10, loc="upper right")

    # Formatting
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(f"{column.name} ($)", fontsize=12)
    ax.set_yticks([])  # no vertical ticks
    ax.grid(axis="x", alpha=0.3)

    if column.name == "price":
        ax.xaxis.set_major_formatter("${x:,.0f}")


def plot_scatter(
    data,
    x_col="living_in_m2",
    y_col="price",
    trendline=True,
    hue=None,
    alpha=0.6,
    palette="viridis",
):
    """
    Create an enhanced scatter plot with optional trendline and coloring.

    Parameters:
        x_col (str): X-axis column name (default: 'living_in_m2')
        y_col (str): Y-axis column name (default: 'price')
        data (DataFrame): Input dataframe (default: df)
        trendline (bool): Add regression line (default: True)
        hue (str): Column for color encoding (default: None)
        alpha (float): Point transparency (default: 0.6)
        palette (str): Color palette (default: 'viridis')
    """

    # Create scatter plot
    if hue:
        sns.scatterplot(
            x=x_col, y=y_col, hue=hue, data=data, alpha=alpha, palette=palette, s=80
        )
    else:
        sns.scatterplot(
            x=x_col, y=y_col, data=data, color="steelblue", alpha=alpha, s=80
        )

    # Add trendline if requested
    if trendline:
        sns.regplot(
            x=x_col,
            y=y_col,
            data=data,
            scatter=False,
            color="red",
            line_kws={"linestyle": "--", "alpha": 0.7},
        )

    # Formatting
    plt.title(f"{y_col} vs {x_col}", fontsize=16, pad=20)
    plt.xlabel(x_col.replace("_", " ").title(), fontsize=12)
    plt.ylabel(y_col.replace("_", " ").title() + " ($)", fontsize=12)

    # Format y-axis as currency if plotting price
    if y_col == "price":
        plt.gca().yaxis.set_major_formatter("${x:,.0f}")

    # Format x-axis for square meters if living area
    if x_col == "living_in_m2":
        plt.xlabel("Living Area (m²)", fontsize=12)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_correlation(
    data,
    columns=None,
    method="pearson",
    annot=True,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    title=None,
):
    """
    Plot a correlation matrix with flexible input options.

    Parameters:
        data (DataFrame): Input dataframe
        columns (list/str): Specific columns to include (None for all numeric)
        method (str): Correlation method ('pearson', 'kendall', 'spearman')
        annot (bool): Whether to annotate correlation values
        cmap (str): Color map for the heatmap
        vmin (float): Min value limits for color scale
        vmax (float): Min value limits for color scale
        title (str): Custom title for the plot
    """
    # Select columns if specified
    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        corr_data = data[columns]
    else:
        corr_data = data.select_dtypes(include=["number"])

    # Calculate correlation matrix
    corr_matrix = corr_data.corr(method=method)

    # Create plot
    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt=".2f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    # Formatting
    if title is None:
        title = f"{method.title()} Correlation Matrix"
        if columns is not None:
            title += f" ({len(columns)} features)"

    plt.title(title, fontsize=14, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
