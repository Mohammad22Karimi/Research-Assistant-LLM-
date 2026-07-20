"""
Required function 2 - compare_results()
=======================================
Compares the performance of all papers:
    * prints a comparison table (pandas)
    * saves a grouped bar chart (matplotlib) as "comparison_chart.png"

It simply reuses extract_experimental_results() and adds presentation.
"""

import matplotlib

matplotlib.use("MacOSX")  # save charts to file - works even without a display
import matplotlib.pyplot as plt

from functions.results_extraction import extract_experimental_results

CHART_FILE = "comparison_chart.png"


def compare_results(df=None):
    """
    Purpose : Produce the comparison table and chart for all papers.
    Inputs  : df (pandas.DataFrame, optional) - metrics table.  If None,
              extract_experimental_results() is called to build it (this
              lets the CLI reuse an already-extracted table).
    Outputs : df (pandas.DataFrame) - the comparison table that was shown.
              Side effects: prints the table, saves the chart PNG.
    """
    if df is None:
        df = extract_experimental_results()

    if df.empty or df.dropna(how="all").empty:
        print("No metrics could be extracted - nothing to compare.")
        return df

    # ---- 1. comparison table ----
    print("\n=== Comparison table ===")
    # to_string() prints the whole table nicely without truncation.
    print(df.to_string())

    # ---- 2. comparison chart ----
    # A grouped bar chart: one group of bars per paper, one bar per metric.
    # pandas' built-in .plot() keeps this a one-liner.
    ax = df.plot(kind="bar", figsize=(10, 6), rot=20)
    ax.set_title("Comparison of experimental results across papers")
    ax.set_xlabel("Paper")
    ax.set_ylabel("Metric value")
    ax.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(CHART_FILE, dpi=150)
    plt.close()
    print(f"\nChart saved to: {CHART_FILE}")

    return df
