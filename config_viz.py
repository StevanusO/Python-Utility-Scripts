import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio

def init_viz(style: str = "seaborn", figsize: tuple = (10, 6), dpi: int = 150) -> None:
    # Matplotlib Global Config
    plt.rcParams.update({
        "figure.figsize": figsize,
        "figure.dpi": dpi,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Times New Roman", "Georgia", "serif"],
        "text.usetex": False,
        # Grid
        "axes.grid": True,
        "grid.color": "black",
        "grid.alpha": 0.12,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        # Axes
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
        "axes.labelweight": "normal",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Seaborn Config
    if style == "seaborn":
      sns.set_style("white", {"axes.grid": True, "grid.color": "black", "grid.alpha": 0.12})
      sns.set_palette("colorblind")
      sns.set_context("paper")
    # Plotly Config
    elif style == "plotly":
        pio.templates.default = "plotly_white"
