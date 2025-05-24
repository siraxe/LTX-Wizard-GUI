import matplotlib
matplotlib.use('Agg') # Set the backend to Agg
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)

def save_loss_plot(
    output_dir: Path,
    loss_history: list[float],
    avg_loss_history: list[float],
    avg_loss_steps: list[int],
    lowest_avg_loss_history: list[float],
    lowest_avg_loss_steps: list[int],
    highest_avg_loss_history: list[float],
    highest_avg_loss_steps: list[int],
    global_step: int,
    base_output_dir: Path
) -> None:
    """Generate and save a plot of the training loss history."""
    if not loss_history:
        logger.warning("No loss data recorded to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, label="Step Loss", alpha=0.5) # Make step loss slightly transparent

    # Plot the overall average loss
    if avg_loss_history:
        plt.plot(avg_loss_steps, avg_loss_history, color='orange', label='Avg Loss (last N steps)') # Update label

    # Plot the lowest 5 average loss
    if lowest_avg_loss_history:
        plt.plot(lowest_avg_loss_steps, lowest_avg_loss_history, color='green', label='Avg Lowest 5 Loss (last N steps)') # Update label

    # Plot the highest 5 average loss
    if highest_avg_loss_history:
        plt.plot(highest_avg_loss_steps, highest_avg_loss_history, color='red', label='Avg Highest 5 Loss (last N steps)') # Update label

    plt.xlabel("Optimization Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.grid(True)
    plt.legend()

    plot_path = output_dir / f"loss_plot_step_{global_step:06d}.png"
    plt.savefig(plot_path)
    plt.close() # Close the plot to free up memory
    logger.info(f"📈 Loss plot for step {global_step} saved to {plot_path.relative_to(base_output_dir)}") 