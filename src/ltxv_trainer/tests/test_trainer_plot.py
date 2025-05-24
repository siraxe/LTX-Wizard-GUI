import os
import shutil
import unittest
from pathlib import Path
import numpy as np
import matplotlib
# Use Agg backend for testing to avoid display issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ltxv_trainer.trainer_plot import save_loss_plot

class TestTrainerPlot(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for test outputs."""
        self.test_output_dir = Path("test_output")
        self.test_output_dir.mkdir(exist_ok=True)
        self.base_output_dir = Path(".") # Simulate base output directory

    def tearDown(self):
        """Clean up the temporary directory after tests."""
        if self.test_output_dir.exists():
            # shutil.rmtree(self.test_output_dir)
            pass # Keep the directory for inspection

    def test_save_loss_plot_with_averages(self):
        """Test that save_loss_plot creates a file with average lines."""
        # Simulate loss history data
        loss_history = [i * 0.1 + np.sin(i * 0.5) + 1 for i in range(100)] # 100 steps of simulated loss

        # Simulate average loss data (e.g., calculated every 10 steps)
        avg_loss_history = [np.mean(loss_history[max(0, i-10):i]) for i in range(10, 101, 10)]
        avg_loss_steps = list(range(10, 101, 10))

        # Simulate lowest and highest 5 average loss data
        lowest_avg_loss_history = [np.mean(sorted(loss_history[max(0, i-10):i])[:5]) for i in range(10, 101, 10)]
        highest_avg_loss_history = [np.mean(sorted(loss_history[max(0, i-10):i])[-5:]) for i in range(10, 101, 10)]
        lowest_avg_loss_steps = avg_loss_steps # Same steps as overall average for simplicity
        highest_avg_loss_steps = avg_loss_steps # Same steps as overall average for simplicity

        global_step = 100 # Simulate current global step

        # Call the plotting function
        save_loss_plot(
            output_dir=self.test_output_dir,
            loss_history=loss_history,
            avg_loss_history=avg_loss_history,
            avg_loss_steps=avg_loss_steps,
            lowest_avg_loss_history=lowest_avg_loss_history,
            lowest_avg_loss_steps=lowest_avg_loss_steps,
            highest_avg_loss_history=highest_avg_loss_history,
            highest_avg_loss_steps=highest_avg_loss_steps,
            global_step=global_step,
            base_output_dir=self.base_output_dir
        )

        # Assert that the plot file was created
        expected_plot_path = self.test_output_dir / f"loss_plot_step_{global_step:06d}.png"
        self.assertTrue(expected_plot_path.exists())

    def test_save_loss_plot_no_averages(self):
        """Test that save_loss_plot creates a file even without average data."""
        # Simulate loss history data (fewer than window size for averaging)
        loss_history = [i * 0.1 for i in range(10)] # 10 steps of simulated loss

        # Empty lists for averages
        avg_loss_history = []
        avg_loss_steps = []
        lowest_avg_loss_history = []
        lowest_avg_loss_steps = []
        highest_avg_loss_history = []
        highest_avg_loss_steps = []

        global_step = 10 # Simulate current global step

        # Call the plotting function
        save_loss_plot(
            output_dir=self.test_output_dir,
            loss_history=loss_history,
            avg_loss_history=avg_loss_history,
            avg_loss_steps=avg_loss_steps,
            lowest_avg_loss_history=lowest_avg_loss_history,
            lowest_avg_loss_steps=lowest_avg_loss_steps,
            highest_avg_loss_history=highest_avg_loss_history,
            highest_avg_loss_steps=highest_avg_loss_steps,
            global_step=global_step,
            base_output_dir=self.base_output_dir
        )

        # Assert that the plot file was created
        expected_plot_path = self.test_output_dir / f"loss_plot_step_{global_step:06d}.png"
        self.assertTrue(expected_plot_path.exists())

if __name__ == '__main__':
    unittest.main() 