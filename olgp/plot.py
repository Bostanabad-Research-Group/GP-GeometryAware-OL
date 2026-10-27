import os
import matplotlib.pyplot as plt

def plot_training_dynamics(
    train_loss, train_epochs,
    val_loss, val_epochs,
    metric_values, metric_epochs,
    save_path,
    title="Training Dynamics",
    metric_label="Relative L2 Error",
    logger=None
):
    """
    Plots training dynamics with Loss on the left axis and a Metric on the right axis.

    Args:
        train_loss (list): List of training loss values.
        train_epochs (list): List of epochs corresponding to training loss.
        val_loss (list): List of validation loss values.
        val_epochs (list): List of epochs corresponding to validation loss.
        metric_values (list): List of metric values (e.g., L2 error).
        metric_epochs (list): List of epochs corresponding to the metric.
        save_path (str): Full path including filename to save the plot.
        title (str): Plot title.
        metric_label (str): Label for the right-hand axis (default: Relative L2 Error).
        logger (object, optional): Logger instance to log progress.
    """
    try:
        if logger:
            logger.info(f"--- Generating Plot: {title} ---")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Create plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # --- Left Axis: Losses (Train & Val) ---
        color_train = 'tab:blue'
        color_val = 'tab:cyan'

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color=color_train, fontweight='bold')

        l1 = ax1.plot(train_epochs, train_loss, label='Train Loss',
                      color=color_train, linestyle='-', linewidth=1.5, alpha=0.9)
        l2 = ax1.plot(val_epochs, val_loss, label='Val Loss',
                      color=color_val, linestyle='--', linewidth=1.5, alpha=0.9)

        ax1.tick_params(axis='y', labelcolor=color_train)
        ax1.grid(True, which='major', linestyle='--', alpha=0.5)

        # Handle Scale: Use symlog if values <= 0 exist (for GP/NLL), else log
        all_losses = train_loss + val_loss
        if any(l <= 0 for l in all_losses):
            ax1.set_yscale('symlog', linthresh=1e-3)
        else:
            ax1.set_yscale('log')

        # --- Right Axis: Metrics ---
        ax2 = ax1.twinx()
        color_metric = 'tab:red'
        ax2.set_ylabel(metric_label, color=color_metric, fontweight='bold')

        l3 = ax2.plot(metric_epochs, metric_values, label=metric_label,
                      color=color_metric, linestyle='-.', linewidth=2)

        ax2.tick_params(axis='y', labelcolor=color_metric)
        ax2.set_yscale('log')

        # --- Legend ---
        # Combine lines from both axes
        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        # Placing legend outside or upper center usually avoids covering data
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

        # Layout & Save
        plt.title(title, pad=30)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)
        plt.close()

        if logger:
            logger.info(f"Plot saved to: {save_path}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to generate plot '{title}': {e}")
        else:
            print(f"Error generating plot: {e}")
