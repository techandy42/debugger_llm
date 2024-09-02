import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(tp, tn, fp, fn, filename='confusion_matrix.png'):
    # Manually create the confusion matrix
    cm = np.array([[tp, fp],
                   [fn, tn]])
    
    # Define color mappings for TP/TN (blue shades) and FP/FN (red shades)
    blue_shades = {
        0: "#d1e7f5",  # Very light blue
        20: "#a6cde2",
        40: "#7bb3d0",
        60: "#5199bd",
        80: "#267eaa",  # Medium blue
        100: "#005c8f"  # Dark blue
    }
    
    red_shades = {
        0: "#f5d1d1",  # Very light red
        20: "#e2a6a6",
        40: "#d07b7b",
        60: "#bd5151",
        80: "#aa2626",  # Medium red
        100: "#8f0000"  # Dark red
    }
    
    # Assign colors based on the value ranges
    def get_color(value, color_dict):
        if value >= 80:
            return color_dict[100]
        elif value >= 60:
            return color_dict[80]
        elif value >= 40:
            return color_dict[60]
        elif value >= 20:
            return color_dict[40]
        elif value > 0:
            return color_dict[20]
        else:
            return color_dict[0]

    # Plot the confusion matrix
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(cm, annot=True, fmt='d', cbar=False, square=True, linewidths=0, 
                     annot_kws={"size": 20, "color": "black"},  # Set the numbers to be black
                     vmin=0, vmax=100)

    # Set the background colors manually based on the value and quadrant type
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            if (i == 0 and j == 0) or (i == 1 and j == 1):  # TP or TN
                color = get_color(value, blue_shades)
            else:  # FP or FN
                color = get_color(value, red_shades)
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color=color, edgecolor='white'))

    # Adjust the axis labels and title
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14, labelpad=5)
    plt.ylabel('True Label', fontsize=14, labelpad=5)
    plt.xticks(ticks=[0.5, 1.5], labels=['Positive', 'Negative'], fontsize=12)
    plt.yticks(ticks=[0.5, 1.5], labels=['Positive', 'Negative'], fontsize=12, rotation=0)

    # Reduce space on the left and adjust layout
    plt.gcf().subplots_adjust(left=0.2)
    plt.tight_layout()

    # Save the confusion matrix as an image
    plt.savefig(filename, format='png', dpi=300)
    
    # Show the plot
    plt.show()

# Example values for TP, TN, FP, FN
tp = 78  # True Positives
tn = 17  # True Negatives
fp = 0  # False Positives
fn = 6  # False Negatives

plot_confusion_matrix(tp, tn, fp, fn, filename='reward_model_confusion_matrix.png')
