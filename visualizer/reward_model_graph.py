import matplotlib.pyplot as plt
import numpy as np

def generate_bar_graph(data, title='Bar Graph', xlabel='Categories', ylabel='Values', filename='bar_graph.png'):
    # Sort the dictionary by values from greatest to least
    sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))
    
    # Extract keys and values from the sorted dictionary
    keys = list(sorted_data.keys())
    values = list(sorted_data.values())
    
    # Create a narrow bar graph
    plt.figure(figsize=(6, 6))  # Narrow figure size
    
    # Generate a gradient of blue colors from darkest to lightest
    colors = plt.cm.Blues(np.linspace(0.8, 0.2, len(keys)))
    
    # Create the bar graph
    bars = plt.bar(keys, values, color=colors, width=0.8)  # Wider bars with less space between them

    # Add titles and labels
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # Add numeric values on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, round(yval, 2), ha='center', va='bottom', fontsize=12)

    # Draw a thin red line at y=50
    plt.axhline(y=50, color='red', linewidth=1, linestyle='--')

    # Label the red line as "baseline"
    plt.text(len(keys) - 0.5, 50 + 1, 'baseline', color='red', fontsize=10, ha='right', va='bottom')

    # Remove horizontal grid lines
    plt.grid(False)

    # Rotate the x labels if necessary
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the graph as an image file
    plt.savefig(filename, format='png', dpi=300)

    # Display the graph
    plt.show()

# Example usage
data = {
    'Zephyr 7B': 74.19,
    'Mistral 7B It': 61.29,
    'Codellama 7B': 58.06,
    'Llama 3.1 8B It': 80.65,
    'Codellama 34B': 70.97,
    'GPT-4o': 85.16,
    'GPT-4o Mini': 92.26
}

generate_bar_graph(data, title='Reward Model Performances', xlabel='Model', ylabel='Accuracy', filename='reward_model_performances.png')
