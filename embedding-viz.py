import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel
import torch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import os

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def plot_embeddings(word_list, save_plots, output_dir, show_animation):
    # Tokenize the words and get embeddings
    inputs = tokenizer(word_list, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=2, perplexity=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Reduce dimensions using t-SNE for 3D plot
    tsne_3d = TSNE(n_components=3, perplexity=2, random_state=42)
    embeddings_3d = tsne_3d.fit_transform(embeddings)

    # Create output directory if saving plots
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize the plot with more space between subplots
    fig, ax2d = plt.subplots(figsize=(10, 10))

    # Plot the 2D t-SNE embeddings
    palette = sns.color_palette("husl", len(word_list))
    for i, (word, color) in enumerate(zip(word_list, palette)):
        ax2d.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], color=color, label=word)
        ax2d.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=14, color=color)

    ax2d.set_title("2D Visualization of BERT Embeddings using t-SNE")
    ax2d.set_xlabel("t-SNE component 1")
    ax2d.set_ylabel("t-SNE component 2")
    ax2d.legend(loc='best')
    plt.tight_layout()

    if save_plots:
        plt.savefig(os.path.join(output_dir, 'tsne_2d.png'))

    # Create the table showing part of the embeddings
    fig, ax_table = plt.subplots(figsize=(10, 4))
    ax_table.axis('off')
    table_data = np.round(embeddings[:, :7], 2)  # Example selection of 7 dimensions
    table = ax_table.table(cellText=table_data, rowLabels=word_list, colLabels=[f'dim_{i+1}' for i in range(table_data.shape[1])], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.0)

    # Color the row labels
    for i, (word, color) in enumerate(zip(word_list, palette)):
        table[(i + 1, -1)].set_text_props(color=color, fontweight='bold')

    if save_plots:
        plt.savefig(os.path.join(output_dir, 'embedding_table.png'))

    # Plot the 3D t-SNE embeddings
    fig = plt.figure(figsize=(10, 10))
    ax3d = fig.add_subplot(111, projection='3d')

    def update(angle):
        ax3d.view_init(30, angle)
        return fig,

    for i, (word, color) in enumerate(zip(word_list, palette)):
        ax3d.scatter(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], color=color, label=word)
        ax3d.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], word, color=color)

    ax3d.set_title("3D Visualization of BERT Embeddings using t-SNE")
    ax3d.set_xlabel("t-SNE component 1")
    ax3d.set_ylabel("t-SNE component 2")
    ax3d.set_zlabel("t-SNE component 3")

    if show_animation:
        ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50)

    if save_plots:
        plt.savefig(os.path.join(output_dir, 'tsne_3d.png'))
        if show_animation:
            try:
                ani.save(os.path.join(output_dir, 'rotation_animation.mp4'), writer='ffmpeg', fps=30)
            except Exception as e:
                print(f"Warning: Could not save animation as MP4. {e}")
    
    if not save_plots: 
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize BERT embeddings with t-SNE.')
    parser.add_argument('--list', nargs='+', default=["king", "kitten", "men", "houses", "peach", "apple", "woman", "cat", "queen", "fox", "sleep", "lazy", "jumps", "dog", "banana", "fish", "water", "spider", "castle", "fly", "web"], help='List of words to visualize.')
    parser.add_argument('--save', type=str, help='Directory to save the plots and table.')
    parser.add_argument('--ani', action='store_true', help='Show the 3D animation.')

    args = parser.parse_args()
    word_list = args.list
    save_plots = args.save is not None
    output_dir = args.save if save_plots else 'plots'
    show_animation = args.ani

    plot_embeddings(word_list, save_plots, output_dir, show_animation)
