import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os

def visualize_anomaly_scores(csv_path, output_dir="results/visualization"):
    """
    Gabungkan histogram distribusi skor dan ROC curve dalam satu figure
    """
    # Load data dan ekstrak nama kategori dengan benar
    filename = os.path.basename(csv_path)  # Contoh: 'mvtec_bottle_image_scores.csv'
    category = filename.split('_')[1]     # Ambil bagian kedua ('bottle')
    
    df = pd.read_csv(csv_path)
    
    # Pisahkan skor normal dan anomaly
    normal_scores = df[df['true_label'] == 0]['anomaly_score']
    anomaly_scores = df[df['true_label'] == 1]['anomaly_score']
    
    # Hitung ROC curve
    fpr, tpr, thresholds = roc_curve(df['true_label'], df['anomaly_score'])
    roc_auc = auc(fpr, tpr)
    optimal_idx = (tpr - fpr).argmax()
    optimal_threshold = thresholds[optimal_idx]

    # Buat figure dengan 2 subplot
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Histogram Distribusi Skor
    plt.subplot(1, 2, 1)
    sns.histplot(normal_scores, label='Normal', color='green', kde=True, alpha=0.5, bins=30)
    sns.histplot(anomaly_scores, label='Anomaly', color='red', kde=True, alpha=0.5, bins=30)
    plt.axvline(x=optimal_threshold, color='blue', linestyle='--', 
                label=f'Threshold: {optimal_threshold:.4f}')
    plt.title(f"[{category}] Score Distribution\n(AUC={roc_auc:.3f})")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.legend()

    # Subplot 2: ROC Curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', 
                label=f'Optimal Threshold: {optimal_threshold:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Simpan gambar dengan nama yang benar
    output_path = os.path.join(output_dir, f"{category}_combined_plot.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_path}")

def visualize_all_datasets():
    scores_dir = "results/scores"
    for file in os.listdir(scores_dir):
        if file.endswith("_image_scores.csv"):
            csv_path = os.path.join(scores_dir, file)
            try:
                visualize_anomaly_scores(csv_path)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    visualize_all_datasets()