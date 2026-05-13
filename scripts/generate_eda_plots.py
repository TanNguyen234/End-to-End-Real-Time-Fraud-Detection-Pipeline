import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots():
    # Load data
    data_path = 'data/creditcard.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Create assets directory
    os.makedirs('assets', exist_ok=True)
    
    # 1. Class Distribution
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    ax = sns.countplot(x='Class', data=df)
    plt.title('Phân phối Class: Normal (0) vs Fraud (1)\n(Trục Y dùng thang đo Logarit)', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.ylabel('Số lượng giao dịch (Log Scale)')
    
    # Add counts
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 10), 
                    textcoords='offset points', fontweight='bold')
    
    plt.savefig('assets/class_distribution.png', bbox_inches='tight', dpi=300)
    print("Saved: assets/class_distribution.png")
    
    # 2. Amount Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Class', y='Amount', data=df)
    plt.title('Phân phối số tiền giao dịch (Amount) theo Class\n(Trục Y dùng thang đo Logarit)', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.ylabel('Amount ($) - Log Scale')
    plt.savefig('assets/amount_distribution.png', bbox_inches='tight', dpi=300)
    print("Saved: assets/amount_distribution.png")
    
    # 3. Time Distribution (Density)
    plt.figure(figsize=(12, 6))
    df_fraud = df[df['Class'] == 1]
    df_normal = df[df['Class'] == 0]
    
    # Convert time to hours
    time_fraud = (df_fraud['Time'] / 3600) % 24
    time_normal = (df_normal['Time'] / 3600) % 24
    
    sns.kdeplot(time_normal, fill=True, label='Normal')
    sns.kdeplot(time_fraud, fill=True, label='Fraud')
    plt.title('Mật độ giao dịch theo Giờ trong ngày (Hour)', fontsize=14, fontweight='bold')
    plt.xlabel('Giờ trong ngày (0 - 24h)')
    plt.ylabel('Mật độ (Density)')
    plt.legend()
    plt.savefig('assets/time_distribution.png', bbox_inches='tight', dpi=300)
    print("Saved: assets/time_distribution.png")
    
    # 4. Feature Correlation (Top features)
    # V14 and V17 are known to be highly correlated with Class
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    sns.kdeplot(df_normal['V14'], fill=True, label='Normal')
    sns.kdeplot(df_fraud['V14'], fill=True, label='Fraud')
    plt.title('Phân phối đặc trưng V14', fontsize=14, fontweight='bold')
    plt.xlabel('Giá trị V14')
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(df_normal['V17'], fill=True, label='Normal')
    sns.kdeplot(df_fraud['V17'], fill=True, label='Fraud')
    plt.title('Phân phối đặc trưng V17', fontsize=14, fontweight='bold')
    plt.xlabel('Giá trị V17')
    
    plt.savefig('assets/feature_distributions.png', bbox_inches='tight', dpi=300)
    print("Saved: assets/feature_distributions.png")

if __name__ == "__main__":
    generate_plots()
