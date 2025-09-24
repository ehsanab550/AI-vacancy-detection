"""
@author: ehsan
"""
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib as mpl

# Set global plot parameters with Times New Roman font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.facecolor'] = 'w'
plt.rc('axes', edgecolor='y')
plt.minorticks_on()
plt.grid(which='minor', color='y', linestyle='--', alpha=0.1)
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

def plot_custom_shap(shap_values, X_data, top_features, highlight_features=None, 
                     cmap='jet', highlight_color='darkred', bg_color='white'):
    """
    Custom SHAP plot with actual feature names ordered by importance
    """
    num_features_to_show = len(top_features)
    
    # Create a new DataFrame with only the top features in importance order
    X_top = X_data[top_features]
    
    # Reorder SHAP values to match top features importance order
    feature_indices = [list(X_data.columns).index(f) for f in top_features]
    shap_top = shap_values[:, feature_indices]
    
    # Create figure with appropriate size
    plt.figure(figsize=(15, 12))
    
    # Create the SHAP plot
    shap.summary_plot(
        shap_top, 
        X_top, 
        show=False,
        max_display=num_features_to_show,
        cmap=cmap,
        plot_size=None,
        feature_names=top_features  # Ensure correct names
    )
    
    ax = plt.gca()
    ax.set_facecolor(bg_color)
    
    # Highlight specified features if provided
    if highlight_features:
        ytick_labels = ax.get_yticklabels()
        present_highlight = []
        
        for label in ytick_labels:
            text = label.get_text()
            if text in highlight_features:
                label.set_color(highlight_color)
                label.set_fontweight('bold')
                label.set_fontstyle('italic')
                label.set_fontsize(34)  # Increased to 34
                present_highlight.append(text)
        
        # Check for missing features
        missing = set(highlight_features) - set(present_highlight)
        if missing:
            print(f"Note: These features not in top {num_features_to_show}: {', '.join(missing)}")
            print("Consider increasing the number of features shown")

    # Apply custom styling to axes with increased font sizes
    ax.tick_params(axis='x', colors='black', labelsize=30)  # Increased to 30
    ax.tick_params(axis='y', colors='black', labelsize=30)  # Increased to 30
    
    # Set font for all text elements in the axes
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontname('Times New Roman')
        item.set_color('black')
    
    # Grid customization
    plt.grid(True, which='major', axis='y', 
             linestyle='--', color='lightgray', alpha=0.7)
    
    # Border styling
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Title and labels with Times New Roman font and increased sizes
    plt.title(f"Top {num_features_to_show} Feature Impacts", 
             fontsize=36, pad=15, color='black', fontname='Times New Roman')  # Increased to 36
    plt.xlabel("SHAP Value Impact", fontsize=34, color='black', fontname='Times New Roman')  # Increased to 34
    
    # Find and modify the colorbar - this is the key fix
    for child in ax.get_children():
        if isinstance(child, mpl.colorbar.Colorbar):
            # Set colorbar label font with increased size
            child.set_label('Feature Value', fontsize=34, fontname='Times New Roman', color='black')  # Increased to 34
            
            # Force the colorbar to use our font settings
            child.ax.tick_params(labelsize=38)  # Increased to 38
            
            # Set custom ticks and labels
            child.set_ticks([0, 1])
            child.set_ticklabels(['Low', 'High'])
            
            # Apply font properties to tick labels
            for label in child.ax.get_yticklabels():
                label.set_fontname('Times New Roman')
                label.set_fontsize(38)  # Increased to 38
                label.set_color('black')
            break
    
    plt.tight_layout()
    return ax

def export_shap_analysis(shap_values, feature_names, target_name, output_dir):
    """
    Export detailed SHAP analysis including sorted feature importance and impact descriptions
    """
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(0)
    
    # Calculate mean SHAP values (direction of impact)
    mean_shap = shap_values.mean(0)
    
    # Create a DataFrame with the results
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'mean_shap': mean_shap,
        'impact_direction': np.where(mean_shap > 0, 'Positive', 'Negative')
    })
    
    # Sort by importance
    shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
    
    # Add impact description
    shap_df['impact_description'] = shap_df.apply(
        lambda row: f"This feature has a {row['impact_direction'].lower()} impact on {target_name} prediction. " +
                   f"Higer values {'increase' if row['mean_shap'] > 0 else 'decrease'} the predicted {target_name}.",
        axis=1
    )
    
    # Save to CSV
    csv_path = output_dir / f"shap_analysis_{target_name}.csv"
    shap_df.to_csv(csv_path, index=False)
    
    # Print summary
    print(f"\nSHAP Analysis for {target_name}:")
    print("=" * 50)
    for i, row in shap_df.iterrows():
        print(f"{i+1}. {row['feature']}:")
        print(f"   Mean |SHAP|: {row['mean_abs_shap']:.4f}")
        print(f"   Mean SHAP: {row['mean_shap']:.4f} ({row['impact_direction']})")
        print(f"   Impact: {row['impact_description']}")
        print()
    
    return shap_df

def create_gradient_bar_plot(importance_df, target, output_dir, num_features=10):
    """
    Create a horizontal bar plot with gradient colors from black (top) to light gray (bottom)
    """
    # Select top features
    top_features_df = importance_df.head(num_features)
    
    # Reverse the order so highest importance is at the top
    top_features_df = top_features_df.iloc[::-1]
    
    # Create gradient colors from black to light gray
    # Using a linear gradient from black (0,0,0) to light gray (0.8,0.8,0.8)
    gradient_colors = []
    n_features = len(top_features_df)
    
    for i in range(n_features):
        # Calculate gradient intensity (0 = black, 0.8 = light gray)
        intensity = 0.8 * (i / (n_features - 1)) if n_features > 1 else 0.4
        gradient_colors.append((intensity, intensity, intensity))
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Create horizontal bar plot with gradient colors
    bars = plt.barh(range(len(top_features_df)), 
                    top_features_df['importance'], 
                    color=gradient_colors,
                    edgecolor='black',
                    linewidth=0.5,
                    height=0.8)
    
    # Set y-axis ticks and labels
    plt.yticks(range(len(top_features_df)), top_features_df['feature'])
    
    # Styling
    plt.title(f'Feature Importance (Mean |SHAP|) for {target}', 
             fontsize=36, fontname='Times New Roman', color='black')
    plt.xlabel('Mean |SHAP value|', fontsize=34, fontname='Times New Roman', color='black')
    
    # Grid customization
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5, axis='x', alpha=0.7)
    plt.gca().set_facecolor('white')
    
    # Set Times New Roman font for all text elements
    for item in ([plt.gca().title, plt.gca().xaxis.label, plt.gca().yaxis.label] +
                 plt.gca().get_xticklabels() + plt.gca().get_yticklabels()):
        item.set_fontname('Times New Roman')
        item.set_color('black')
        if item == plt.gca().title:
            item.set_fontsize(36)
        else:
            item.set_fontsize(32)
    
    # Modify spines
    for spine in plt.gca().spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.5)
    
    # Ensure proper layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir / f"shap_bar_plot_{target}.png", dpi=600, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.show()

def shap_analysis(models_dir, X_train, feature_cols, output_dir, highlight_features=None):
    """
    Perform SHAP analysis for all trained models with custom styling.
    """
    # Load models and feature indices
    feature_indices = joblib.load(models_dir / "feature_indices.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")
    
    # Scale the features
    X_train_scaled = scaler.transform(X_train)
    X_train_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
    
    # Define targets
    targets = ['X', 'Y', 'R', 'target']
    
    for target in targets:
        # Load model
        model = joblib.load(models_dir / f"{target}_model.pkl")
        
        # Get feature indices for this target
        key_map = {
            'X': 'top_features_x',
            'Y': 'top_features_y', 
            'R': 'top_features_r',
            'target': 'top_features_target'
        }
        feature_idx = feature_indices[key_map[target]]
        
        # Get feature names for the subset
        feature_names_subset = [feature_cols[i] for i in feature_idx]
        
        # Create a subset of the training data
        X_train_subset = X_train_scaled[:, feature_idx]
        X_train_subset_df = pd.DataFrame(X_train_subset, columns=feature_names_subset)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values (using a sample for efficiency)
        sample_size = min(100, X_train_subset_df.shape[0])
        X_sample = X_train_subset_df.iloc[:sample_size, :]
        shap_values = explainer.shap_values(X_sample)
        
        # Get the top features based on mean absolute SHAP values
        mean_shap_values = np.abs(shap_values).mean(0)
        importance_df = pd.DataFrame({
            'feature': feature_names_subset,
            'importance': mean_shap_values
        }).sort_values('importance', ascending=False)  # Sort descending for gradient
        
        top_features = importance_df.head(10)['feature'].tolist()
        
        # Create the custom SHAP plot
        ax = plot_custom_shap(
            shap_values,
            X_sample,
            top_features=top_features,
            highlight_features=highlight_features,
            cmap='jet',
            highlight_color='darkred',
            bg_color='white'
        )
        
        # Save the plot
        plt.savefig(output_dir / f"shap_custom_plot_{target}.jpg", dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        # Export detailed SHAP analysis
        export_shap_analysis(shap_values, feature_names_subset, target, output_dir)
        
        # Create the new gradient bar plot
        create_gradient_bar_plot(importance_df, target, output_dir, num_features=10)

def main():
    # Set paths
    output_dir = Path(r"L:\py.pro\Defect\experimental_results")
    models_dir = output_dir / "models"
    
    # Define features to highlight (modify this list as needed)
    physics_informed_features = []  # Add your specific feature names here
    
    # Load training data
    simulated_data_path = r"L:\py.pro\Defect\defpicture\comprehensive_features.csv"
    df_simulated = pd.read_csv(simulated_data_path)
    
    # Separate features and targets
    non_feature_cols = ['X', 'Y', 'R', 'target', 'image_path']
    feature_cols = [col for col in df_simulated.columns if col not in non_feature_cols]
    
    X_train = df_simulated[feature_cols]
    
    # Perform SHAP analysis
    print("Performing SHAP analysis with custom styling...")
    shap_analysis(models_dir, X_train, feature_cols, output_dir, physics_informed_features)
    print("SHAP analysis completed!")

if __name__ == "__main__":
    main()
