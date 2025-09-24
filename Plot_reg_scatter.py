"""
@author: ehsan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set global plot parameters
plt.rcParams["font.family"] = "serif" 
plt.rcParams['axes.facecolor'] = 'w'
plt.rc('axes', edgecolor='y')
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'

def load_and_prepare_data():
    """Load and prepare the data for analysis"""
    # Load training data
    simulated_data_path = r"L:\py.pro\Defect\defpicture\comprehensive_features.csv"
    df_simulated = pd.read_csv(simulated_data_path)
    
    # Separate features and targets
    non_feature_cols = ['X', 'Y', 'R', 'target', 'image_path']
    feature_cols = [col for col in df_simulated.columns if col not in non_feature_cols]
    
    X = df_simulated[feature_cols]
    y = df_simulated[['X', 'Y', 'R', 'target']]
    
    return X, y, feature_cols

def main():
    """Main function to perform 5-fold CV and plot results"""
    # Load and prepare data
    X, y, feature_cols = load_and_prepare_data()
    
    # Define best parameters for each target
    best_params = {
        'X': {'bootstrap': True, 'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 100},
        'Y': {'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200},
        'R': {'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 200},
        'target': {'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 2, 'min_samples_split': 10, 'n_estimators': 50}
    }
    
    # Create figure with subplots for each target
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()
    
    # Define target labels for plotting
    target_labels = {
        'X': 'X Position',
        'Y': 'Y Position', 
        'R': 'Radius R',
        'target': 'Target Value'
    }
    
    for i, target in enumerate(['X', 'Y', 'R', 'target']):
        # Split data for this target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y[target], test_size=0.2, random_state=42
        )
        
        # Initialize model with best parameters
        model = RandomForestRegressor(
            n_estimators=best_params[target]['n_estimators'],
            max_depth=best_params[target]['max_depth'],
            min_samples_split=best_params[target]['min_samples_split'],
            min_samples_leaf=best_params[target]['min_samples_leaf'],
            bootstrap=best_params[target]['bootstrap'],
            random_state=42
        )
        
        # Perform 5-fold cross-validation on training data
        kfold = KFold(n_splits=5, random_state=7, shuffle=True)
        train_predicted = cross_val_predict(model, X_train, y_train, cv=kfold)
        
        # Fit model on full training set and predict on test set
        model.fit(X_train, y_train)
        test_predicted = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, train_predicted)
        test_r2 = r2_score(y_test, test_predicted)
        train_mae = mean_absolute_error(y_train, train_predicted)
        test_mae = mean_absolute_error(y_test, test_predicted)
        
        # Determine axis limits based on actual values
        all_actual = np.concatenate([y_train, y_test])
        min_val = min(all_actual)
        max_val = max(all_actual)
        
        # Add a little padding to the axis limits
        padding = (max_val - min_val) * 0.05
        axis_min = min_val - padding
        axis_max = max_val + padding
        
        # Create scatter plot
        ax = axes[i]
        ax.scatter(y_train, train_predicted, edgecolors=(0.7, 1, 0.2), label="Train", alpha=0.7)
        ax.scatter(y_test, test_predicted, edgecolors=(1, 0.1, 0.1), label="Test", alpha=0.7)
        
        # Set limits based on actual values
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        
        # Add labels and title
        ax.set_xlabel(f'Actual {target_labels[target]}', fontsize=26)
        ax.set_ylabel(f'Predicted {target_labels[target]}', fontsize=26)
        ax.set_title(f'{target_labels[target]} Prediction', fontsize=28)
        
        # Add perfect prediction line
        ax.plot([axis_min, axis_max], [axis_min, axis_max], lw=4, color='red', linestyle='--', 
                label=f'Train R²: {train_r2:.3f}, MAE: {train_mae:.3f}\nTest R²: {test_r2:.3f}, MAE: {test_mae:.3f}')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend(prop={'size': 18}, loc='best')
        
        # Set tick parameters
        ax.tick_params(labelsize=24)
    
    plt.tight_layout()
    
    # Save the figure
    save_path = r"L:\py.pro\Defect\experimental_results\actual_vs_predicted_5fold_Gragh2.jpg"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
