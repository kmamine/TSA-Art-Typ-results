import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import tqdm with fallback for script/notebook environments
try:
    from tqdm.autonotebook import tqdm
except:
    from tqdm import tqdm

# Try to import seaborn for enhanced visualizations
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Try to import statsmodels for power analysis
try:
    from statsmodels.stats.power import TTestPower
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Note: Install statsmodels for power analysis: pip install statsmodels")

# ==================== METRIC CALCULATION FUNCTIONS ====================

def binarize_fixation_map(fix_map, threshold=0.5):
    """Convert continuous fixation map to binary"""
    if np.max(fix_map) == 0:
        return np.zeros_like(fix_map)
    binary_map = np.zeros_like(fix_map)
    binary_map[fix_map > threshold * np.max(fix_map)] = 1.0
    return binary_map

def safe_normalize_map(s_map):
    """Safe normalization with error handling"""
    if s_map.max() == s_map.min():
        return np.ones_like(s_map) / s_map.size
    norm_s_map = (s_map - np.min(s_map)) / ((np.max(s_map) - np.min(s_map)) + 1e-10)
    return norm_s_map

def kldiv(p_map, gt_map):
    """Kullback-Leibler Divergence"""
    eps = 1e-10
    p_map = p_map.astype(np.float64) + eps
    gt_map = gt_map.astype(np.float64) + eps
    
    # Normalize to probability distributions
    p_map = p_map / (np.sum(p_map) + eps)
    gt_map = gt_map / (np.sum(gt_map) + eps)
    
    # Calculate KL divergence
    kl_div = np.sum(gt_map * np.log2(eps + gt_map / (p_map + eps)))
    return kl_div

def cc(p_map, gt_map):
    """Pearson's Correlation Coefficient"""
    p_map = p_map.flatten()
    gt_map = gt_map.flatten()
    
    p_map = (p_map - np.mean(p_map)) / (np.std(p_map) + 1e-10)
    gt_map = (gt_map - np.mean(gt_map)) / (np.std(gt_map) + 1e-10)
    
    correlation = np.corrcoef(p_map, gt_map)[0, 1]
    return correlation

def similarity(p_map, gt_map):
    """Similarity (Histogram Intersection)"""
    eps = 1e-10
    p_map = p_map.astype(np.float64) + eps
    gt_map = gt_map.astype(np.float64) + eps
    
    # Normalize to probability distributions
    p_map = p_map / (np.sum(p_map) + eps)
    gt_map = gt_map / (np.sum(gt_map) + eps)
    
    # Calculate similarity (intersection)
    sim = np.sum(np.minimum(p_map, gt_map))
    return sim

def safe_auc_judd(s_map, gt_fix):
    """Safe AUC-JUDD implementation with binary fixation map"""
    try:
        # Ensure gt is binary
        gt_fix = binarize_fixation_map(gt_fix)
        if np.sum(gt_fix) == 0:
            return np.nan
            
        # Normalize saliency map
        s_map = safe_normalize_map(s_map)
        
        # Get thresholds from fixation points
        thresholds = []
        for i in range(gt_fix.shape[0]):
            for k in range(gt_fix.shape[1]):
                if gt_fix[i][k] > 0:
                    thresholds.append(s_map[i][k])
        
        if not thresholds:
            return np.nan
            
        num_fixations = np.sum(gt_fix)
        thresholds = sorted(set(thresholds))
        
        area = [(0.0, 0.0)]
        for thresh in thresholds:
            temp = np.zeros(s_map.shape)
            temp[s_map >= thresh] = 1.0
            
            num_overlap = np.where(np.add(temp, gt_fix) == 2)[0].shape[0]
            tp = num_overlap / (num_fixations * 1.0)
            fp = (np.sum(temp) - num_overlap) / ((gt_fix.shape[0] * gt_fix.shape[1]) - num_fixations + 1e-10)
            
            area.append((tp, fp))
        
        area.append((1.0, 1.0))
        area.sort(key=lambda x: x[0])
        tp_list = [x[0] for x in area]
        fp_list = [x[1] for x in area]
        
        return np.trapz(tp_list, fp_list)
    except Exception as e:
        print(f"Error in AUC: {e}")
        return np.nan

def safe_nss(s_map, gt_fix):
    """Safe NSS calculation with binary fixation map"""
    try:
        # Ensure gt is binary
        gt_fix = binarize_fixation_map(gt_fix)
        if np.sum(gt_fix) == 0:
            return np.nan
            
        # Normalize and standardize saliency map
        s_map_norm = (s_map - np.mean(s_map)) / (np.std(s_map) + 1e-10)
        
        # Get values at fixation points
        x, y = np.where(gt_fix == 1)
        if len(x) == 0:
            return np.nan
            
        temp = []
        for i in zip(x, y):
            temp.append(s_map_norm[i[0], i[1]])
        
        return np.mean(temp)
    except Exception as e:
        print(f"Error in NSS: {e}")
        return np.nan

def safe_infogain(s_map, gt_fix, baseline_map):
    """Safe information gain calculation with binary fixation map"""
    try:
        # Ensure gt is binary
        gt_fix = binarize_fixation_map(gt_fix)
        if np.sum(gt_fix) == 0:
            return np.nan
            
        eps = 2.2204e-16
        
        # Add small constant to avoid zeros in baseline
        baseline_map = baseline_map + eps
        
        # Normalize to probability distributions
        s_map = safe_normalize_map(s_map)
        baseline_map = safe_normalize_map(baseline_map)
        
        # Convert to probability distributions
        s_map = s_map / (np.sum(s_map) + eps)
        baseline_map = baseline_map / (np.sum(baseline_map) + eps)
        
        # Calculate info gain at fixation points
        temp = []
        x, y = np.where(gt_fix == 1)
        for i in zip(x, y):
            s_val = s_map[i[0], i[1]]
            b_val = baseline_map[i[0], i[1]]
            # Now b_val should never be zero due to eps addition
            ig = np.log2(eps + s_val) - np.log2(eps + b_val)
            temp.append(ig)
        
        return np.mean(temp) if temp else np.nan
    except Exception as e:
        print(f"Error in InfoGain: {e}")
        return np.nan

def safe_calc_metrics(pred_sal, gt_sal, pred_fix=None, gt_fix=None):
    """Calculate all metrics between predicted and ground truth maps"""
    metrics = {
        'KLDiv': np.nan, 
        'CC': np.nan,
        'SIM': np.nan, 
        'AuC': np.nan, 
        'NSS': np.nan, 
        'InfoGain': np.nan
    }
    
    try:
        # Ensure maps are float and properly sized
        pred_sal = pred_sal.astype(np.float64)
        gt_sal = gt_sal.astype(np.float64)
        
        # Use fixation maps if provided, otherwise use saliency maps
        if gt_fix is not None:
            gt_fix_used = binarize_fixation_map(gt_fix.astype(np.float64))
        else:
            gt_fix_used = binarize_fixation_map(gt_sal)
            
        if pred_fix is not None:
            pred_fix_used = binarize_fixation_map(pred_fix.astype(np.float64))
        else:
            pred_fix_used = binarize_fixation_map(pred_sal)
        
        # Calculate metrics
        metrics['KLDiv'] = kldiv(pred_sal, gt_sal)
        metrics['CC'] = cc(pred_sal, gt_sal)
        metrics['SIM'] = similarity(pred_sal, gt_sal)
        metrics['AuC'] = safe_auc_judd(pred_sal, gt_fix_used)
        metrics['NSS'] = safe_nss(pred_sal, gt_fix_used)
        metrics['InfoGain'] = safe_infogain(pred_sal, gt_fix_used, gt_sal)
        
    except Exception as e:
        print(f"Error in metrics calculation: {e}")
    
    return metrics

# ==================== DATA LOADING FUNCTIONS ====================

def find_image_file(folder_path):
    """Find image file in folder, excluding notebooks"""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Look for image files
    image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.tif']
    files = [f for f in os.listdir(folder_path) 
             if not f.startswith('.') and any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not files:
        # Try to find any file that's not a directory
        files = [f for f in os.listdir(folder_path) 
                 if not f.startswith('.') and os.path.isfile(os.path.join(folder_path, f))]
        
    if not files:
        raise FileNotFoundError(f"No image files found in {folder_path}")
    
    # Sort to ensure consistent ordering
    files.sort()
    return os.path.join(folder_path, files[0])

def load_and_preprocess_image(file_path, target_size=(224, 224)):
    """Load and preprocess image with proper normalization"""
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    if img is None:
        # Try loading with different methods
        img = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR)
        if img is None:
            raise ValueError(f"Could not load image: {file_path}")
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize if needed
    if img.shape != target_size:
        img = cv2.resize(img, target_size)
    
    # Convert to float
    img = img.astype(np.float64)
    
    # Normalize
    img = safe_normalize_map(img)
    
    return img

# ==================== COMPARISON FUNCTIONS ====================

def compare_groups(pred_group, gt_group, stimuli_names, salmap_folder, fixation_folder):
    """Compare one group's predictions against another group's ground truth"""
    metrics_names = ['KLDiv', 'CC', 'SIM', 'AuC', 'NSS', 'InfoGain']
    results = pd.DataFrame(columns=metrics_names)
    
    for stimulus_name in tqdm(stimuli_names, desc=f"{pred_group}→{gt_group}", unit="stim"):
        try:
            # Load predictor group maps
            pred_fix_path = find_image_file(os.path.join(fixation_folder, pred_group, stimulus_name))
            pred_sal_path = find_image_file(os.path.join(salmap_folder, pred_group, stimulus_name))
            
            # Load ground truth group maps
            gt_fix_path = find_image_file(os.path.join(fixation_folder, gt_group, stimulus_name))
            gt_sal_path = find_image_file(os.path.join(salmap_folder, gt_group, stimulus_name))
            
            # Load and preprocess images
            pred_fix = load_and_preprocess_image(pred_fix_path)
            pred_sal = load_and_preprocess_image(pred_sal_path)
            gt_fix = load_and_preprocess_image(gt_fix_path)
            gt_sal = load_and_preprocess_image(gt_sal_path)
            
            # Calculate metrics
            metrics_row = safe_calc_metrics(
                pred_sal=pred_sal, 
                gt_sal=gt_sal,
                pred_fix=pred_fix,
                gt_fix=gt_fix
            )
            
            results = pd.concat([results, pd.DataFrame([metrics_row])], ignore_index=True)
            
        except Exception as e:
            print(f"\nError processing {stimulus_name}: {e}")
            # Add row of NaNs to maintain alignment
            results = pd.concat([results, pd.DataFrame([{k: np.nan for k in metrics_names}])], ignore_index=True)
            continue
    
    return results

def run_all_comparisons():
    """Run all group comparisons"""
    # Define paths and groups
    salmap_folder = "./salmaps/" 
    fixation_folder = "./fix_maps/"
    stimuli_folder = "./peintures/"
    
    groups = {
        'art': "artiste",
        'typ': "neurotypique", 
        'tsa': "TSA"
    }
    
    # Get stimuli names
    stimuli_names = [f for f in os.listdir(stimuli_folder) if f.lower().endswith('.jpeg')]
    if not stimuli_names:
        stimuli_names = [f for f in os.listdir(stimuli_folder) if f.lower().endswith(('.jpg', '.png'))]
    
    print(f"Found {len(stimuli_names)} stimuli")
    
    # Run comparisons
    print("\n" + "="*60)
    print("Running comparisons...")
    print("="*60)
    
    # TSA → Art
    print("\n1. TSA predicting Artist fixations:")
    tsa_art = compare_groups(groups['tsa'], groups['art'], stimuli_names, salmap_folder, fixation_folder)
    
    # TSA → Typ
    print("\n2. TSA predicting Neurotypique fixations:")
    tsa_typ = compare_groups(groups['tsa'], groups['typ'], stimuli_names, salmap_folder, fixation_folder)
    
    # Typ → Art
    print("\n3. Neurotypique predicting Artist fixations:")
    typ_art = compare_groups(groups['typ'], groups['art'], stimuli_names, salmap_folder, fixation_folder)
    
    # Art → TSA (reverse)
    print("\n4. Artist predicting TSA fixations (reverse):")
    art_tsa = compare_groups(groups['art'], groups['tsa'], stimuli_names, salmap_folder, fixation_folder)
    
    # Typ → TSA (reverse)
    print("\n5. Neurotypique predicting TSA fixations (reverse):")
    typ_tsa = compare_groups(groups['typ'], groups['tsa'], stimuli_names, salmap_folder, fixation_folder)

    # Art → Typ
    print("\n6. Artist predicting Neurotypique fixations:")
    art_typ = compare_groups(groups['art'], groups['typ'], stimuli_names, salmap_folder, fixation_folder)
    
    # Save results
    tsa_art.to_pickle("metrics_tsa_art.pickle")
    tsa_typ.to_pickle("metrics_tsa_typ.pickle") 
    typ_art.to_pickle("metrics_typ_art.pickle")
    art_tsa.to_pickle("metrics_art_tsa.pickle")
    typ_tsa.to_pickle("metrics_typ_tsa.pickle")
    art_typ.to_pickle("metrics_art_typ.pickle")    
    return {
        'tsa_art': tsa_art,
        'tsa_typ': tsa_typ,
        'typ_art': typ_art,
        'art_tsa': art_tsa,
        'typ_tsa': typ_tsa,
        'art_typ': art_typ
    }

# ==================== ENHANCED DESCRIPTIVE STATISTICS ====================

def generate_detailed_statistics(results_dict):
    """Generate comprehensive descriptive statistics for all metrics"""
    print("\n" + "="*80)
    print("DETAILED DESCRIPTIVE STATISTICS")
    print("="*80)
    
    all_stats = {}
    
    for comp_name, df in results_dict.items():
        print(f"\n{comp_name.upper()} - Complete Statistics:")
        print("-"*60)
        
        # Use pandas describe for comprehensive statistics
        desc_stats = df.describe(percentiles=[.25, .5, .75])
        
        # Add additional statistics
        for metric in df.columns:
            if metric in df.columns and df[metric].notna().any():
                # Get data
                data = df[metric].dropna()
                
                # Calculate additional statistics
                stats_dict = {
                    'count': len(data),
                    'mean': np.mean(data),
                    'std': np.std(data, ddof=1),
                    'min': np.min(data),
                    '25%': np.percentile(data, 25),
                    'median': np.median(data),
                    '75%': np.percentile(data, 75),
                    'max': np.max(data),
                    'range': np.max(data) - np.min(data),
                    'IQR': np.percentile(data, 75) - np.percentile(data, 25),
                    'skewness': stats.skew(data) if len(data) > 2 else np.nan,
                    'kurtosis': stats.kurtosis(data) if len(data) > 3 else np.nan,
                    'CV': (np.std(data, ddof=1) / np.mean(data) * 100) if np.mean(data) != 0 else np.nan,
                    'missing': df[metric].isna().sum()
                }
                
                # Print formatted statistics
                print(f"\n{metric}:")
                print(f"  Count: {stats_dict['count']} (Missing: {stats_dict['missing']})")
                print(f"  Mean ± SD: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
                print(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
                print(f"  Quartiles: Q1={stats_dict['25%']:.3f}, Median={stats_dict['median']:.3f}, Q3={stats_dict['75%']:.3f}")
                print(f"  IQR: {stats_dict['IQR']:.3f}")
                if not np.isnan(stats_dict['skewness']):
                    print(f"  Skewness: {stats_dict['skewness']:.3f} ", end="")
                    if abs(stats_dict['skewness']) > 1:
                        print("(Highly skewed)")
                    elif abs(stats_dict['skewness']) > 0.5:
                        print("(Moderately skewed)")
                    else:
                        print("(Approx symmetric)")
                if not np.isnan(stats_dict['kurtosis']):
                    print(f"  Kurtosis: {stats_dict['kurtosis']:.3f}")
                if not np.isnan(stats_dict['CV']):
                    print(f"  CV: {stats_dict['CV']:.1f}%")
                
                # Store for later use
                if comp_name not in all_stats:
                    all_stats[comp_name] = {}
                all_stats[comp_name][metric] = stats_dict
    
    return all_stats

# ==================== DISTRIBUTION VISUALIZATIONS ====================

def create_distribution_histograms(results_dict):
    """Create histograms showing the distribution of each metric for each comparison"""
    print("\n" + "="*80)
    print("DISTRIBUTION HISTOGRAMS")
    print("="*80)
    
    metrics = ['AuC', 'NSS', 'SIM', 'CC', 'KLDiv', 'InfoGain']
    comparisons = list(results_dict.keys())
    
    # Create a separate figure for each metric
    for metric in metrics:
        print(f"\nGenerating histograms for {metric}...")
        
        # Create figure with subplots for each comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Set color palette
        colors = ['blue', 'lightblue', 'green', 'orange', 'yellow']
        
        for idx, (comp_name, color) in enumerate(zip(comparisons[:6], colors)):
            ax = axes[idx]
            
            if comp_name in results_dict and metric in results_dict[comp_name].columns:
                data = results_dict[comp_name][metric].dropna().values
                
                if len(data) > 0:
                    # Create histogram
                    if SEABORN_AVAILABLE:
                        sns.histplot(data, ax=ax, color=color, kde=True, bins=15, alpha=0.6)
                        # Add rug plot
                        sns.rugplot(data, ax=ax, color='black', alpha=0.5)
                    else:
                        ax.hist(data, bins=15, color=color, alpha=0.6, edgecolor='black')
                        # Add density curve
                        from scipy.stats import gaussian_kde
                        try:
                            kde = gaussian_kde(data)
                            x_range = np.linspace(min(data), max(data), 100)
                            ax.plot(x_range, kde(x_range) * len(data) * (max(data)-min(data))/15, 
                                   color='darkred', linewidth=2)
                        except:
                            pass
                    
                    # Add vertical lines for statistics
                    mean_val = np.mean(data)
                    median_val = np.median(data)
                    
                    ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, 
                              label=f'Mean: {mean_val:.3f}')
                    ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                              label=f'Median: {median_val:.3f}')
                    
                    # Add shaded area for IQR
                    q1 = np.percentile(data, 25)
                    q3 = np.percentile(data, 75)
                    ax.axvspan(q1, q3, alpha=0.2, color='gray', label=f'IQR: [{q1:.3f}, {q3:.3f}]')
                    
                    # Set title and labels
                    ax.set_title(f'{comp_name}\nn={len(data)}', fontsize=12, fontweight='bold')
                    ax.set_xlabel(metric)
                    ax.set_ylabel('Frequency')
                    ax.legend(fontsize=9)
                    ax.grid(True, alpha=0.3)
                    
                    # Add text box with statistics
                    stats_text = (f"Mean: {mean_val:.3f}\n"
                                 f"SD: {np.std(data):.3f}\n"
                                 f"Min: {np.min(data):.3f}\n"
                                 f"Max: {np.max(data):.3f}")
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Hide empty subplots
        for idx in range(len(comparisons), 6):
            axes[idx].axis('off')
        
        plt.suptitle(f'Distribution of {metric} Across Different Comparisons', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'distribution_{metric}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Create combined histogram for each comparison (all metrics together)
    print("\nGenerating combined histograms for each comparison...")
    
    for comp_name in comparisons:
        print(f"\nCreating combined histogram for {comp_name}...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            if comp_name in results_dict and metric in results_dict[comp_name].columns:
                data = results_dict[comp_name][metric].dropna().values
                
                if len(data) > 0:
                    # Create histogram
                    if SEABORN_AVAILABLE:
                        sns.histplot(data, ax=ax, color='skyblue', kde=True, bins=12, alpha=0.7)
                    else:
                        ax.hist(data, bins=12, color='skyblue', alpha=0.7, edgecolor='black')
                    
                    # Add statistics
                    ax.axvline(np.mean(data), color='red', linestyle='-', linewidth=2, 
                              label=f'Mean: {np.mean(data):.3f}')
                    
                    # Set title and labels
                    ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Value')
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Distribution of All Metrics for {comp_name}\n(n={len(results_dict[comp_name])})', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'combined_distribution_{comp_name}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("✓ All distribution histograms generated and saved as PNG files")

# ==================== ADDITIONAL DISTRIBUTION ANALYSES ====================

def create_qq_plots(results_dict):
    """Create Q-Q plots to check normality of distributions"""
    print("\n" + "="*80)
    print("NORMALITY CHECK: Q-Q PLOTS")
    print("="*80)
    
    metrics = ['AuC', 'NSS', 'SIM', 'CC']
    comparisons = list(results_dict.keys())
    
    # Create Q-Q plots for key metrics
    for metric in metrics:
        print(f"\nGenerating Q-Q plots for {metric}...")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, comp_name in enumerate(comparisons[:6]):
            ax = axes[idx]
            
            if comp_name in results_dict and metric in results_dict[comp_name].columns:
                data = results_dict[comp_name][metric].dropna().values
                
                if len(data) > 3:
                    # Create Q-Q plot
                    stats.probplot(data, dist="norm", plot=ax)
                    
                    # Add R² value
                    from scipy.stats import pearsonr
                    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
                    r_value, _ = pearsonr(theoretical_quantiles, np.sort(data))
                    
                    # Set title and labels
                    ax.set_title(f'{comp_name}\nR² = {r_value**2:.3f}', fontsize=11)
                    ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Q-Q Plots for {metric} (Checking Normality)', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'qq_plot_{metric}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("✓ Q-Q plots generated for normality assessment")

def create_density_plots(results_dict):
    """Create density plots comparing distributions across comparisons"""
    print("\n" + "="*80)
    print("DENSITY PLOT COMPARISONS")
    print("="*80)
    
    metrics = ['AuC', 'NSS', 'SIM']
    
    for metric in metrics:
        print(f"\nCreating density plot for {metric}...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = {'tsa_art': 'blue', 'tsa_typ': 'lightblue', 
                 'typ_art': 'green', 'art_tsa': 'orange', 'typ_tsa': 'yellow', 'art_typ': 'purple'}
        labels = {'tsa_art': 'TSA→Art', 'tsa_typ': 'TSA→Typ', 
                 'typ_art': 'Typ→Art', 'art_tsa': 'Art→TSA', 'typ_tsa': 'Typ→TSA', 'art_typ': 'Art→Typ'}
        
        for comp_name, color in colors.items():
            if comp_name in results_dict and metric in results_dict[comp_name].columns:
                data = results_dict[comp_name][metric].dropna().values
                
                if len(data) > 2:
                    if SEABORN_AVAILABLE:
                        sns.kdeplot(data, ax=ax, color=color, label=labels[comp_name], 
                                  linewidth=2, alpha=0.7)
                    else:
                        from scipy.stats import gaussian_kde
                        try:
                            kde = gaussian_kde(data)
                            x_range = np.linspace(min(data), max(data), 100)
                            ax.plot(x_range, kde(x_range), color=color, 
                                   label=labels[comp_name], linewidth=2, alpha=0.7)
                        except:
                            pass
        
        ax.set_title(f'Density Comparison: {metric} Distributions', fontsize=16, fontweight='bold')
        ax.set_xlabel(metric)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'density_comparison_{metric}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("✓ Density plots generated for comparison across groups")

# ==================== STATISTICAL ANALYSIS FUNCTIONS ====================

def calculate_power_paired(differences, alpha=0.05):
    """Calculate statistical power for paired t-test"""
    if not STATSMODELS_AVAILABLE or len(differences) < 2:
        return np.nan
    
    try:
        # Effect size: Cohen's d for paired samples
        # d = mean_difference / std_differences
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        if std_diff == 0:
            return np.nan
            
        effect_size = np.abs(mean_diff / std_diff)
        n = len(differences)
        
        # Calculate power
        power_analysis = TTestPower()
        power = power_analysis.solve_power(
            effect_size=effect_size,
            nobs=n,
            alpha=alpha,
            power=None
        )
        return power
    except Exception as e:
        print(f"Error calculating power: {e}")
        return np.nan

def perform_comprehensive_statistical_tests(results_dict, primary_comparison=('tsa_art', 'tsa_typ')):
    """Perform comprehensive statistical analysis on ALL metrics"""
    print("\n" + "="*80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("="*80)
    
    # 1. PRIMARY COMPARISON: TSA→Art vs TSA→Typ
    print("\n1. PRIMARY COMPARISON: TSA→Artists vs TSA→Neurotypiques")
    print("-"*60)
    
    df1 = results_dict[primary_comparison[0]]
    df2 = results_dict[primary_comparison[1]]
    
    # List ALL metrics for analysis
    all_metrics = ['AuC', 'NSS', 'CC', 'SIM', 'KLDiv', 'InfoGain']
    
    results_table = []
    
    for metric in all_metrics:
        # Extract data
        data1 = df1[metric].dropna().values
        data2 = df2[metric].dropna().values
        
        # Ensure same length
        min_len = min(len(data1), len(data2))
        if min_len < 2:
            print(f"  {metric}: Insufficient data (n={min_len})")
            continue
            
        data1 = data1[:min_len]
        data2 = data2[:min_len]
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(data1, data2)
        
        # Calculate differences for effect size
        differences = data1 - data2
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        # Effect sizes
        # Cohen's d for paired samples
        if std_diff > 0:
            cohens_d = mean_diff / std_diff
        else:
            cohens_d = 0
            
        # Power analysis
        power = calculate_power_paired(differences)
        
        # Interpret effect size
        effect_magnitude = "Trivial"
        if abs(cohens_d) >= 0.2:
            effect_magnitude = "Small"
        if abs(cohens_d) >= 0.5:
            effect_magnitude = "Medium"
        if abs(cohens_d) >= 0.8:
            effect_magnitude = "Large"
        
        # Store results
        result = {
            'Metric': metric,
            'Mean_TSA→Art': np.mean(data1),
            'Mean_TSA→Typ': np.mean(data2),
            'Difference': mean_diff,
            't': t_stat,
            'p': p_value,
            'Cohen_d': cohens_d,
            'Effect_Size': effect_magnitude,
            'Power': power,
            'n': min_len
        }
        results_table.append(result)
        
        # Print detailed results
        print(f"\n  {metric}:")
        print(f"    TSA→Art: {np.mean(data1):.3f} ± {np.std(data1):.3f}")
        print(f"    TSA→Typ: {np.mean(data2):.3f} ± {np.std(data2):.3f}")
        print(f"    Difference: {mean_diff:.3f} (t={t_stat:.3f}, p={p_value:.4f})")
        print(f"    Effect: Cohen's d = {cohens_d:.3f} ({effect_magnitude})")
        
        if not np.isnan(power):
            print(f"    Statistical Power: {power:.3f}")
            if power < 0.8:
                print(f"    ⚠️  LOW POWER - may miss true effects")
            else:
                print(f"    ✓ Adequate power")
        
        if p_value < 0.05:
            if mean_diff > 0:
                print(f"    🎯 SIGNIFICANT: TSA more similar to Artists!")
            else:
                print(f"    🎯 SIGNIFICANT: TSA more similar to Neurotypiques!")
        else:
            print(f"    ❌ No significant difference")
    
    # Convert to DataFrame for easy viewing
    stats_df = pd.DataFrame(results_table)
    
    # 2. ALL PAIRWISE COMPARISONS
    print("\n\n2. ALL PAIRWISE COMPARISONS")
    print("-"*60)
    
    comparisons = [
        ('tsa_art', 'tsa_typ', 'TSA→Art vs TSA→Typ'),
        ('typ_art', 'tsa_art', 'Typ→Art vs TSA→Art'),
        ('typ_art', 'tsa_typ', 'Typ→Art vs TSA→Typ'),
        ('art_tsa', 'typ_tsa', 'Art→TSA vs Typ→TSA'),
        ('art_typ', 'typ_art', 'Art→Typ vs Typ→Art'),  
        ('art_typ', 'tsa_typ', 'Art→Typ vs TSA→Typ'),  
    ]
    
    pairwise_results = []
    
    for comp1, comp2, label in comparisons:
        if comp1 in results_dict and comp2 in results_dict:
            df1 = results_dict[comp1]
            df2 = results_dict[comp2]
            
            for metric in ['AuC', 'NSS', 'SIM']:  # Key metrics for pairwise
                data1 = df1[metric].dropna().values
                data2 = df2[metric].dropna().values
                
                min_len = min(len(data1), len(data2))
                if min_len >= 2:
                    data1 = data1[:min_len]
                    data2 = data2[:min_len]
                    
                    t_stat, p_value = stats.ttest_rel(data1, data2)
                    mean1, mean2 = np.mean(data1), np.mean(data2)
                    
                    pairwise_results.append({
                        'Comparison': label,
                        'Metric': metric,
                        f'Mean_{comp1}': mean1,
                        f'Mean_{comp2}': mean2,
                        'Difference': mean1 - mean2,
                        't': t_stat,
                        'p': p_value
                    })
    
    if pairwise_results:
        pairwise_df = pd.DataFrame(pairwise_results)
        print("\nPairwise t-test results:")
        print(pairwise_df[['Comparison', 'Metric', 'Difference', 't', 'p']].to_string())
    
    # 3. EFFECT SIZE SUMMARY
    print("\n\n3. EFFECT SIZE SUMMARY (Cohen's d)")
    print("-"*60)
    
    if not stats_df.empty:
        print("\nEffect sizes for TSA→Art vs TSA→Typ:")
        effect_summary = stats_df[['Metric', 'Cohen_d', 'Effect_Size', 'Power']]
        print(effect_summary.to_string(index=False))
        
        # Identify largest effects
        largest_effect = stats_df.loc[stats_df['Cohen_d'].abs().idxmax()]
        print(f"\n🔍 Largest effect: {largest_effect['Metric']} (d = {largest_effect['Cohen_d']:.3f}, {largest_effect['Effect_Size']})")
    
    return stats_df

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    print("="*80)
    print("COMPREHENSIVE VISUAL SALIENCY METRICS ANALYSIS")
    print("="*80)
    print("Enhanced with detailed statistics and distribution visualizations")
    print("="*80)
    
    # Check for optional packages
    if SEABORN_AVAILABLE:
        print("✓ Seaborn available for enhanced visualizations")
    else:
        print("Note: Install seaborn for better plots: pip install seaborn")
    
    if STATSMODELS_AVAILABLE:
        print("✓ Statsmodels available for power analysis")
    else:
        print("Note: Install statsmodels for power analysis: pip install statsmodels")
    
    # Run all comparisons
    print("\n" + "="*80)
    print("RUNNING COMPARISONS")
    print("="*80)
    
    results = run_all_comparisons()
    
    # Generate detailed descriptive statistics
    detailed_stats = generate_detailed_statistics(results)
    
    # COMPREHENSIVE STATISTICAL ANALYSIS
    stats_df = perform_comprehensive_statistical_tests(results)
    
    # Save statistical results
    if not stats_df.empty:
        stats_df.to_csv('statistical_analysis_results.csv', index=False)
        print("\n✓ Statistical results saved to 'statistical_analysis_results.csv'")
    
    # Create comprehensive summary table
    summary_data = []
    for comp_name, df in results.items():
        row = {'Comparison': comp_name}
        for metric in ['AuC', 'NSS', 'SIM', 'CC', 'KLDiv', 'InfoGain']:
            if metric in df.columns:
                data = df[metric].dropna()
                row[f'{metric}_mean'] = np.mean(data)
                row[f'{metric}_std'] = np.std(data)
                row[f'{metric}_min'] = np.min(data)
                row[f'{metric}_25%'] = np.percentile(data, 25)
                row[f'{metric}_median'] = np.median(data)
                row[f'{metric}_75%'] = np.percentile(data, 75)
                row[f'{metric}_max'] = np.max(data)
                row[f'{metric}_n'] = len(data)
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('summary_statistics_detailed.csv', index=False)
    print("✓ Detailed summary statistics saved to 'summary_statistics_detailed.csv'")
    
    # DISTRIBUTION VISUALIZATIONS
    print("\n" + "="*80)
    print("CREATING DISTRIBUTION VISUALIZATIONS")
    print("="*80)
    
    # Create histograms
    create_distribution_histograms(results)
    
    # Create Q-Q plots for normality check
    create_qq_plots(results)
    
    # Create density plots
    create_density_plots(results)
    
    # GENERATE FINAL REPORT
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\n📁 Output Files Generated:")
    print("\n1. Raw Data Files:")
    print("   • metrics_*.pickle - Raw metric data for each comparison")
    
    print("\n2. Statistical Analysis Files:")
    print("   • statistical_analysis_results.csv - Statistical test results")
    print("   • summary_statistics_detailed.csv - Comprehensive descriptive statistics")
    
    print("\n3. Distribution Visualization Files:")
    print("   • distribution_*.png - Histograms for each metric")
    print("   • combined_distribution_*.png - Combined histograms for each comparison")
    print("   • qq_plot_*.png - Q-Q plots for normality assessment")
    print("   • density_comparison_*.png - Density plots across comparisons")
    
    print("\n4. Key Metrics Summary:")
    print("   Comparison          AuC (Mean ± SD)        NSS (Mean ± SD)        SIM (Mean ± SD)        CC (Mean ± SD)        KLD (Mean ± SD)        IG (Mean ± SD)")
    print("   ----------------------------------------------------------------------------------------------------------------------------------------------------")
    
    for comp_name, df in results.items():
        auc_mean = df['AuC'].mean() if 'AuC' in df.columns else np.nan
        auc_std = df['AuC'].std() if 'AuC' in df.columns else np.nan
        nss_mean = df['NSS'].mean() if 'NSS' in df.columns else np.nan
        nss_std = df['NSS'].std() if 'NSS' in df.columns else np.nan
        sim_mean = df['SIM'].mean() if 'SIM' in df.columns else np.nan
        sim_std = df['SIM'].std() if 'SIM' in df.columns else np.nan
        cc_mean = df['CC'].mean() if 'CC' in df.columns else np.nan
        cc_std = df['CC'].std() if 'CC' in df.columns else np
        kld_mean = df['KLDiv'].mean() if 'KLDiv' in df.columns else np.nan
        kld_std = df['KLDiv'].std() if 'KLDiv' in df.columns else np.nan
        ig_mean = df['InfoGain'].mean() if 'InfoGain' in df.columns else np.nan
        ig_std = df['InfoGain'].std() if 'InfoGain' in df.columns else np.nan
        
        print(f"   {comp_name:<10}        {auc_mean:.3f} ± {auc_std:.3f}           {nss_mean:.3f} ± {nss_std:.3f}           {sim_mean:.3f} ± {sim_std:.3f}           {cc_mean:.3f} ± {cc_std:.3f}           {kld_mean:.3f} ± {kld_std:.3f}           {ig_mean:.3f} ± {ig_std:.3f}")
    
    print("\n🔍 Key Insights:")
    print("   • Check distribution_*.png for histogram visualizations")
    print("   • Check qq_plot_*.png for normality assessment")
    print("   • Statistical power analysis included for all tests")
    print("   • Complete descriptive statistics (min, max, quartiles, median) saved to CSV")
    
    return results, stats_df, detailed_stats

# ==================== ENTRY POINT ====================

if __name__ == "__main__":
    # Check if required folders exist
    required_folders = ['./salmaps/', './fix_maps/', './peintures/']
    missing_folders = [f for f in required_folders if not os.path.exists(f)]
    
    if missing_folders:
        print(f"Error: Missing required folders:")
        for folder in missing_folders:
            print(f"  - {folder}")
        print("\nPlease create these folders and add your data before running.")
    else:
        # Run main analysis
        results, stats, detailed_stats = main()
        print("\n✅ Analysis complete! Check the generated files for detailed results.")