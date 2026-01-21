import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import pickle

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the pipeline."""
    
    # Paths (adjusted for your structure)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # cardioTrainProject/
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA = os.path.join(DATA_DIR, 'raw', 'cardio_train.csv')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
    
    # Output file paths
    FINAL_DATASET = os.path.join(MODELS_DIR, 'cardio_final.csv')
    
    # Cleaning thresholds
    THRESHOLDS = {
        'bp_systolic': (50, 250),
        'bp_diastolic': (40, 150),
        'height': (130, 220),
        'weight': (30, 200)
    }
    
    # Feature engineering
    BMI_BINS = [0, 18.5, 25, 30, 100]
    BMI_LABELS = ['Underweight', 'Normal', 'Overweight', 'Obese']
    
    AGE_BINS = [0, 40, 50, 60, 100]
    AGE_LABELS = ['<40', '40-50', '50-60', '60+']
    
    # Labels
    GENDER_MAP = {1: 'Female', 2: 'Male'}
    CHOL_MAP = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}
    GLUC_MAP = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}
    
    # Visualization
    PLOT_STYLE = 'whitegrid'
    PLOT_DPI = 300


# ============================================================================
# PHASE 1: DATA LOADING
# ============================================================================

def load_data():
    """Load and perform initial inspection of the dataset."""
    print("\n" + "="*70)
    print("PHASE 1: DATA LOADING & INSPECTION")
    print("="*70)
    
    # Check if file exists
    if not os.path.exists(Config.RAW_DATA):
        raise FileNotFoundError(f"Data file not found: {Config.RAW_DATA}")
    
    # Load data
    df = pd.read_csv(Config.RAW_DATA, delimiter=';')
    print(f"\n✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Basic info
    print(f"✓ Missing values: {df.isnull().sum().sum()}")
    print(f"✓ Duplicates: {df.duplicated().sum()}")
    
    print("\nColumns:", ', '.join(df.columns))
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    return df


# ============================================================================
# PHASE 1.5: STATISTICAL PROFILING & CORRELATION ANALYSIS
# ============================================================================

def profile_data(df):
    """Perform comprehensive statistical profiling and correlation analysis."""
    print("\n" + "="*70)
    print("PHASE 1.5: STATISTICAL PROFILING & CORRELATION ANALYSIS")
    print("="*70)

    # Convert age to years for analysis
    df_profile = df.copy()
    df_profile['age_years'] = (df_profile['age'] / 365.25).round(1)

    # ========================================================================
    # 1. CORRELATION ANALYSIS
    # ========================================================================
    print("\n📊 Feature Correlation Analysis:")

    # Select relevant features for correlation
    corr_features = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo',
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
    correlation_matrix = df_profile[corr_features].corr()

    # Identify strong correlations with target variable
    target_corr = correlation_matrix['cardio'].drop('cardio').sort_values(ascending=False)
    print("\nCorrelations with Cardiovascular Disease:")
    for feature, corr in target_corr.items():
        strength = "Strong" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.1 else "Weak"
        print(f"  {feature:15s}: {corr:6.3f} ({strength})")

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'correlation_heatmap.png'),
                dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved: correlation_heatmap.png")

    # ========================================================================
    # 2. TARGET VARIABLE RELATIONSHIP ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("🎯 Target Variable Relationship Analysis:")
    print("="*70)

    # Overall disease prevalence
    disease_prev = df['cardio'].mean() * 100
    print(f"\nOverall Disease Prevalence: {disease_prev:.2f}%")

    # Disease prevalence by categorical features
    print("\nDisease Prevalence by Feature:")

    # Gender
    gender_prev = df.groupby('gender')['cardio'].agg(['mean', 'count'])
    gender_prev['mean'] *= 100
    print(f"\n  Gender:")
    print(f"    Female (1): {gender_prev.loc[1, 'mean']:.2f}% (n={gender_prev.loc[1, 'count']:,})")
    print(f"    Male (2):   {gender_prev.loc[2, 'mean']:.2f}% (n={gender_prev.loc[2, 'count']:,})")

    # Cholesterol
    chol_prev = df.groupby('cholesterol')['cardio'].agg(['mean', 'count'])
    chol_prev['mean'] *= 100
    print(f"\n  Cholesterol:")
    for level in [1, 2, 3]:
        if level in chol_prev.index:
            print(f"    Level {level}: {chol_prev.loc[level, 'mean']:.2f}% (n={chol_prev.loc[level, 'count']:,})")

    # Glucose
    gluc_prev = df.groupby('gluc')['cardio'].agg(['mean', 'count'])
    gluc_prev['mean'] *= 100
    print(f"\n  Glucose:")
    for level in [1, 2, 3]:
        if level in gluc_prev.index:
            print(f"    Level {level}: {gluc_prev.loc[level, 'mean']:.2f}% (n={gluc_prev.loc[level, 'count']:,})")

    # Smoking
    smoke_prev = df.groupby('smoke')['cardio'].agg(['mean', 'count'])
    smoke_prev['mean'] *= 100
    print(f"\n  Smoking:")
    print(f"    Non-smoker: {smoke_prev.loc[0, 'mean']:.2f}% (n={smoke_prev.loc[0, 'count']:,})")
    print(f"    Smoker:     {smoke_prev.loc[1, 'mean']:.2f}% (n={smoke_prev.loc[1, 'count']:,})")

    # Continuous variables comparison
    print("\n  Continuous Variables (Mean by Disease Status):")
    continuous_features = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']
    for feature in continuous_features:
        no_disease = df_profile[df_profile['cardio'] == 0][feature].mean()
        disease = df_profile[df_profile['cardio'] == 1][feature].mean()
        diff = disease - no_disease
        print(f"    {feature:12s}: No Disease={no_disease:6.2f}, Disease={disease:6.2f}, Diff={diff:+6.2f}")

    # Visualizations - Save each as separate file
    print("\n  Creating individual visualizations...")

    # 1. Age distribution by disease
    plt.figure(figsize=(8, 6))
    plt.hist(df_profile[df_profile['cardio']==0]['age_years'], bins=30,
             alpha=0.6, label='No Disease', color='green', edgecolor='black')
    plt.hist(df_profile[df_profile['cardio']==1]['age_years'], bins=30,
             alpha=0.6, label='Disease', color='red', edgecolor='black')
    plt.xlabel('Age (years)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Age Distribution by Disease Status', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'target_age_distribution.png'),
                dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()

    # 2. Gender vs Disease
    plt.figure(figsize=(8, 6))
    gender_disease = pd.crosstab(df['gender'], df['cardio'], normalize='index') * 100
    x_pos = np.arange(len(gender_disease))
    width = 0.35
    plt.bar(x_pos - width/2, gender_disease[0], width, label='No Disease',
            color='green', edgecolor='black', alpha=0.7)
    plt.bar(x_pos + width/2, gender_disease[1], width, label='Disease',
            color='red', edgecolor='black', alpha=0.7)
    plt.xlabel('Gender', fontsize=11)
    plt.ylabel('Percentage (%)', fontsize=11)
    plt.title('Disease Prevalence by Gender', fontsize=12, fontweight='bold')
    plt.xticks(x_pos, ['Female', 'Male'])
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'target_gender_prevalence.png'),
                dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()

    # 3. Cholesterol vs Disease
    plt.figure(figsize=(8, 6))
    chol_disease = pd.crosstab(df['cholesterol'], df['cardio'], normalize='index') * 100
    x_pos = np.arange(len(chol_disease))
    plt.bar(x_pos - width/2, chol_disease[0], width, label='No Disease',
            color='green', edgecolor='black', alpha=0.7)
    plt.bar(x_pos + width/2, chol_disease[1], width, label='Disease',
            color='red', edgecolor='black', alpha=0.7)
    plt.xlabel('Cholesterol Level', fontsize=11)
    plt.ylabel('Percentage (%)', fontsize=11)
    plt.title('Disease Prevalence by Cholesterol', fontsize=12, fontweight='bold')
    plt.xticks(x_pos, ['Normal', 'Above\nNormal', 'Well Above\nNormal'])
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'target_cholesterol_prevalence.png'),
                dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()

    # 4. Systolic BP box plot by disease
    plt.figure(figsize=(8, 6))
    bp_data_sys = [df[df['cardio']==0]['ap_hi'], df[df['cardio']==1]['ap_hi']]
    bp_sys = plt.boxplot(bp_data_sys, labels=['No Disease', 'Disease'],
                         patch_artist=True, widths=0.6)
    bp_sys['boxes'][0].set_facecolor('green')
    bp_sys['boxes'][1].set_facecolor('red')
    for box in bp_sys['boxes']:
        box.set_alpha(0.7)
    plt.ylabel('Systolic BP (mmHg)', fontsize=11)
    plt.title('Systolic BP by Disease Status', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'target_systolic_bp.png'),
                dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()

    # 5. Diastolic BP box plot by disease
    plt.figure(figsize=(8, 6))
    bp_data_dia = [df[df['cardio']==0]['ap_lo'], df[df['cardio']==1]['ap_lo']]
    bp_dia = plt.boxplot(bp_data_dia, labels=['No Disease', 'Disease'],
                         patch_artist=True, widths=0.6)
    bp_dia['boxes'][0].set_facecolor('green')
    bp_dia['boxes'][1].set_facecolor('red')
    for box in bp_dia['boxes']:
        box.set_alpha(0.7)
    plt.ylabel('Diastolic BP (mmHg)', fontsize=11)
    plt.title('Diastolic BP by Disease Status', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'target_diastolic_bp.png'),
                dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()

    # 6. Glucose vs Disease
    plt.figure(figsize=(8, 6))
    gluc_disease = pd.crosstab(df['gluc'], df['cardio'], normalize='index') * 100
    x_pos = np.arange(len(gluc_disease))
    plt.bar(x_pos - width/2, gluc_disease[0], width, label='No Disease',
            color='green', edgecolor='black', alpha=0.7)
    plt.bar(x_pos + width/2, gluc_disease[1], width, label='Disease',
            color='red', edgecolor='black', alpha=0.7)
    plt.xlabel('Glucose Level', fontsize=11)
    plt.ylabel('Percentage (%)', fontsize=11)
    plt.title('Disease Prevalence by Glucose', fontsize=12, fontweight='bold')
    plt.xticks(x_pos, ['Normal', 'Above\nNormal', 'Well Above\nNormal'])
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'target_glucose_prevalence.png'),
                dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()

    print("  ✓ Saved: target_age_distribution.png")
    print("  ✓ Saved: target_gender_prevalence.png")
    print("  ✓ Saved: target_cholesterol_prevalence.png")
    print("  ✓ Saved: target_systolic_bp.png")
    print("  ✓ Saved: target_diastolic_bp.png")
    print("  ✓ Saved: target_glucose_prevalence.png")

    print("\n✓ Statistical profiling complete")

    return correlation_matrix


# ============================================================================
# PHASE 2: DATA QUALITY CHECK
# ============================================================================

def check_quality(df):
    """Identify data quality issues and outliers."""
    print("\n" + "="*70)
    print("PHASE 2: DATA QUALITY CHECK")
    print("="*70)
    
    issues = {}
    
    # Blood pressure outliers
    bp_sys_low = (df['ap_hi'] < Config.THRESHOLDS['bp_systolic'][0]).sum()
    bp_sys_high = (df['ap_hi'] > Config.THRESHOLDS['bp_systolic'][1]).sum()
    bp_dia_low = (df['ap_lo'] < Config.THRESHOLDS['bp_diastolic'][0]).sum()
    bp_dia_high = (df['ap_lo'] > Config.THRESHOLDS['bp_diastolic'][1]).sum()
    impossible_bp = (df['ap_lo'] > df['ap_hi']).sum()
    
    issues['bp_total'] = bp_sys_low + bp_sys_high + bp_dia_low + bp_dia_high + impossible_bp
    
    # Height outliers
    height_low = (df['height'] < Config.THRESHOLDS['height'][0]).sum()
    height_high = (df['height'] > Config.THRESHOLDS['height'][1]).sum()
    issues['height_total'] = height_low + height_high
    
    # Weight outliers
    weight_low = (df['weight'] < Config.THRESHOLDS['weight'][0]).sum()
    weight_high = (df['weight'] > Config.THRESHOLDS['weight'][1]).sum()
    issues['weight_total'] = weight_low + weight_high
    
    # Report
    print("\nOutliers detected:")
    print(f"  Blood Pressure:  {issues['bp_total']:6,} rows ({issues['bp_total']/len(df)*100:.2f}%)")
    print(f"    - Systolic:    {bp_sys_low + bp_sys_high:6,}")
    print(f"    - Diastolic:   {bp_dia_low + bp_dia_high:6,}")
    print(f"    - Impossible:  {impossible_bp:6,}")
    print(f"  Height:          {issues['height_total']:6,} rows ({issues['height_total']/len(df)*100:.2f}%)")
    print(f"  Weight:          {issues['weight_total']:6,} rows ({issues['weight_total']/len(df)*100:.2f}%)")
    
    total_issues = sum(issues.values())
    print(f"\n✓ Total affected: ~{total_issues:,} rows (~{total_issues/len(df)*100:.2f}%)")
    
    return issues


# ============================================================================
# PHASE 3: DATA CLEANING
# ============================================================================

def clean_data(df):
    """Remove outliers and invalid values."""
    print("\n" + "="*70)
    print("PHASE 3: DATA CLEANING")
    print("="*70)
    
    initial_count = len(df)
    
    # Clean blood pressure
    df = df[
        (df['ap_hi'] >= Config.THRESHOLDS['bp_systolic'][0]) & 
        (df['ap_hi'] <= Config.THRESHOLDS['bp_systolic'][1]) &
        (df['ap_lo'] >= Config.THRESHOLDS['bp_diastolic'][0]) & 
        (df['ap_lo'] <= Config.THRESHOLDS['bp_diastolic'][1]) &
        (df['ap_lo'] <= df['ap_hi'])
    ].copy()
    
    after_bp = len(df)
    print(f"\n  After BP cleaning:     {after_bp:,} rows (-{initial_count - after_bp:,})")
    
    # Clean height
    df = df[
        (df['height'] >= Config.THRESHOLDS['height'][0]) & 
        (df['height'] <= Config.THRESHOLDS['height'][1])
    ].copy()
    
    after_height = len(df)
    print(f"  After height cleaning: {after_height:,} rows (-{after_bp - after_height:,})")
    
    # Clean weight
    df = df[
        (df['weight'] >= Config.THRESHOLDS['weight'][0]) & 
        (df['weight'] <= Config.THRESHOLDS['weight'][1])
    ].copy()
    
    final_count = len(df)
    print(f"  After weight cleaning: {final_count:,} rows (-{after_height - final_count:,})")
    
    # Summary
    total_removed = initial_count - final_count
    retention = (final_count / initial_count) * 100
    
    print(f"\n✓ Total removed: {total_removed:,} ({100-retention:.2f}%)")
    print(f"✓ Retention rate: {retention:.2f}%")
    
    return df


# ============================================================================
# PHASE 3.5: CLEANING IMPACT ASSESSMENT
# ============================================================================

def assess_cleaning_impact(df_before, df_after):
    """Quantify the impact of data cleaning on statistical properties."""
    print("\n" + "="*70)
    print("PHASE 3.5: CLEANING IMPACT ASSESSMENT")
    print("="*70)

    # ========================================================================
    # 1. STATISTICAL COMPARISON
    # ========================================================================
    print("\n📊 Statistical Impact of Cleaning:")

    continuous_vars = ['ap_hi', 'ap_lo', 'height', 'weight']
    comparison_data = []

    for var in continuous_vars:
        before_mean = df_before[var].mean()
        after_mean = df_after[var].mean()
        before_std = df_before[var].std()
        after_std = df_after[var].std()

        comparison_data.append({
            'Variable': var,
            'Before_Mean': f"{before_mean:.2f}",
            'After_Mean': f"{after_mean:.2f}",
            'Mean_Shift': f"{after_mean - before_mean:+.2f}",
            'Before_Std': f"{before_std:.2f}",
            'After_Std': f"{after_std:.2f}",
            'Std_Reduction': f"{((before_std - after_std) / before_std * 100):.1f}%"
        })

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    # ========================================================================
    # 2. DISTRIBUTION METRICS
    # ========================================================================
    print("\n\n📈 Distribution Metrics (Before → After):")

    for var in continuous_vars:
        before_q25 = df_before[var].quantile(0.25)
        after_q25 = df_after[var].quantile(0.25)
        before_median = df_before[var].median()
        after_median = df_after[var].median()
        before_q75 = df_before[var].quantile(0.75)
        after_q75 = df_after[var].quantile(0.75)

        print(f"\n  {var}:")
        print(f"    Q1:     {before_q25:7.2f} → {after_q25:7.2f} ({after_q25 - before_q25:+.2f})")
        print(f"    Median: {before_median:7.2f} → {after_median:7.2f} ({after_median - before_median:+.2f})")
        print(f"    Q3:     {before_q75:7.2f} → {after_q75:7.2f} ({after_q75 - before_q75:+.2f})")

    # ========================================================================
    # 3. TARGET VARIABLE ANALYSIS
    # ========================================================================
    print("\n\n🎯 Disease Prevalence Impact:")

    before_prevalence = df_before['cardio'].mean() * 100
    after_prevalence = df_after['cardio'].mean() * 100
    prevalence_change = after_prevalence - before_prevalence

    print(f"  Before cleaning: {before_prevalence:.2f}%")
    print(f"  After cleaning:  {after_prevalence:.2f}%")
    print(f"  Change:          {prevalence_change:+.2f} percentage points")

    if abs(prevalence_change) < 1:
        print("  ✓ Minimal bias introduced by cleaning (< 1%)")
    else:
        print("  ⚠ Note: Cleaning affected disease prevalence")

    # ========================================================================
    # 4. VISUALIZATION: BEFORE/AFTER STATISTICAL COMPARISON
    # ========================================================================
    print("\n  Creating individual cleaning impact visualizations...")

    variables = continuous_vars

    # 1. Mean comparison
    plt.figure(figsize=(10, 6))
    before_means = [df_before[var].mean() for var in variables]
    after_means = [df_after[var].mean() for var in variables]
    x_pos = np.arange(len(variables))
    width = 0.35
    plt.bar(x_pos - width/2, before_means, width, label='Before',
            color='red', edgecolor='black', alpha=0.7)
    plt.bar(x_pos + width/2, after_means, width, label='After',
            color='green', edgecolor='black', alpha=0.7)
    plt.xlabel('Variable', fontsize=11)
    plt.ylabel('Mean Value', fontsize=11)
    plt.title('Mean Values: Before vs After Cleaning', fontsize=12, fontweight='bold')
    plt.xticks(x_pos, variables)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'cleaning_mean_comparison.png'),
                dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()

    # 2. Standard deviation comparison
    plt.figure(figsize=(10, 6))
    before_stds = [df_before[var].std() for var in variables]
    after_stds = [df_after[var].std() for var in variables]
    plt.bar(x_pos - width/2, before_stds, width, label='Before',
            color='red', edgecolor='black', alpha=0.7)
    plt.bar(x_pos + width/2, after_stds, width, label='After',
            color='green', edgecolor='black', alpha=0.7)
    plt.xlabel('Variable', fontsize=11)
    plt.ylabel('Standard Deviation', fontsize=11)
    plt.title('Standard Deviation: Before vs After Cleaning', fontsize=12, fontweight='bold')
    plt.xticks(x_pos, variables)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'cleaning_std_comparison.png'),
                dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()

    # 3. Sample size and disease prevalence
    plt.figure(figsize=(10, 6))
    categories = ['Total Samples', 'No Disease', 'Disease']
    before_counts = [len(df_before),
                    len(df_before[df_before['cardio']==0]),
                    len(df_before[df_before['cardio']==1])]
    after_counts = [len(df_after),
                   len(df_after[df_after['cardio']==0]),
                   len(df_after[df_after['cardio']==1])]
    x_pos = np.arange(len(categories))
    plt.bar(x_pos - width/2, before_counts, width, label='Before',
            color='red', edgecolor='black', alpha=0.7)
    plt.bar(x_pos + width/2, after_counts, width, label='After',
            color='green', edgecolor='black', alpha=0.7)
    plt.xlabel('Category', fontsize=11)
    plt.ylabel('Count', fontsize=11)
    plt.title('Sample Distribution: Before vs After Cleaning', fontsize=12, fontweight='bold')
    plt.xticks(x_pos, categories, rotation=15)
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'cleaning_sample_distribution.png'),
                dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()

    # 4. Variability reduction
    plt.figure(figsize=(10, 6))
    std_reduction = [((df_before[var].std() - df_after[var].std()) / df_before[var].std() * 100)
                     for var in variables]
    colors_reduction = ['green' if x > 0 else 'red' for x in std_reduction]
    plt.bar(range(len(variables)), std_reduction, color=colors_reduction,
            edgecolor='black', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.xlabel('Variable', fontsize=11)
    plt.ylabel('Std Reduction (%)', fontsize=11)
    plt.title('Variability Reduction by Cleaning', fontsize=12, fontweight='bold')
    plt.xticks(range(len(variables)), variables)
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'cleaning_variability_reduction.png'),
                dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()

    print("  ✓ Saved: cleaning_mean_comparison.png")
    print("  ✓ Saved: cleaning_std_comparison.png")
    print("  ✓ Saved: cleaning_sample_distribution.png")
    print("  ✓ Saved: cleaning_variability_reduction.png")

    print("\n✓ Cleaning impact assessment complete")


# ============================================================================
# PHASE 4: DATA TRANSFORMATION
# ============================================================================

def transform_data(df):
    """Create new features from cleaned data."""
    print("\n" + "="*70)
    print("PHASE 4: DATA TRANSFORMATION")
    print("="*70)
    
    print("\nCreating features...")
    
    # 1. Age conversion
    df['age_years'] = (df['age'] / 365.25).round(1)
    print("  ✓ age_years")
    
    # 2. Age groups
    df['age_group'] = pd.cut(df['age_years'], bins=Config.AGE_BINS, labels=Config.AGE_LABELS)
    print("  ✓ age_group")
    
    # 3. BMI calculation
    df['bmi'] = (df['weight'] / ((df['height'] / 100) ** 2)).round(2)
    print("  ✓ bmi")
    
    # 4. BMI categories
    df['bmi_category'] = pd.cut(df['bmi'], bins=Config.BMI_BINS, labels=Config.BMI_LABELS)
    print("  ✓ bmi_category")
    
    # 5. Blood pressure categories
    def categorize_bp(row):
        s, d = row['ap_hi'], row['ap_lo']
        if s < 120 and d < 80:
            return 'Normal'
        elif s < 130 and d < 80:
            return 'Elevated'
        elif s < 140 or d < 90:
            return 'High Stage 1'
        else:
            return 'High Stage 2'
    
    df['bp_category'] = df.apply(categorize_bp, axis=1)
    print("  ✓ bp_category")
    
    # 6. Gender labels
    df['gender_label'] = df['gender'].map(Config.GENDER_MAP)
    print("  ✓ gender_label")
    
    # 7. Cholesterol labels
    df['cholesterol_label'] = df['cholesterol'].map(Config.CHOL_MAP)
    print("  ✓ cholesterol_label")
    
    # 8. Glucose labels
    df['glucose_label'] = df['gluc'].map(Config.GLUC_MAP)
    print("  ✓ glucose_label")
    
    # 9. Health risk score
    df['health_risk_score'] = 0
    
    bp_risk = {'Normal': 0, 'Elevated': 1, 'High Stage 1': 2, 'High Stage 2': 3}
    df['health_risk_score'] += df['bp_category'].map(bp_risk)
    df['health_risk_score'] += (df['cholesterol'] - 1)
    df['health_risk_score'] += (df['gluc'] - 1)
    
    bmi_risk = {'Underweight': 1, 'Normal': 0, 'Overweight': 1, 'Obese': 2}
    df['health_risk_score'] += df['bmi_category'].map(bmi_risk)
    
    df['health_risk_score'] += df['smoke']
    df['health_risk_score'] += (1 - df['active'])
    
    print("  ✓ health_risk_score")
    
    # 10. Risk level
    df['risk_level'] = pd.cut(df['health_risk_score'], 
                              bins=[-1, 2, 5, 12],
                              labels=['Low Risk', 'Moderate Risk', 'High Risk'])
    print("  ✓ risk_level")
    
    print(f"\n✓ Created 10 new features")
    print(f"✓ Final dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return df


# ============================================================================
# PHASE 5: MACHINE LEARNING MODELING
# ============================================================================

def train_and_evaluate_models(df):
    """Train multiple classification models and evaluate performance."""
    print("\n" + "="*70)
    print("PHASE 5: MACHINE LEARNING MODELING")
    print("="*70)

    # ========================================================================
    # 1. FEATURE SELECTION & DATA PREPARATION
    # ========================================================================
    print("\n📊 Preparing data for modeling...")

    # Select features for modeling
    feature_cols = ['age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                    'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    target_col = 'cardio'

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    print(f"  Features selected: {len(feature_cols)}")
    print(f"  Target variable: {target_col}")
    print(f"  Dataset size: {len(X):,} samples")
    print(f"  Class distribution:")
    print(f"    No disease: {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)")
    print(f"    Disease:    {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)")

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Train set: {len(X_train):,} samples")
    print(f"  Test set:  {len(X_test):,} samples")

    # Feature scaling (important for Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    scaler_path = os.path.join(Config.MODELS_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Feature scaling applied and scaler saved")

    # ========================================================================
    # 2. MODEL TRAINING
    # ========================================================================
    print("\n" + "="*70)
    print("🤖 Training Models...")
    print("="*70)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")

        # Use scaled data for Logistic Regression, original for tree-based
        if name == 'Logistic Regression':
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        else:
            X_train_use = X_train
            X_test_use = X_test

        # Train model
        model.fit(X_train_use, y_train)
        trained_models[name] = model

        # Predictions
        y_pred = model.predict(X_test_use)
        y_pred_proba = model.predict_proba(X_test_use)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_use, y_train, cv=5, scoring='accuracy')

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        print(f"    ✓ Accuracy:  {accuracy:.4f}")
        print(f"    ✓ Precision: {precision:.4f}")
        print(f"    ✓ Recall:    {recall:.4f}")
        print(f"    ✓ F1-Score:  {f1:.4f}")
        print(f"    ✓ ROC-AUC:   {roc_auc:.4f}")
        print(f"    ✓ CV Score:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # ========================================================================
    # 3. MODEL COMPARISON & BEST MODEL SELECTION
    # ========================================================================
    print("\n" + "="*70)
    print("🏆 Model Comparison & Selection")
    print("="*70)

    # Find best model by ROC-AUC
    best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']

    print(f"\n  Best Model: {best_model_name}")
    print(f"  ROC-AUC Score: {results[best_model_name]['roc_auc']:.4f}")

    # Save best model
    best_model_path = os.path.join(Config.MODELS_DIR, 'best_model.pkl')
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"  ✓ Best model saved: models/best_model.pkl")

    # Save all models
    for name, model_obj in trained_models.items():
        model_filename = name.lower().replace(' ', '_') + '.pkl'
        model_path = os.path.join(Config.MODELS_DIR, model_filename)
        with open(model_path, 'wb') as f:
            pickle.dump(model_obj, f)
    print(f"  ✓ All models saved to models/ directory")

    # ========================================================================
    # 4. FEATURE IMPORTANCE ANALYSIS
    # ========================================================================
    print("\n" + "="*70)
    print("📊 Feature Importance Analysis")
    print("="*70)

    # Get feature importance from best tree-based model
    if best_model_name in ['Random Forest', 'Gradient Boosting']:
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print(f"\n  Top 5 Most Important Features ({best_model_name}):")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"    {row['Feature']:15s}: {row['Importance']:.4f}")
    else:
        # For Logistic Regression, use coefficients
        feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': np.abs(best_model.coef_[0])
        }).sort_values('Importance', ascending=False)

        print(f"\n  Top 5 Most Important Features (by coefficient magnitude):")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"    {row['Feature']:15s}: {row['Importance']:.4f}")

    # ========================================================================
    # 5. CREATE COMPREHENSIVE VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*70)
    print("📈 Creating Model Visualizations...")
    print("="*70)

    # Create 4-panel visualization
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Panel 1: Model Performance Comparison (Metrics Bar Chart)
    ax1 = fig.add_subplot(gs[0, :])
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    x = np.arange(len(metrics))
    width = 0.25

    for i, (name, result) in enumerate(results.items()):
        values = [result[m] for m in metrics]
        ax1.bar(x + i*width, values, width, label=name, alpha=0.8, edgecolor='black')

    ax1.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.1)

    # Panel 2: ROC Curves
    ax2 = fig.add_subplot(gs[1, 0])
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        ax2.plot(fpr, tpr, label=f'{name} (AUC={result["roc_auc"]:.3f})', linewidth=2)

    ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
    ax2.set_xlabel('False Positive Rate', fontsize=11)
    ax2.set_ylabel('True Positive Rate', fontsize=11)
    ax2.set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)

    # Panel 3: Confusion Matrix (Best Model)
    ax3 = fig.add_subplot(gs[1, 1])
    cm = results[best_model_name]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax3,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax3.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Actual', fontsize=11, fontweight='bold')
    ax3.set_title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold')

    # Panel 4: Feature Importance
    ax4 = fig.add_subplot(gs[1, 2])
    top_features = feature_importance.head(10)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    ax4.barh(range(len(top_features)), top_features['Importance'], color=colors, edgecolor='black')
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(top_features['Feature'])
    ax4.set_xlabel('Importance', fontsize=11, fontweight='bold')
    ax4.set_title(f'Top 10 Feature Importance - {best_model_name}', fontsize=12, fontweight='bold')
    ax4.invert_yaxis()
    ax4.grid(alpha=0.3, axis='x')

    # Panel 5: Classification Report Heatmap (Best Model)
    ax5 = fig.add_subplot(gs[2, :2])
    report = classification_report(y_test, results[best_model_name]['y_pred'],
                                   target_names=['No Disease', 'Disease'], output_dict=True)
    report_df = pd.DataFrame(report).iloc[:-1, :2].T  # Exclude support, macro/weighted avg
    sns.heatmap(report_df, annot=True, fmt='.3f', cmap='RdYlGn', cbar=True, ax=ax5,
                vmin=0, vmax=1, linewidths=1, linecolor='black')
    ax5.set_xlabel('Metric', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Class', fontsize=11, fontweight='bold')
    ax5.set_title(f'Classification Report - {best_model_name}', fontsize=12, fontweight='bold')

    # Panel 6: Cross-Validation Scores
    ax6 = fig.add_subplot(gs[2, 2])
    cv_means = [results[name]['cv_mean'] for name in results.keys()]
    cv_stds = [results[name]['cv_std'] for name in results.keys()]
    x_pos = np.arange(len(results))
    ax6.bar(x_pos, cv_means, yerr=cv_stds, capsize=5, alpha=0.8,
            color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([name.replace(' ', '\n') for name in results.keys()], fontsize=9)
    ax6.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax6.set_title('5-Fold Cross-Validation Scores', fontsize=12, fontweight='bold')
    ax6.grid(alpha=0.3, axis='y')
    ax6.set_ylim(0, 1.1)

    # Add overall title
    fig.suptitle('Cardiovascular Disease Prediction - Model Evaluation Dashboard',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save visualization
    plt.savefig(os.path.join(Config.OUTPUT_DIR, 'model_evaluation.png'),
                dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved: model_evaluation.png")

    # ========================================================================
    # 6. SAVE DETAILED RESULTS REPORT
    # ========================================================================
    print("\n" + "="*70)
    print("💾 Saving Model Results...")
    print("="*70)

    results_report_path = os.path.join(Config.OUTPUT_DIR, 'model_results.txt')

    with open(results_report_path, 'w') as f:
        f.write("MACHINE LEARNING MODEL EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("DATASET INFORMATION:\n")
        f.write("-"*70 + "\n")
        f.write(f"Total samples:       {len(X):,}\n")
        f.write(f"Training samples:    {len(X_train):,} (80%)\n")
        f.write(f"Test samples:        {len(X_test):,} (20%)\n")
        f.write(f"Number of features:  {len(feature_cols)}\n")
        f.write(f"Target variable:     {target_col}\n\n")

        f.write("FEATURES USED:\n")
        f.write("-"*70 + "\n")
        for i, feat in enumerate(feature_cols, 1):
            f.write(f"{i:2d}. {feat}\n")
        f.write("\n")

        f.write("MODEL PERFORMANCE COMPARISON:\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}\n")
        f.write("-"*70 + "\n")
        for name, result in results.items():
            f.write(f"{name:<25} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
                   f"{result['recall']:<10.4f} {result['f1']:<10.4f} {result['roc_auc']:<10.4f}\n")
        f.write("\n")

        f.write("BEST MODEL:\n")
        f.write("-"*70 + "\n")
        f.write(f"Model: {best_model_name}\n")
        f.write(f"ROC-AUC Score: {results[best_model_name]['roc_auc']:.4f}\n")
        f.write(f"Accuracy: {results[best_model_name]['accuracy']:.4f}\n")
        f.write(f"F1-Score: {results[best_model_name]['f1']:.4f}\n\n")

        f.write("FEATURE IMPORTANCE (Top 10):\n")
        f.write("-"*70 + "\n")
        for idx, row in feature_importance.head(10).iterrows():
            f.write(f"{row['Feature']:<20s}: {row['Importance']:.4f}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("MODELING COMPLETE\n")

    print(f"✓ Results report saved: outputs/model_results.txt")
    print("\n✓ Modeling phase complete")

    return results, best_model_name, feature_importance


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(df_before, df_after):
    """Create before/after comparison plots."""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    sns.set_style(Config.PLOT_STYLE)
    
    # Plot 1: Before/After Cleaning Comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Data Cleaning: Before vs After', fontsize=16, fontweight='bold')
    
    # Systolic BP
    axes[0, 0].hist(df_before['ap_hi'], bins=50, color='red', alpha=0.6, label='Before')
    axes[0, 0].hist(df_after['ap_hi'], bins=50, color='green', alpha=0.6, label='After')
    axes[0, 0].set_title('Systolic BP')
    axes[0, 0].set_xlabel('mmHg')
    axes[0, 0].legend()
    
    # Diastolic BP
    axes[0, 1].hist(df_before['ap_lo'], bins=50, color='red', alpha=0.6, label='Before')
    axes[0, 1].hist(df_after['ap_lo'], bins=50, color='green', alpha=0.6, label='After')
    axes[0, 1].set_title('Diastolic BP')
    axes[0, 1].set_xlabel('mmHg')
    axes[0, 1].legend()
    
    # BP Scatter
    axes[0, 2].scatter(df_after['ap_hi'], df_after['ap_lo'], alpha=0.3, s=1, color='green')
    axes[0, 2].plot([40, 250], [40, 250], 'k--', alpha=0.5)
    axes[0, 2].set_title('BP Relationship (After)')
    axes[0, 2].set_xlabel('Systolic')
    axes[0, 2].set_ylabel('Diastolic')
    
    # Height
    axes[1, 0].hist(df_before['height'], bins=50, color='red', alpha=0.6, label='Before')
    axes[1, 0].hist(df_after['height'], bins=50, color='green', alpha=0.6, label='After')
    axes[1, 0].set_title('Height')
    axes[1, 0].set_xlabel('cm')
    axes[1, 0].legend()
    
    # Weight
    axes[1, 1].hist(df_before['weight'], bins=50, color='red', alpha=0.6, label='Before')
    axes[1, 1].hist(df_after['weight'], bins=50, color='green', alpha=0.6, label='After')
    axes[1, 1].set_title('Weight')
    axes[1, 1].set_xlabel('kg')
    axes[1, 1].legend()
    
    # Age
    axes[1, 2].hist(df_after['age_years'], bins=30, color='skyblue', edgecolor='black')
    axes[1, 2].set_title('Age Distribution (After)')
    axes[1, 2].set_xlabel('years')
    
    plt.tight_layout()
    plot1_path = os.path.join(Config.OUTPUT_DIR, 'cleaning_comparison.png')
    plt.savefig(plot1_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: cleaning_comparison.png")
    
    # Plot 2: New Features
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('New Features Created', fontsize=16, fontweight='bold')
    
    # BMI
    axes[0, 0].hist(df_after['bmi'], bins=30, color='lightcoral', edgecolor='black')
    axes[0, 0].set_title('BMI Distribution')
    axes[0, 0].set_xlabel('BMI')
    
    # BMI Categories
    bmi_counts = df_after['bmi_category'].value_counts()
    axes[0, 1].bar(bmi_counts.index, bmi_counts.values, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('BMI Categories')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Age Groups
    age_counts = df_after['age_group'].value_counts().sort_index()
    axes[0, 2].bar(range(len(age_counts)), age_counts.values, 
                   tick_label=age_counts.index, color='plum', edgecolor='black')
    axes[0, 2].set_title('Age Groups')
    
    # BP Categories
    bp_order = ['Normal', 'Elevated', 'High Stage 1', 'High Stage 2']
    bp_counts = df_after['bp_category'].value_counts().reindex(bp_order, fill_value=0)
    axes[1, 0].bar(range(len(bp_counts)), bp_counts.values,
                   tick_label=bp_counts.index, color='salmon', edgecolor='black')
    axes[1, 0].set_title('BP Categories')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Risk Score
    axes[1, 1].hist(df_after['health_risk_score'], 
                   bins=range(0, df_after['health_risk_score'].max()+2),
                   color='gold', edgecolor='black')
    axes[1, 1].set_title('Health Risk Score')
    axes[1, 1].set_xlabel('Score')
    
    # Risk Levels
    risk_order = ['Low Risk', 'Moderate Risk', 'High Risk']
    risk_counts = df_after['risk_level'].value_counts().reindex(risk_order, fill_value=0)
    colors = ['green', 'orange', 'red']
    axes[1, 2].bar(range(len(risk_counts)), risk_counts.values,
                   tick_label=risk_counts.index, color=colors, edgecolor='black', alpha=0.7)
    axes[1, 2].set_title('Risk Levels')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plot2_path = os.path.join(Config.OUTPUT_DIR, 'new_features.png')
    plt.savefig(plot2_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: new_features.png")


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(df, summary_stats):
    """Save cleaned data and summary report."""
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    # Create directories
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Save cleaned data to models/ folder
    df.to_csv(Config.FINAL_DATASET, index=False)
    print(f"\n✓ Dataset saved: models/cardio_final.csv")
    
    # Save summary report to outputs/ folder
    report_path = os.path.join(Config.OUTPUT_DIR, 'wrangling_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("DATA WRANGLING SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET TRANSFORMATION:\n")
        f.write("-"*70 + "\n")
        f.write(f"Original rows:      {summary_stats['original_rows']:,}\n")
        f.write(f"Final rows:         {summary_stats['final_rows']:,}\n")
        f.write(f"Rows removed:       {summary_stats['rows_removed']:,} ({summary_stats['removal_pct']:.2f}%)\n")
        f.write(f"Original columns:   {summary_stats['original_cols']}\n")
        f.write(f"Final columns:      {summary_stats['final_cols']}\n")
        f.write(f"New features:       {summary_stats['new_features']}\n\n")
        
        f.write("CLEANING APPLIED:\n")
        f.write("-"*70 + "\n")
        f.write(f"Blood Pressure:  {Config.THRESHOLDS['bp_systolic'][0]}-{Config.THRESHOLDS['bp_systolic'][1]} / {Config.THRESHOLDS['bp_diastolic'][0]}-{Config.THRESHOLDS['bp_diastolic'][1]} mmHg\n")
        f.write(f"Height:          {Config.THRESHOLDS['height'][0]}-{Config.THRESHOLDS['height'][1]} cm\n")
        f.write(f"Weight:          {Config.THRESHOLDS['weight'][0]}-{Config.THRESHOLDS['weight'][1]} kg\n\n")
        
        f.write("NEW FEATURES CREATED:\n")
        f.write("-"*70 + "\n")
        features = ['age_years', 'age_group', 'bmi', 'bmi_category', 'bp_category',
                   'gender_label', 'cholesterol_label', 'glucose_label', 
                   'health_risk_score', 'risk_level']
        for i, feat in enumerate(features, 1):
            f.write(f"{i:2d}. {feat}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("DATA WRANGLING COMPLETE\n")
    
    print(f"✓ Report saved: outputs/wrangling_report.txt")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute complete data wrangling pipeline."""
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  CARDIOVASCULAR DISEASE DATA WRANGLING PIPELINE".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    start_time = datetime.now()
    print(f"\nStarted: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Phase 1: Load data
        df_original = load_data()

        # Phase 1.5: Statistical profiling & correlation analysis
        correlation_matrix = profile_data(df_original)

        # Phase 2: Check quality
        quality_issues = check_quality(df_original)

        # Phase 3: Clean data
        df_clean = clean_data(df_original)

        # Phase 3.5: Assess cleaning impact
        assess_cleaning_impact(df_original, df_clean)

        # Phase 4: Transform data
        df_final = transform_data(df_clean)

        # Phase 5: Machine Learning Modeling
        model_results, best_model_name, feature_importance = train_and_evaluate_models(df_final)

        # Create wrangling visualizations
        create_visualizations(df_original, df_final)

        # Save results
        summary_stats = {
            'original_rows': len(df_original),
            'final_rows': len(df_final),
            'rows_removed': len(df_original) - len(df_final),
            'removal_pct': ((len(df_original) - len(df_final)) / len(df_original)) * 100,
            'original_cols': len(df_original.columns),
            'final_cols': len(df_final.columns),
            'new_features': len(df_final.columns) - len(df_original.columns)
        }

        save_results(df_final, summary_stats)

        # Final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "█"*70)
        print("█" + " "*68 + "█")
        print("█" + "  ✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY".center(68) + "█")
        print("█" + " "*68 + "█")
        print("█"*70)

        print(f"\nExecution time: {duration:.1f} seconds")
        print(f"Final dataset: {len(df_final):,} rows × {len(df_final.columns)} columns")
        print(f"Best Model: {best_model_name} (ROC-AUC: {model_results[best_model_name]['roc_auc']:.4f})")

        print(f"\n📁 Output files:")
        print(f"   Data:")
        print(f"   • models/cardio_final.csv")
        print(f"   \n   Profiling & Correlation (7 files):")
        print(f"   • outputs/correlation_heatmap.png")
        print(f"   • outputs/target_age_distribution.png")
        print(f"   • outputs/target_gender_prevalence.png")
        print(f"   • outputs/target_cholesterol_prevalence.png")
        print(f"   • outputs/target_systolic_bp.png")
        print(f"   • outputs/target_diastolic_bp.png")
        print(f"   • outputs/target_glucose_prevalence.png")
        print(f"   \n   Cleaning Impact (4 files):")
        print(f"   • outputs/cleaning_mean_comparison.png")
        print(f"   • outputs/cleaning_std_comparison.png")
        print(f"   • outputs/cleaning_sample_distribution.png")
        print(f"   • outputs/cleaning_variability_reduction.png")
        print(f"   \n   Data Wrangling (2 combined dashboards):")
        print(f"   • outputs/cleaning_comparison.png")
        print(f"   • outputs/new_features.png")
        print(f"   \n   Model Evaluation (1 combined dashboard - FOR ASSIGNMENT UPLOAD):")
        print(f"   • outputs/model_evaluation.png")
        print(f"   \n   Reports:")
        print(f"   • outputs/wrangling_report.txt")
        print(f"   • outputs/model_results.txt")
        print(f"   \n   Trained Models:")
        print(f"   • models/best_model.pkl ({best_model_name})")
        print(f"   • models/logistic_regression.pkl")
        print(f"   • models/random_forest.pkl")
        print(f"   • models/gradient_boosting.pkl")
        print(f"   • models/scaler.pkl")
        print("\n🎉 All done! Your complete ML pipeline is ready!\n")
        print("📝 FOR ASSIGNMENT:")
        print("   Upload: outputs/model_evaluation.png")
        print("   This 6-panel dashboard shows all model results comprehensively.\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please make sure cardio_train.csv is in data/raw/ folder")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()