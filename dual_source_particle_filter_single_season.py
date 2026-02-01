"""
Single Season Analysis Version of Dual-Source Particle Filter
==============================================================
This version allows you to specify which season(s) to analyze,
instead of running all 34 seasons every time.

Usage:
1. Modify SEASONS_TO_ANALYZE list below to specify which seasons to run
2. Run: python dual_source_particle_filter_single_season.py

Examples:
- SEASONS_TO_ANALYZE = [15] - Run only Season 15
- SEASONS_TO_ANALYZE = [15, 20, 25] - Run Seasons 15, 20, and 25
- SEASONS_TO_ANALYZE = list(range(1, 11)) - Run Seasons 1-10
"""

import dual_source_particle_filter as dsf
import pandas as pd
import os

# ==========================================
# CONFIGURATION - Modify these settings
# ==========================================

# Specify which seasons to analyze (list of season numbers)
SEASONS_TO_ANALYZE = [8]  # Default: Season 15 only

# Number of particles (adjust for speed vs accuracy trade-off)
N_PARTICLES = 500  # 500 is a good balance for single season analysis

# Which season to use for visualizations
VISUALIZATION_SEASON = 15  # Must be in SEASONS_TO_ANALYZE list

# ==========================================
# Main Execution
# ==========================================

def main():
    """
    Run particle filter for selected seasons only
    """
    print("\n" + "=" * 60)
    print("SINGLE SEASON DUAL-SOURCE PARTICLE FILTER ANALYSIS")
    print("=" * 60)
    print(f"Seasons to analyze: {SEASONS_TO_ANALYZE}")
    print(f"Particles per season: {N_PARTICLES}")
    print("=" * 60)

    # Override particle count
    dsf.CONFIG['N_PARTICLES'] = N_PARTICLES

    # Initialize estimator with preprocessed data
    data_path = 'fan_est_cache.csv'

    if not os.path.exists(data_path):
        print(f"\nError: Data file not found: {data_path}")
        print("Please run fan_percent_estimation.py first to generate the cache file.")
        return

    estimator = dsf.DualSourceEstimator(data_path)
    estimator.load_data()

    # Filter data to selected seasons only
    print(f"\n[Step 2] Filtering data to selected seasons...")
    original_count = len(estimator.df)
    estimator.df = estimator.df[estimator.df['season'].isin(SEASONS_TO_ANALYZE)]
    filtered_count = len(estimator.df)
    print(f"  Filtered from {original_count} to {filtered_count} records")

    if filtered_count == 0:
        print(f"\nError: No data found for seasons {SEASONS_TO_ANALYZE}")
        return

    # Create organized directory structure for this run
    paths = dsf.create_run_directories(SEASONS_TO_ANALYZE, N_PARTICLES)

    print(f"\nOutput directories created:")
    print(f"  Base: {paths['base']}")
    print(f"  Prediction: {paths['prediction']}")
    print(f"  Fan Support: {paths['fan_support']}")
    print(f"  Analysis: {paths['analysis']}")

    # Run particle filter for selected seasons
    results_df, credibility_df = estimator.run_all()

    # Save results to base directory
    results_path = os.path.join(paths['base'], 'dual_source_estimates.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n[Results] Saved estimates to: {results_path}")

    credibility_path = os.path.join(paths['base'], 'model_credibility.csv')
    credibility_df.to_csv(credibility_path, index=False, encoding='utf-8-sig')
    print(f"[Results] Saved credibility scores to: {credibility_path}")

    # Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    evaluator = dsf.ModelEvaluator(paths)

    # Check if visualization season is in the analyzed seasons
    if VISUALIZATION_SEASON not in SEASONS_TO_ANALYZE:
        print(f"\nWarning: Visualization season {VISUALIZATION_SEASON} not in analyzed seasons.")
        print(f"Using first analyzed season: {SEASONS_TO_ANALYZE[0]}")
        viz_season = SEASONS_TO_ANALYZE[0]
    else:
        viz_season = VISUALIZATION_SEASON

    # 1. Model credibility heatmap (for selected seasons)
    evaluator.plot_confidence_heatmap(credibility_df)

    # 2. Confidence over time (for selected seasons)
    evaluator.plot_confidence_over_time(credibility_df)

    # 3. Top contestants comparison (from selected seasons)
    evaluator.plot_top_contestants_comparison(results_df, top_n=10)

    # 4. Fan support trend for visualization season
    evaluator.plot_fan_support_trend_season(results_df, season=viz_season)

    # 5. Prediction interval comparison - Top 3 contestants overlay
    evaluator.plot_prediction_interval_comparison(data_path, results_df,
                                                  season=viz_season, top_n=3)

    # 6. Single contestant detailed comparison
    season_data = results_df[results_df['season'] == viz_season]
    if len(season_data) > 0:
        top_contestant = season_data.groupby('celebrity')['fan_share_mean'].mean().idxmax()
        print(f"\n  Top contestant in Season {viz_season}: {top_contestant}")
        evaluator.plot_single_contestant_interval_comparison(data_path, results_df,
                                                            top_contestant, season=viz_season)

        # 7. Latent dynamics for top 3 contestants in visualization season
        print(f"\n[Visualization] Plotting latent dynamics for top 3 in Season {viz_season}...")
        top_3 = season_data.groupby('celebrity')['fan_share_mean'].mean().nlargest(3).index
        for celebrity in top_3:
            evaluator.plot_latent_dynamics(results_df, celebrity)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED!")
    print(f"Analyzed seasons: {SEASONS_TO_ANALYZE}")
    print(f"Results saved to: {dsf.OUTPUT_DIR}")
    print(f"Visualizations saved to: {dsf.PLOT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
