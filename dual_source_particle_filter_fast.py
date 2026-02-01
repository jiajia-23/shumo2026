"""
Fast version of Dual-Source Particle Filter (200 particles for testing)
Includes new visualizations:
- Fan support trend for Season 15 (similar to fan_percent_estimation style)
- Prediction interval comparison (old vs new method)
"""
import dual_source_particle_filter as dsf

# Override configuration for faster execution
dsf.CONFIG['N_PARTICLES'] = 200

if __name__ == "__main__":
    dsf.main()
