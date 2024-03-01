config_research = {
    'run_mode': {'test': False, 'exp': True, 'live-plot': True},
    'computation': {'numpy': True, 'jax': False},
    # Statistical Analysis
    'num_sim': 10,

    # Open vs Closed Loop
    'sensing': {'open': False, 'closed': True},
    
    # Only run k timesteps in simulation
    'max_timesteps_sim': 1,

    # test mode
    'freq_stat_data': 10
}