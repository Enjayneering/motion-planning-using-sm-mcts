config_payoffs = {
    
    # Payoff Parameters
    'discount_factor': 0.9,

    'weight_interm': 0.5,
    'weight_final': 1,
    
    'weight_interm_vec': {'dist': 0,
                        'coll': 1,
                        'prog': 0,
                        'comp': 1,},

    'weight_final_vec': {'time': 0,
                        'win': 0, # better not, because too ambiguous
                        'lead': 1,},
}