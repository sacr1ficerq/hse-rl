def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """
    q = 0
    next_states = mdp.get_next_states(state, action)

    for next_state in next_states:
        p = mdp.get_transition_prob(state, action, next_state)
        r = mdp.get_reward(state, action, next_state)
        v = state_values[next_state]
        q += p * (r + gamma * v)
    
    return q
