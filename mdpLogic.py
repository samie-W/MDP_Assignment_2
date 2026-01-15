import numpy as np

class GridWorldMDP:
    def __init__(self, rows=4, cols=4, gamma=0.9):
        self.rows = rows
        self.cols = cols
        self.gamma = gamma 
        self.states = [(r, c) for r in range(rows) for c in range(cols)]
        self.actions = ['Up', 'Down', 'Left', 'Right']
        
       
        self.rewards = np.full((rows, cols), -0.1) 
        self.terminals = {(0, 3): 10, (1, 3): -10} 
        for (r, c), val in self.terminals.items():
            self.rewards[r, c] = val

    def get_transitions(self, state, action):
        if state in self.terminals:
            return []
        
     
        moves = {'Up': (-1, 0), 'Down': (1, 0), 'Left': (0, -1), 'Right': (0, 1)}
        sides = {'Up': ['Left', 'Right'], 'Down': ['Left', 'Right'], 
                 'Left': ['Up', 'Down'], 'Right': ['Up', 'Down']}
        
        results = []
     
        next_s = self._move(state, moves[action])
        results.append((0.8, next_s))
      
        for s_act in sides[action]:
            next_s = self._move(state, moves[s_act])
            results.append((0.1, next_s))
        return results

    def _move(self, state, delta):
        r, c = state[0] + delta[0], state[1] + delta[1]
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return (r, c)
        return state 

def value_iteration(mdp, iterations=100):
    V = np.zeros((mdp.rows, mdp.cols))
    history = []
    for _ in range(iterations):
        new_V = V.copy()
        delta = 0
        for s in mdp.states:
            if s in mdp.terminals:
                new_V[s] = mdp.terminals[s]
                continue
            v_list = []
            for a in mdp.actions:
                v_list.append(sum(p * (mdp.rewards[ns] + mdp.gamma * V[ns]) 
                              for p, ns in mdp.get_transitions(s, a)))
            new_V[s] = max(v_list)
            delta = max(delta, abs(new_V[s] - V[s]))
        V = new_V
        history.append(V.copy())
        if delta < 1e-4: break
    return V, history