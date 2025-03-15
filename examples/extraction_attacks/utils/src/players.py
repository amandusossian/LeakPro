import numpy as np
import scipy.optimize as opt

class TsallisInf:
    def __init__(self, num_actions):
        self.num_actions = num_actions # number of actions available to the player
        self.alpha = 0.5 # alpha value in the Tsallis-Inf algorithm
        self.cumulative_losses = np.zeros(self.num_actions) # Keep track of the losses observed for each action
        self.last_played_action = -1 # keep track of latest action
        self.weights = np.full((self.num_actions), 1/self.num_actions) # current strategy
    
    
    def sample_action(self):
        """Sample an action based on the strategy."""
        self.last_played_action = np.random.choice(a=np.arange(self.num_actions), p=self.weights)
        return self.last_played_action
    
    # Inspired by https://smpybandits.github.io/_modules/Policies/TsallisInf.html
    def update_policy(self, reward, time):
        """Update the strategy based on the observed reward using the Tsallis-Inf algorithm"""
        # for a reward in [0,1], loss = 1 - reward
        biased_loss = 1.0 - reward
        # unbiased estimate, from the weights of the previous step
        unbiased_loss = biased_loss / self.weights[self.last_played_action]
        self.cumulative_losses[self.last_played_action] += unbiased_loss
        eta_t = 1.0 / np.sqrt(max(1,time))
        
        # solve f(x)=1 to get an approximation of the (unique) Lagrange multiplier x
        def objective_function(x):
            return (np.sum( (eta_t * (self.cumulative_losses - x + np.finfo(float).eps)) ** -2) - 1)**2 
        result_of_minimization = opt.minimize_scalar(objective_function)
        x = result_of_minimization.x
        #  use x to compute the new weights
        new_weights =  ( eta_t * (self.cumulative_losses - x) ) ** -2
        
        # Bad case, where the sum is so small that it's only rounding errors
        # or where all values where bad and forced to 0, start with new_weights=[1/K...]
        if not np.all(np.isfinite(new_weights)):
            new_weights[:] = 1.0
        # 3. Renormalize weights at each step
        new_weights /= np.sum(new_weights)
        # 4. store weights
        self.weights =  new_weights
        
    def reset(self):
        """Reset the strategy to the uniform strategy"""
        self.weights = np.full((self.num_actions), 1/self.num_actions)     
        self.cumulative_losses = np.zeros(self.num_actions) # Keep track of the losses observed for each action
        self.last_played_action = -1 # keep track of latest action
             
    def get_policy(self):
        """Read out the policy"""
        return self.weights
    