from matplotlib import pyplot as plt
import random
from src.games import SimpleGame

random.seed(10) # set random seed


class GameInstance:
   def __init__(self, T=10000):
      self.T = T
      self.game = SimpleGame(self.T)   
     
   
   def play_game(self):
      
      # allocate space to keep track of everything
      observed_rewards = []
      self.game.reset_game()
      for t in range(self.T):
         observed_rewards.append(self.game.step())
         self.game.update_policies(observed_rewards[t], t)

      for i in range(self.game.num_players):
         actions = self.game.players[i].get_policy()
         print(f"Learned policy of player {i} is: {[f'{action:.2f}' for action in actions]}")

      return observed_rewards

def main():

   T = 25 # number of guesses
   game_instance = GameInstance(T)
   reward_trajectories = game_instance.play_game()  

   # --------------   Plot the results   -------------- 
   # plot rewards
   fig, ax1 =plt.subplots(1,1)
   ax1.plot(range(1,T+1), reward_trajectories, '--')
   ax1.set(xlabel='Guesses', ylabel='Reward')
   plt.show()
   
if __name__ == '__main__':
   main()



