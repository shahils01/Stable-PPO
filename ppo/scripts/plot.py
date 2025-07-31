import numpy as np
import matplotlib.pyplot as plt

#reward_file = "dairl_rewards_0.npy"
#plot_name = "dairl_rewards_0"

reward_file = "half_cheetah_training.npy"
plot_name = "half_cheetah_training"

rewards = np.load(reward_file).flatten()
print('shape = ', rewards.shape)

# Create x values (e.g., episodes or time steps)
x_values = np.arange(1, len(rewards) + 1)

plt.clf()  # Clear the previous plot
plt.plot(rewards, label=plot_name)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.pause(0.01)  # Pause for a short time to allow the plot to update
plt.savefig(plot_name+'.png')

reward_file = "half_cheetah_eval.npy"
plot_name = "half_cheetah_eval"

rewards = np.load(reward_file).flatten()
print('shape = ', rewards.shape)

# Create x values (e.g., episodes or time steps)
x_values = np.arange(1, len(rewards) + 1)

plt.clf()  # Clear the previous plot
plt.plot(rewards, label=plot_name)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.legend()
plt.pause(0.01)  # Pause for a short time to allow the plot to update
plt.savefig(plot_name+'.png')