# run.py
import os
import torch
import numpy as np
from env.env_sf import SFEnv  # replace with actual path if needed
from agent.PPO_agent import PPOModel, PPOAgent  # replace with your PPO code file/module
import time

# -------------------------------
# Configuration
# -------------------------------
# ENV_XML = "half_cheetah.xml"   # path to your MuJoCo XML
OBS_DIM = 17                   # set appropriately (self.obs_dim)
N_ACTIONS = 16                  # set appropriately (env.model.nu)
IS_DISCRETE = True            # HalfCheetah is continuous
#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"

# Training hyperparameters
NUM_STEPS_PER_UPDATE = 2048
TOTAL_UPDATES = 2
ACTOR_LR = 3e-4
CRITIC_LR = 1e-4
GAMMA = 0.99
CLIP_EPS = 0.3
BATCH_SIZE = 512
EPOCHS = 10
VALUE_COEF = 0.5
ENTROPY_COEF = 0.02

# -------------------------------
# Training function
# -------------------------------
def train_ppo( total_updates, num_steps_per_update, model_save_path):
    env = SFEnv()
    #print(env.action_space)
    model = PPOModel(env.obs_dim, len(env.action_space), IS_DISCRETE)
    agent = PPOAgent(
        env,
        model,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        gamma=GAMMA,
        clip_eps=CLIP_EPS,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        
        device=DEVICE
    )

    print("Starting PPO training...")
    agent.train(num_steps_per_update=NUM_STEPS_PER_UPDATE, total_updates=TOTAL_UPDATES)
    print(f"Environment initialized with observation dim {env.obs_dim} and action dim {env.model.nu}")
    print("Training completed.")

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(model_save_path, "ppo_model.pth"))
    print("Model saved as ppo_model.pth")
    return env, agent

# -------------------------------
# Testing / Visualization function
# -------------------------------
def test_ppo(model_path="ppo_model.pth", xml_path="half_cheetah.xml", episodes=5, render=True, device=None):
    """
    Test a trained PPO model independently.

    Args:
        model_path (str): Path to the saved PPO model weights.
        xml_path (str): Path to MuJoCo XML environment.
        episodes (int): Number of episodes to run.
        render (bool): Whether to render the environment.
        device (str or torch.device): 'cpu' or 'cuda'. Default auto-detect.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    env = SFEnv(xml_path)
    obs_dim = env.obs_dim
    n_actions = env.action_space.shape[0]
    is_discrete = True

    # Load model
    model = PPOModel(obs_dim, n_actions, is_discrete)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Create PPO agent
    agent = PPOAgent(env, model, device=device)

    # Run episodes
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(agent._device)
            with torch.no_grad():
                action, _, _ = agent.get_action(obs_tensor)
            action = action.cpu().numpy().squeeze()

            obs, reward, done, _ = env.step(action)
            total_reward += reward

            if render:
                env.render()
                time.sleep(0.01)

        print(f"Episode {ep+1} reward: {total_reward}")

    env.close()
    print("Testing completed.")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    # Example usage
    
    model_path = os.path.join("output")

    # Train
    train_ppo(  total_updates=1000, num_steps_per_update=2048, model_save_path=model_path)

    # Test
    #test_ppo(model_path=model_path, xml_path=xml_path, episodes=5, render=True)
