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
ENV_XML = "half_cheetah.xml"   # path to your MuJoCo XML
OBS_DIM = 17                   # set appropriately (self.obs_dim)
N_ACTIONS = 6                  # set appropriately (env.model.nu)
IS_DISCRETE = True           # HalfCheetah is continuous
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
# Testing / Visualization function
# -------------------------------
def test_ppo(model_save_path,model_name = None, episodes=100,  device=None):
    """
    Test a trained PPO model independently.

    
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    env = SFEnv()
    #print(env.action_space)
    model = PPOModel(env.obs_dim, len(env.action_space), IS_DISCRETE)

    # Load model
    model.load_state_dict(torch.load(os.path.join(model_save_path, model_name), map_location=DEVICE))
    model.to(device)

    # Create PPO agent
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

    # Run episodes
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _,dist = agent.get_action(obs_tensor)
            action = action.cpu().numpy().squeeze()

            obs, reward, done, _ = env.step(dist,test=True)
            total_reward += reward

            

        print(f"Episode {ep+1} reward: {total_reward}")

    env.close()
    print("Testing completed.")

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    # Example usage
    
    model_save_path = os.path.join("output")
    model_name="ppo_model_20251206_120304.pth"

    # Train
    #train_ppo(xml_path=xml_path, total_updates=1000, num_steps_per_update=2048, model_save_path=model_path)

    # Test
    test_ppo(model_save_path,model_name ,  episodes=100)
