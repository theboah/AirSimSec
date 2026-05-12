import json

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage

from secEnv import SecEnvAll
from SecUtils import generate_tmp_scene_config

def main():
    #Take training params from config
    with open("sim_config/sec_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    num_envs = config["num_envs"]
    name = config.get("model_name")
    timesteps = config.get("total_timesteps")
    scene_config_name = config["scene_config_name"]

    tmp_config_name = generate_tmp_scene_config(num_envs, scene_config_name)

    #create secEnv
    env = SecEnvAll(config, tmp_config_name, num_envs)
    env = VecMonitor(env, filename= name)
    
    env = VecTransposeImage(env)

    #create model
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./logs/",
        device="cuda"
    )

    #learn
    model.learn(total_timesteps=timesteps, progress_bar=True, tb_log_name=name)

    #save
    model.save(name + ".zip")
    env.close()

if __name__ == "__main__":
    main()
