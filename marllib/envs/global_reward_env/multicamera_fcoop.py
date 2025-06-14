import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Discrete, Box, Tuple
import mate

policy_mapping_dict = {
    "all_scenario": {
        "description": "multicamera scenarios",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": True,
    },
}


class RLlibMultiCamera_FCOOP(MultiAgentEnv):
    def __init__(self,env_config):
        map = env_config["map_name"]
        env_config.pop("map_name", None)
        self.env = mate.make(map)

        # self.env = mate.MultiCamera(self.env, target_agent=mate.agents.GreedyTargetAgent(seed=0))

        self.observation_space = GymDict({"obs":self.env.observation_space.spaces[0]})
        self.action_space = self.env.action_space.spaces[0]

        self.num_agents = self.env.num_cameras

        self.agents = ["agent_{}".format(i) for i in range(self.env.num_cameras)]

        env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i, name in enumerate(self.agents):
            obs[name] = {"obs": original_obs[i]}
        return obs

    def step(self,action_dict):
        action = []
        for camera_name in self.agents:
            action.append(action_dict[camera_name])
        joint_observation, team_reward, done, infos = self.env.step(np.array(action))
        rewards={}
        obs={}

        for i,name in enumerate(self.agents):
            rewards[name] = team_reward
            obs[name] = {"obs": joint_observation[i]}
        dones = {"__all__": done}
        return obs,rewards,dones,{}

    def close(self):
        self.env.close()

    def render(self, mode=None):
        self.env.render()
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "agents": self.agents,
            "map_name": self.env_config["map_name"],
            "episode_limit": 1000,
            "policy_mapping_info": policy_mapping_dict,
        }
        return env_info
