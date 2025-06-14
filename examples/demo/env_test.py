from marllib import marl
# prepare env

env = marl.make_env(environment_name="mate", map_name="MATE-4v2-9-v0", coop_team="camera",force_coop=True)
# initialize algorithm with appointed hyper-parameters
mappo = marl.algos.ippo(hyperparam_source="test")
# build agent model based on env + algorithms + user preference
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-256"})
# start training
mappo.fit(env, model, stop={"timesteps_total": 1000000}, checkpoint_freq=100, share_policy="group", local_mode=False, num_gpus=1)