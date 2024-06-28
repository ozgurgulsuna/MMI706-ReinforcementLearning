https://github.com/google-deepmind/mujoco/issues/1172
https://github.com/dfki-ric/phobos
https://github.com/zalo/mujoco_wasm
https://kzakka.com/robopianist/


multiple robot configurations 
https://roboti.us/forum/index.php?threads/can-the-parameters-of-a-model-be-changed-on-the-fly.3354/
https://github.com/openai/gym/issues/1860

Q4. If I want to continue training the agent in stages and test its intermediate performance, how can I do that?

You need to save the Q-table (or, in general, the weights/parameters of a function approximator like a neural-net) but also the current value of the episilon (if using some schedule, so if changing during the episodes) and similar. Then you instantiate the agent, load the various parameters, and continue training where you left.