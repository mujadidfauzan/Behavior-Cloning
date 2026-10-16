import time

import numpy as np
import robosuite as suite

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
)

obs = env.reset()

print("Franka Panda random movement started!")
print("Action dimension:", env.action_dim)

low, high = env.action_spec

# Initial action
action = np.zeros(env.action_dim)
target_action = action.copy()

# New target every 2 seconds
change_interval = 40
step_count = 0

try:
    while True:

        # Generate a new random target
        if step_count % change_interval == 0:
            target_action = np.random.uniform(low, high)

            # Keep gripper open
            target_action[-1] = -1.0

        # Smooth movement
        action = 0.95 * action + 0.05 * target_action

        # Apply action
        obs, reward, done, info = env.step(action)

        # Render
        env.render()

        # Approximately 20 Hz
        time.sleep(1.0 / 20.0)

        step_count += 1

        if done:
            obs = env.reset()
            action[:] = 0
            step_count = 0

except KeyboardInterrupt:
    print("\nStopping simulation...")

finally:
    env.close()
