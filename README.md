🚘 Project Overview
This project simulates an autonomous vehicle that learns to drive by itself using Deep Reinforcement Learning (DRL).

Built from scratch with a custom Pygame environment, the AI agent must navigate a road, manage its speed, and interact with traffic elements without any pre-programmed rules. It learns solely through trial and error, getting rewarded for safe driving and penalized for mistakes.

✨ Key Features
The autonomous agent is trained to handle several real-world driving scenarios:


Obstacle Avoidance: Detecting and dodging other vehicles on the road.


Traffic Regulation: Recognizing and stopping at red lights.


Lane Keeping: Maintaining a safe lateral position on the road.


Speed Control: Managing acceleration and braking based on the context.

🤖 How It Works
The system uses an algorithm called Double Deep Q-Network (Double DQN). Here is the high-level loop:


Perception: The car uses "sensors" to see its environment (distance to obstacles, current speed, traffic light status).


Decision: A Neural Network processes this information and chooses the best action (steer left/right, accelerate, or brake).

Learning:

If the car drives safely, it receives a positive reward.

If it crashes or runs a red light, it receives a negative penalty.

Over time, the AI updates its strategy to maximize rewards and minimize accidents.

🛠️ Built With
Python: Core logic.


Pygame: Rendering the simulation and handling physics.


PyTorch: Implementation of the Neural Network and Double DQN algorithm.
