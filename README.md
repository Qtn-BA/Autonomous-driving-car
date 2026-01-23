# 🚘 Autonomous Driving with Deep Reinforcement Learning

This project simulates an **autonomous vehicle** that learns how to drive using **Deep Reinforcement Learning (DRL)**.  
The environment is built **from scratch** using **Pygame**, and the agent learns entirely through **trial and error**, without any hard-coded driving rules.

The goal is to navigate a road safely while managing speed, lanes, obstacles, and traffic lights.

---

## ✨ Key Features

The autonomous agent is trained to handle several real-world driving scenarios:

- **Obstacle Avoidance**  
  Detects and avoids other vehicles on the road.

- **Traffic Regulation**  
  Recognizes traffic lights and stops at red lights.

- **Lane Keeping**  
  Maintains a safe lateral position within the lane.

- **Speed Control**  
  Adjusts acceleration and braking based on the driving context.

---

## 🤖 How It Works

The system is based on a **Double Deep Q-Network (Double DQN)** algorithm.

### High-Level Training Loop

1. **Perception**  
   The vehicle uses simulated sensors to observe its environment:
   - Distance to obstacles
   - Current speed
   - Traffic light status
   - Lane position

2. **Decision**  
   A neural network processes the state information and selects an action:
   - Steer left / right
   - Accelerate
   - Brake

3. **Learning**
   - Safe driving → **positive reward**
   - Collision or running a red light → **negative penalty**

Over time, the agent updates its policy to **maximize cumulative rewards** and **minimize dangerous behavior**.

---

## 🧠 Reinforcement Learning Algorithm

- **Double Deep Q-Network (Double DQN)**
  - Reduces overestimation bias found in standard DQN
  - Uses a target network for more stable learning

---

## 🛠️ Built With

- **Python** – Core logic and training loop  
- **Pygame** – Simulation environment, rendering, and physics  
- **PyTorch** – Neural network and Double DQN implementation  

---

## 📌 Project Goal

The objective is to demonstrate how a reinforcement learning agent can learn **complex driving behaviors** in a custom environment using only rewards and penalties—without predefined rules or heuristics.

---

## 🚀 Future Improvements (Optional)

- Add more complex road layouts
- Introduce pedestrian agents
- Implement continuous control (DDPG / PPO)
- Improve sensor realism

---

## 📷 Demo

*(Add screenshots or a GIF of the simulation here)*

