# ParkSmart-RL
ParkSmart-RL: PPO-based intelligent parking slot selection using reinforcement learning. Agent optimizes cost, distance &amp; availability in a dynamic 12-slot lot. Real-time Pygame visualization with vehicle path tracing, reward heatmap &amp; interactive controls. Built with Stable-Baselines3, Gymnasium and Python.


### Objective  
The ParkSmart-RL project implements an intelligent parking slot selection system using **Proximal Policy Optimization (PPO)** — a state-of-the-art reinforcement learning algorithm. The agent learns to autonomously choose the optimal parking spot in a dynamic 12-slot environment by balancing **cost**, **distance**, and **availability**, simulating real-world smart parking scenarios. This project demonstrates how RL can power decision-making in autonomous vehicles and smart city applications.

### Skills Learned  
- Deep understanding of Reinforcement Learning (RL) and Markov Decision Processes (MDPs)  
- Mastery of Proximal Policy Optimization (PPO) algorithm and policy gradients  
- Designing custom Gymnasium environments for real-world problems  
- Training stable RL agents using Stable-Baselines3  
- Real-time 2D visualization and interactive simulation with Pygame  
- Reward shaping and exploration-exploitation trade-offs  

### Tools & Technologies Used  
- **Python 3.13** – Core programming language  
- **Stable-Baselines3** – PPO implementation  
- **Gymnasium** – RL environment framework  
- **Pygame** – Real-time visualization engine  
- **NumPy** – State processing and normalization  
- **Matplotlib** (optional) – Training reward plotting  

### Key Features  
- 12-slot dynamic parking environment with random cost, distance & occupancy  
- PPO agent trained for 15,000 timesteps  
- Interactive Pygame visualization (900×500) with:  
  → Vehicle movement & path tracing  
  → Reward-based heatmap (red = bad, green = good)  
  → Real-time slot info (cost, distance, reward)  
- User controls:  
  `SPACE` → Manual PPO decision  
  `A` → Auto mode (continuous autonomous parking)  
  `R` → Reset environment  
  `ESC` → Exit  

### Demo Screenshots  

**Ref 1: Parking Environment & Trained Agent in Action**  
![Simulation Demo](screenshots/parksmart_demo.gif)  
*Agent autonomously selects high-reward (green) slots*

**Ref 2: Reward Heatmap Visualization**  
![Heatmap](screenshots/heatmap.png)  
*Red-to-green gradient shows learned slot values*

**Ref 3: Training Progress (Reward Curve)**  
![Training](screenshots/reward_curve.png)  
*Average reward increases over 15,000 timesteps*

**Ref 4: Interactive Controls & Info Panel**  
![Controls](screenshots/controls_info.png)  
*Episode count, last reward, and selected slot details*
