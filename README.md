# StrategicNavigations
Strategic Navigations: Reinforcement Learning in Grid-World

1 Introduction

In the third assignment, you must implement two reinforcement learning al- gorithms, Q-Learning and SARSA, to develop an intelligent agent navigating through Grid-World and find the optimal path. The Grid-World is a world consisting of four types of nodes: “Flat”, “Mountain”, “Goal” and “Pitfall”. The agent aims to navigate on the Grid-World and reach a “Goal” node while maximizing the total reward/score. It is worth noting that more than one goal node can exist in some scenarios. The agent must visit one of the goal nodes when there are multiple goals. Besides, the agent can move in four directions: ‘UP’, ‘DOWN’, ‘LEFT’, and ‘RIGHT’. Additionally, the transition from one node to another has a reward based on the node type.
We provide a code base in Python; you should use it to implement the algorithms. The algorithms should be able to find the optimal list of actions that lead to the maximum total reward. Additionally, you will be required to compare the performance of the algorithms in various scenarios. You can use the grid generator in the code base to create various scenarios. The performance metrics you should consider include the optimality gap, computational time, the number of iterations to converge, the convergence speed, and so on.


.1 Grid-World
Grid-World is a square map with dimensions n × n containing n2 nodes, as illustrated in Figure 1. The agent aims to reach the goal nodes by riding a bike and moving in the directions of ‘UP’, ‘DOWN’, ‘LEFT’, and ‘RIGHT’. The Grid-World consists of four types of nodes, each with its own characteristics:
• Flat: It is easy to ride a bike.
1
Figure 1: Demonstration of Example Grid-World
<img width="221" alt="Screenshot 2024-07-10 at 18 14 54" src="https://github.com/FaridGahramanov2/StrategicNavigations/assets/153610282/f67b3499-f6db-4fbb-b672-c083e6e1e920">

Mountain: It is hard to ride a bike.
• Goal: This node is the target destination for the agent. • Pitfall: If the agent enters this node, it will fail.
One of the nodes in the map is designated as the starting point for the agent. Moving from one node to another provides a reward based on the type of the nodes, as shown in Table 1. The reinforcement learning (RL) algorithms aim to find the best sequence of actions to maximize the total reward.

Table 1: Transition Reward Function between two Node Types
Note: If the agent tries moving out of a Grid-World, it stays on the same node and gets a −1 reward.
For this assignment, we provide both single-goal and multiple-goal scenarios. In both cases, the agent must visit one of goal nodes without falling pitfalls by minimizing the total traveling cost. Note that the ultimate aim is maximizing the total collected score.

2.2 Implementation
This text presents the instructions for the programming assignment. The as- signment involves implementing Q-Learning and SARSA algorithms in Python for a grid-world environment. The students are expected to complete the cor- responding methods in the “QLearning.py” and “SARSA.py” files in the given code base. The code base is implemented in Python, and Jupyter is not al- lowed. Students who need to learn how to code with Python can learn from available sources on the internet. The “Environment.py” file contains the nec- essary information about the grid world and provides moving on it and the transition reward. The “RLAgent.py” file contains an abstract class for Rein- forcement Learning (i.e., RL) agents; each RL algorithm must be a sub-class. The “QLearning.py” file contains the implementation of the Q-Learning algo- rithm, while the “SARSA.py” file contains the implementation of the SARSA algorithm. Finally, some helpful links to learn Python programming language from scratch are shared at the end of the document (Section 6).
• Environment.py: The “Environment” class in this file holds the neces- sary information about the Grid-World and provides moving on it and the transition reward. It takes the file path where the scenario data exists. The scenario data file is in Pickle format. Some essential methods in this class are listed:
– reset(): The agent goes to a predefined starting point/node, then the method returns the node index of the starting point.
– to node index(position): This method converts a given position (i.e., row and column indices as a list) to a node index.
– to position(node index): This method converts a node index to a position.
– set current position(node index): It changes the agent’s current position to the given node index.
– get node type(position): It returns the node type of the given po- sition as a string. This string is the node type name’s first (upper) character.
– get reward(previous pos, next pos): This method provides the tran- sition reward from the previous position (previous pos) to the next position (new pos) based on Table 1.
– is done(position): It states whether the given position ends the episode based on the current environmental state. In other words, it determines whether any termination condition is met (i.e., reaching a “Goal” node or entering a “Pitfall” node).
– move(action): It moves the agent from the current node based on the given action. The action can be “UP” (0), “LEFT” (1), “DOWN” (2) or “RIGHT” (3). Besides, it returns the next node index, the transition reward, and whether the episode is done.
3
– get goals(): It finds the state indices of all goals and then returns them as a list.
• rl agents/RLAgent.py: “RLAgent” class in this file is an abstract class. Each RL algorithm must be a sub-class of it.
– Constructor(env, discount rate, action size): This constructor takes some necessary parameters and “Environment” object.
– train(**kwargs): This abstract method trains the RL agent with the corresponding RL algorithm.
– act(state, is training): This abstract method decides on an action that will be taken based on the given “state”. Note that the action decision can differ depending on whether it is called during training or validation.
– validate(): This method plays on the provided environment and returns a list of actions decided by the trained policy and the to- tal collected reward. Note that this method should be called after training.
• rl agents/QLearning.py: “QLearningAgent” class in this file must con- tain the implementation of the Q-Learning algorithm for this assignment. Also, it is the sub-class of the “RLAgent” class. This means that you must write your whole Q-Learning approach inside of the corresponding methods (i.e., train and act). Note that you can initiate some parame- ters/variables in the constructor.
• rl agents/SARSA.py: “SARSAAgent” class in this file must contain the implementation of the SARSA algorithm for this assignment. Also, it is the sub-class of the “RLAgent” class. This means that you must write your whole SARSA approach inside of the corresponding methods (i.e., train and act). Note that you can initiate some parameters/variables in the constructor.
• Main.py: This file has a script to run the algorithms and print some results, such as the total scores, elapsed time in microseconds, etc. You are free to edit this file if you want to add more analyses or RL algorithms. Although we will test your code with the original script, you can share your own “Main.py” file with us. To run this script, you should enter the name of the grid file, such as “Grid1.pkl” in the console. Besides, you must define the parameters you determine in this file in the constructor of the corresponding RL agent classes.
• grid generator.py: This file contains a script to generate a new scenario randomly. We provided this file so that you can test your algorithms with more scenarios. When you run this script, it will ask for some parameters in the console.
4

Briefly, you will implement “Q-Learning” and “SARSA” RL algorithms with Python programming language in the corresponding methods. Do not forget to read the comments in all methods.
3 Reinforcement Learning Algorithms
This section provides essential background information on Reinforcement Learn- ing (RL) and the algorithms you will be expected to implement for the given problem. RL is a machine learning category where an agent learns to make de- cisions in an environment by interacting with it. The agent observes the current state and selects an action, and then the environment responds with feedback called “reward” (as shown in Equation 1). The primary objective of RL is to maximize the cumulative reward (as illustrated in Equation 2).
st+1,rt ←Env(st,at) (1)
T
maximize(X rt) (2)
t
Reinforcement learning enables an agent to learn how to identify the best actions to take in different states, leading to more successful outcomes in the environment. The Q-value is a critical concept in RL, which indicates how good a particular state-action pair is. By using a Dynamic Programming approach, the Q-value function (Q(s, a)) can be defined as shown in Equation 3 (Bellman Optimality Equation), where γ is the discount rate.
Q(st,at)=rt +γ×rt+1 +γ2 ×rt+2 +γ3 ×rt+3 +... (3) = rt + γ × maxat+1 (Q(st+1, at+1))
π(st) = argmaxat (Q(st, at)) (4)
The Q-value function represents the expected cumulative discounted reward the agent can receive by taking a specific action in a particular state and fol- lowing its policy afterward. Note that the policy function can be defined as dis- played in Equation 4. The Q-value function is learned through the Q-learning process, where the Q-value estimates are updated based on the rewards the agent receives in response to its actions. By continually updating the Q-value estimates, the agent can make more informed decisions about which actions to take in different states, ultimately leading to more successful environmental outcomes.
For this assignment, you are expected to implement Q-Learning and SARSA algorithms for the given problem. Q-learning and SARSA are algorithms used in RL to find an optimal policy for an agent to take action in an environment. Both of these algorithms use the concept of Q-values to estimate the value of a
5

state-action pair. Q-learning (Algorithm 1) is an off-policy algorithm that learns the optimal Q-value function by updating the Q-value of the current state-action pair using the maximum Q-value of the next state. The agent selects actions based on the highest Q-value for the current state, which may not necessarily be the action taken by the agent. This makes Q-learning an off-policy algorithm, as the agent updates its Q-values based on the optimal action, not the action it took.

<img width="432" alt="Screenshot 2024-07-10 at 18 15 51" src="https://github.com/FaridGahramanov2/StrategicNavigations/assets/153610282/b72caa64-d031-4d1a-ba91-6622ea037986">


SARSA (Algorithm 2), on the other hand, is an on-policy algorithm that learns the Q-value function by updating the Q-value of the current state-action pair using the Q-value of the next state-action pair according to the policy the agent is following. The agent selects actions based on the policy it is follow- ing, which means that the Q-values are updated based on the action the agent actually takes.
The main difference between Q-learning and SARSA is that Q-learning is an off-policy algorithm, whereas SARSA is an on-policy algorithm. This difference affects how the algorithms update their Q-values and choose actions. In general, Q-learning is better suited for environments with high variance, where exploring and maximizing rewards are equally important. On the other hand, SARSA is better suited for environments with low variance, where the agent should focus more on maximizing rewards.
(π(s) with probability 1 − ε
a= (5)
random action
with probability ε
6
▷ Update ε
 

▷ Update ε
 On the other hand, in both Q-learning and SARSA, the agent’s behavior pol- icy is typically an epsilon-greedy approach during training. This means that the agent selects the best action with probability 1 − ε (exploitation) and a random action with probability ε (exploration), as defined in Equation 5. The epsilon- greedy approach is crucial because it allows the agent to explore and learn from different actions. Without exploration, the agent may not encounter all possi- ble state-action pairs and could miss out on learning optimal policies. However, exploration also comes at a cost since the agent may choose sub-optimal actions that lead to lower rewards. By balancing exploration and exploitation through the epsilon-greedy approach, the agent can learn an optimal policy while still exploring the environment. The value of ε typically decreases over time as the agent learns more about the environment. In the beginning, when the agent has little knowledge of the environment, it is essential to explore more to learn op- timal policies. As the agent’s knowledge grows, the exploitation term becomes more important, and the exploration term can be decreased. This approach is known as annealing epsilon-greedy, where the value of epsilon is annealed or gradually reduced over time.
For this assignment, Q-Learning and SARSA algorithms must be imple- mented; their performance must be compared in terms of computational time and total reward during the validation phase. Additionally, their performance depends on some hyper-parameters (e.g., maximum iteration, γ, εmin, εdecay,
7

and alpha). Thus, you are expected to tune these parameters and analyze the impact on the performance of these methods in your report. Some examples of graphs you can prepare for your report are in Section 5.



