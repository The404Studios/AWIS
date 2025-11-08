using System;
using System.Collections.Generic;
using System.Linq;

namespace AWIS.MachineLearning
{
    /// <summary>
    /// Q-Learning agent for reinforcement learning
    /// </summary>
    public class QLearningAgent
    {
        private readonly Dictionary<string, Dictionary<int, double>> qTable = new();
        private readonly double learningRate;
        private readonly double discountFactor;
        private readonly double explorationRate;
        private readonly Random random = new();

        public QLearningAgent(double learningRate = 0.1, double discountFactor = 0.95, double explorationRate = 0.1)
        {
            this.learningRate = learningRate;
            this.discountFactor = discountFactor;
            this.explorationRate = explorationRate;
        }

        public int ChooseAction(string state, int numActions)
        {
            if (random.NextDouble() < explorationRate)
                return random.Next(numActions); // Explore

            if (!qTable.ContainsKey(state))
                InitializeState(state, numActions);

            return qTable[state].OrderByDescending(kv => kv.Value).First().Key; // Exploit
        }

        public void Learn(string state, int action, double reward, string nextState, int numActions)
        {
            if (!qTable.ContainsKey(state))
                InitializeState(state, numActions);
            if (!qTable.ContainsKey(nextState))
                InitializeState(nextState, numActions);

            double currentQ = qTable[state][action];
            double maxNextQ = qTable[nextState].Values.Max();
            double newQ = currentQ + learningRate * (reward + discountFactor * maxNextQ - currentQ);
            qTable[state][action] = newQ;
        }

        private void InitializeState(string state, int numActions)
        {
            qTable[state] = new Dictionary<int, double>();
            for (int a = 0; a < numActions; a++)
                qTable[state][a] = 0.0;
        }

        public Dictionary<string, Dictionary<int, double>> GetQTable() => qTable;
    }

    /// <summary>
    /// Deep Q-Network (DQN) for more complex environments
    /// </summary>
    public class DQNAgent
    {
        private readonly DeepNeuralNetwork network;
        private readonly List<(double[] state, int action, double reward, double[] nextState)> replayBuffer = new();
        private readonly int bufferSize;
        private readonly int batchSize;
        private readonly double explorationRate;
        private readonly Random random = new();

        public DQNAgent(int stateSize, int actionSize, int hiddenSize = 64, int bufferSize = 10000, int batchSize = 32)
        {
            network = new DeepNeuralNetwork();
            network.AddLayer(stateSize, hiddenSize, "relu");
            network.AddLayer(hiddenSize, hiddenSize, "relu");
            network.AddLayer(hiddenSize, actionSize, "linear");

            this.bufferSize = bufferSize;
            this.batchSize = batchSize;
            this.explorationRate = 0.1;
        }

        public int ChooseAction(double[] state)
        {
            if (random.NextDouble() < explorationRate)
                return random.Next(network.Predict(state).Length);

            var qValues = network.Predict(state);
            return Array.IndexOf(qValues, qValues.Max());
        }

        public void Remember(double[] state, int action, double reward, double[] nextState)
        {
            replayBuffer.Add((state, action, reward, nextState));
            if (replayBuffer.Count > bufferSize)
                replayBuffer.RemoveAt(0);
        }

        public void Replay()
        {
            if (replayBuffer.Count < batchSize) return;

            var batch = replayBuffer.OrderBy(x => random.Next()).Take(batchSize).ToArray();
            var states = batch.Select(x => x.state).ToArray();
            var targets = new double[batchSize][];

            for (int i = 0; i < batchSize; i++)
            {
                var (state, action, reward, nextState) = batch[i];
                var qValues = network.Predict(state);
                var nextQValues = network.Predict(nextState);
                qValues[action] = reward + 0.95 * nextQValues.Max();
                targets[i] = qValues;
            }

            network.Train(states, targets, epochs: 1);
        }
    }

    /// <summary>
    /// Policy gradient agent (REINFORCE algorithm)
    /// </summary>
    public class PolicyGradientAgent
    {
        private readonly DeepNeuralNetwork policyNetwork;
        private readonly List<(double[] state, int action, double reward)> episode = new();
        private readonly double learningRate = 0.01;

        public PolicyGradientAgent(int stateSize, int actionSize, int hiddenSize = 64)
        {
            policyNetwork = new DeepNeuralNetwork();
            policyNetwork.AddLayer(stateSize, hiddenSize, "relu");
            policyNetwork.AddLayer(hiddenSize, actionSize, "sigmoid");
        }

        public int SampleAction(double[] state)
        {
            var probabilities = policyNetwork.Predict(state);
            var sum = probabilities.Sum();
            probabilities = probabilities.Select(p => p / sum).ToArray();

            double r = new Random().NextDouble();
            double cumulative = 0;
            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                if (r < cumulative) return i;
            }
            return probabilities.Length - 1;
        }

        public void StoreTransition(double[] state, int action, double reward)
        {
            episode.Add((state, action, reward));
        }

        public void Learn()
        {
            if (episode.Count == 0) return;

            // Compute discounted returns
            double[] returns = new double[episode.Count];
            double G = 0;
            for (int t = episode.Count - 1; t >= 0; t--)
            {
                G = episode[t].reward + 0.99 * G;
                returns[t] = G;
            }

            // Normalize returns
            double mean = returns.Average();
            double std = Math.Sqrt(returns.Select(r => Math.Pow(r - mean, 2)).Average());
            returns = returns.Select(r => (r - mean) / (std + 1e-8)).ToArray();

            // Update policy (simplified)
            var states = episode.Select(e => e.state).ToArray();
            var targets = new double[episode.Count][];
            for (int i = 0; i < episode.Count; i++)
            {
                var probs = policyNetwork.Predict(states[i]);
                probs[episode[i].action] += learningRate * returns[i];
                targets[i] = probs;
            }

            policyNetwork.Train(states, targets, epochs: 1);
            episode.Clear();
        }
    }

    /// <summary>
    /// Actor-Critic agent combining value and policy methods
    /// </summary>
    public class ActorCriticAgent
    {
        private readonly DeepNeuralNetwork actorNetwork;
        private readonly DeepNeuralNetwork criticNetwork;
        private readonly double actorLearningRate = 0.001;
        private readonly double criticLearningRate = 0.005;

        public ActorCriticAgent(int stateSize, int actionSize, int hiddenSize = 64)
        {
            // Actor network (policy)
            actorNetwork = new DeepNeuralNetwork();
            actorNetwork.AddLayer(stateSize, hiddenSize, "relu");
            actorNetwork.AddLayer(hiddenSize, actionSize, "sigmoid");

            // Critic network (value function)
            criticNetwork = new DeepNeuralNetwork();
            criticNetwork.AddLayer(stateSize, hiddenSize, "relu");
            criticNetwork.AddLayer(hiddenSize, 1, "linear");
        }

        public int SelectAction(double[] state)
        {
            var probabilities = actorNetwork.Predict(state);
            var sum = probabilities.Sum();
            probabilities = probabilities.Select(p => p / sum).ToArray();

            double r = new Random().NextDouble();
            double cumulative = 0;
            for (int i = 0; i < probabilities.Length; i++)
            {
                cumulative += probabilities[i];
                if (r < cumulative) return i;
            }
            return probabilities.Length - 1;
        }

        public void Update(double[] state, int action, double reward, double[] nextState)
        {
            // Critic update (TD error)
            double value = criticNetwork.Predict(state)[0];
            double nextValue = criticNetwork.Predict(nextState)[0];
            double tdError = reward + 0.99 * nextValue - value;

            // Update critic
            criticNetwork.Train(new[] { state }, new[] { new[] { value + criticLearningRate * tdError } }, epochs: 1);

            // Update actor using TD error as advantage
            var actionProbs = actorNetwork.Predict(state);
            actionProbs[action] += actorLearningRate * tdError;
            actorNetwork.Train(new[] { state }, new[] { actionProbs }, epochs: 1);
        }
    }

    /// <summary>
    /// Monte Carlo Tree Search for planning
    /// </summary>
    public class MCTSNode
    {
        public string State { get; set; } = "";
        public int Visits { get; set; }
        public double Value { get; set; }
        public MCTSNode? Parent { get; set; }
        public List<MCTSNode> Children { get; set; } = new();
        public int Action { get; set; }

        public double UCB1(double c = 1.41)
        {
            if (Visits == 0) return double.MaxValue;
            return Value / Visits + c * Math.Sqrt(Math.Log(Parent?.Visits ?? 1) / Visits);
        }
    }

    public class MCTS
    {
        private readonly int simulations;
        private readonly Random random = new();

        public MCTS(int simulations = 1000)
        {
            this.simulations = simulations;
        }

        public int Search(string rootState, Func<string, int, (string nextState, double reward, bool done)> simulator, int numActions)
        {
            var root = new MCTSNode { State = rootState };

            for (int i = 0; i < simulations; i++)
            {
                // Selection
                var node = root;
                while (node.Children.Count > 0)
                {
                    node = node.Children.OrderByDescending(c => c.UCB1()).First();
                }

                // Expansion
                if (node.Visits > 0)
                {
                    for (int a = 0; a < numActions; a++)
                    {
                        var (nextState, _, _) = simulator(node.State, a);
                        node.Children.Add(new MCTSNode
                        {
                            State = nextState,
                            Parent = node,
                            Action = a
                        });
                    }
                    node = node.Children[random.Next(node.Children.Count)];
                }

                // Simulation (rollout)
                double totalReward = Rollout(node.State, simulator, numActions);

                // Backpropagation
                while (node != null)
                {
                    node.Visits++;
                    node.Value += totalReward;
                    node = node.Parent;
                }
            }

            return root.Children.OrderByDescending(c => c.Visits).First().Action;
        }

        private double Rollout(string state, Func<string, int, (string, double, bool)> simulator, int numActions, int maxSteps = 50)
        {
            double totalReward = 0;
            for (int step = 0; step < maxSteps; step++)
            {
                int action = random.Next(numActions);
                var (nextState, reward, done) = simulator(state, action);
                totalReward += reward;
                if (done) break;
                state = nextState;
            }
            return totalReward;
        }
    }

    /// <summary>
    /// Multi-armed bandit algorithms
    /// </summary>
    public class EpsilonGreedyBandit
    {
        private readonly int numArms;
        private readonly double epsilon;
        private readonly double[] qValues;
        private readonly int[] armCounts;
        private readonly Random random = new();

        public EpsilonGreedyBandit(int numArms, double epsilon = 0.1)
        {
            this.numArms = numArms;
            this.epsilon = epsilon;
            this.qValues = new double[numArms];
            this.armCounts = new int[numArms];
        }

        public int SelectArm()
        {
            if (random.NextDouble() < epsilon)
                return random.Next(numArms);
            return Array.IndexOf(qValues, qValues.Max());
        }

        public void Update(int arm, double reward)
        {
            armCounts[arm]++;
            qValues[arm] += (reward - qValues[arm]) / armCounts[arm];
        }
    }

    public class UCBBandit
    {
        private readonly int numArms;
        private readonly double c;
        private readonly double[] qValues;
        private readonly int[] armCounts;
        private int totalCount;

        public UCBBandit(int numArms, double c = 2.0)
        {
            this.numArms = numArms;
            this.c = c;
            this.qValues = new double[numArms];
            this.armCounts = new int[numArms];
        }

        public int SelectArm()
        {
            // Try each arm once first
            for (int i = 0; i < numArms; i++)
                if (armCounts[i] == 0) return i;

            double[] ucbValues = new double[numArms];
            for (int i = 0; i < numArms; i++)
            {
                ucbValues[i] = qValues[i] + c * Math.Sqrt(Math.Log(totalCount) / armCounts[i]);
            }
            return Array.IndexOf(ucbValues, ucbValues.Max());
        }

        public void Update(int arm, double reward)
        {
            armCounts[arm]++;
            totalCount++;
            qValues[arm] += (reward - qValues[arm]) / armCounts[arm];
        }
    }

    /// <summary>
    /// Temporal Difference Learning
    /// </summary>
    public class TDLearning
    {
        private readonly Dictionary<string, double> values = new();
        private readonly double learningRate;
        private readonly double discountFactor;

        public TDLearning(double learningRate = 0.1, double discountFactor = 0.9)
        {
            this.learningRate = learningRate;
            this.discountFactor = discountFactor;
        }

        public double GetValue(string state)
        {
            return values.ContainsKey(state) ? values[state] : 0.0;
        }

        public void Update(string state, double reward, string nextState)
        {
            double currentValue = GetValue(state);
            double nextValue = GetValue(nextState);
            double tdTarget = reward + discountFactor * nextValue;
            double tdError = tdTarget - currentValue;
            values[state] = currentValue + learningRate * tdError;
        }
    }
}
