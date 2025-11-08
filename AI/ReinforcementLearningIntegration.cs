using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AWIS.Core;
using AWIS.MachineLearning;

namespace AWIS.AI;

/// <summary>
/// Reinforcement learning integration with reward loops and continuous learning
/// </summary>
public class ReinforcementLearningAgent : IReinforcementAgent, ISubsystem
{
    private readonly DQNAgent _dqnAgent;
    private readonly ExperienceReplayBuffer _replayBuffer;
    private readonly RewardShaper _rewardShaper;
    private readonly IEventBus? _eventBus;
    private bool _isInitialized;
    private readonly RLMetrics _metrics = new();

    public string Name => "ReinforcementLearningAgent";
    public bool IsInitialized => _isInitialized;

    public ReinforcementLearningAgent(int stateSize, int actionSize, IEventBus? eventBus = null)
    {
        _dqnAgent = new DQNAgent(stateSize, actionSize, hiddenSize: 128);
        _replayBuffer = new ExperienceReplayBuffer(capacity: 100000);
        _rewardShaper = new RewardShaper();
        _eventBus = eventBus;
    }

    public async Task InitializeAsync()
    {
        await Task.Run(() =>
        {
            _metrics.Reset();
            _isInitialized = true;
        });
    }

    public async Task ShutdownAsync()
    {
        _isInitialized = false;
        await Task.CompletedTask;
    }

    public async Task<HealthStatus> GetHealthAsync()
    {
        return await Task.FromResult(new HealthStatus
        {
            IsHealthy = _isInitialized,
            Status = _isInitialized ? "Learning" : "Not Initialized",
            Metrics = new Dictionary<string, object>
            {
                ["TotalSteps"] = _metrics.TotalSteps,
                ["TotalEpisodes"] = _metrics.TotalEpisodes,
                ["AverageReward"] = _metrics.AverageReward,
                ["ExplorationRate"] = _dqnAgent.Epsilon,
                ["BufferSize"] = _replayBuffer.Size
            }
        });
    }

    public async Task<int> SelectActionAsync(double[] state)
    {
        var action = _dqnAgent.ChooseAction(state);
        await Task.CompletedTask;
        return action;
    }

    public async Task UpdateAsync(double[] state, int action, double reward, double[] nextState, bool done)
    {
        // Shape reward based on context
        var shapedReward = _rewardShaper.ShapeReward(state, action, reward, nextState, done);

        // Store experience in replay buffer
        _replayBuffer.Add(new Experience
        {
            State = state,
            Action = action,
            Reward = shapedReward,
            NextState = nextState,
            Done = done
        });

        // Update metrics
        _metrics.RecordStep(shapedReward);

        if (done)
        {
            _metrics.RecordEpisode();

            // Publish learning event
            if (_eventBus != null)
            {
                await _eventBus.PublishAsync(new LearningCompletedEvent
                {
                    ModelName = Name,
                    FinalLoss = 0, // Would be computed from training
                    Accuracy = 0,
                    Epochs = _metrics.TotalEpisodes
                });
            }
        }

        // Train on mini-batch if buffer has enough samples
        if (_replayBuffer.Size >= 64)
        {
            _dqnAgent.Replay();
        }

        await Task.CompletedTask;
    }

    public async Task<AgentMetrics> GetMetricsAsync()
    {
        return await Task.FromResult(new AgentMetrics
        {
            AverageReward = _metrics.AverageReward,
            ExplorationRate = _dqnAgent.Epsilon,
            TotalEpisodes = _metrics.TotalEpisodes,
            TotalSteps = _metrics.TotalSteps
        });
    }

    public async Task LearnAsync(object input, object output)
    {
        // Not directly applicable for RL
        await Task.CompletedTask;
    }

    public async Task<double> GetConfidenceAsync(object input)
    {
        if (input is double[] state)
        {
            // Confidence based on Q-value magnitude
            var action = await SelectActionAsync(state);
            return 0.8; // Placeholder
        }
        return 0.0;
    }

    public async Task SaveModelAsync(string path)
    {
        // Save DQN weights
        await Task.CompletedTask;
    }

    public async Task LoadModelAsync(string path)
    {
        // Load DQN weights
        await Task.CompletedTask;
    }
}

/// <summary>
/// Experience replay buffer for stable RL training
/// </summary>
public class ExperienceReplayBuffer
{
    private readonly ConcurrentQueue<Experience> _buffer = new();
    private readonly int _capacity;
    private readonly Random _random = new();

    public int Size => _buffer.Count;

    public ExperienceReplayBuffer(int capacity = 100000)
    {
        _capacity = capacity;
    }

    public void Add(Experience experience)
    {
        _buffer.Enqueue(experience);

        // Remove old experiences if over capacity
        while (_buffer.Count > _capacity)
        {
            _buffer.TryDequeue(out _);
        }
    }

    public Experience[] Sample(int batchSize)
    {
        var allExperiences = _buffer.ToArray();
        if (allExperiences.Length <= batchSize)
            return allExperiences;

        // Random sampling
        return Enumerable.Range(0, batchSize)
            .Select(_ => allExperiences[_random.Next(allExperiences.Length)])
            .ToArray();
    }

    public void Clear()
    {
        _buffer.Clear();
    }
}

/// <summary>
/// Experience tuple for RL
/// </summary>
public class Experience
{
    public double[] State { get; set; } = Array.Empty<double>();
    public int Action { get; set; }
    public double Reward { get; set; }
    public double[] NextState { get; set; } = Array.Empty<double>();
    public bool Done { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public Dictionary<string, object> Metadata { get; set; } = new();
}

/// <summary>
/// Reward shaping for faster learning
/// </summary>
public class RewardShaper
{
    private readonly Dictionary<string, double> _shaping Weights = new()
    {
        ["progress"] = 0.1,
        ["efficiency"] = 0.2,
        ["safety"] = 0.5,
        ["novelty"] = 0.05
    };

    public double ShapeReward(double[] state, int action, double reward, double[] nextState, bool done)
    {
        double shapedReward = reward;

        // Progress reward (getting closer to goal)
        if (nextState.Length > 0 && state.Length > 0)
        {
            double progressReward = ComputeProgress(state, nextState);
            shapedReward += _shapingWeights["progress"] * progressReward;
        }

        // Efficiency reward (penalize time)
        shapedReward -= _shapingWeights["efficiency"] * 0.01;

        // Safety reward (avoid dangerous states)
        if (IsDangerousState(nextState))
        {
            shapedReward -= _shapingWeights["safety"];
        }

        // Novelty reward (encourage exploration)
        double noveltyReward = ComputeNovelty(state);
        shapedReward += _shapingWeights["novelty"] * noveltyReward;

        return shapedReward;
    }

    private double ComputeProgress(double[] state, double[] nextState)
    {
        // Simplified progress metric (distance to goal)
        // In practice, this would be domain-specific
        return 0.01;
    }

    private bool IsDangerousState(double[] state)
    {
        // Check if state violates safety constraints
        return state.Any(s => Math.Abs(s) > 10);
    }

    private double ComputeNovelty(double[] state)
    {
        // Measure how novel this state is
        // Could use state visitation counts or prediction error
        return 0.01;
    }
}

/// <summary>
/// RL metrics tracking
/// </summary>
public class RLMetrics
{
    private readonly List<double> _episodeRewards = new();
    private double _currentEpisodeReward;

    public int TotalSteps { get; private set; }
    public int TotalEpisodes { get; private set; }
    public double AverageReward => _episodeRewards.Count > 0 ? _episodeRewards.Average() : 0;
    public double BestReward => _episodeRewards.Count > 0 ? _episodeRewards.Max() : 0;

    public void RecordStep(double reward)
    {
        _currentEpisodeReward += reward;
        TotalSteps++;
    }

    public void RecordEpisode()
    {
        _episodeRewards.Add(_currentEpisodeReward);

        // Keep only last 100 episodes
        if (_episodeRewards.Count > 100)
        {
            _episodeRewards.RemoveAt(0);
        }

        _currentEpisodeReward = 0;
        TotalEpisodes++;
    }

    public void Reset()
    {
        _episodeRewards.Clear();
        _currentEpisodeReward = 0;
        TotalSteps = 0;
        TotalEpisodes = 0;
    }
}

/// <summary>
/// Multi-agent RL coordinator
/// </summary>
public class MultiAgentRLSystem
{
    private readonly List<ReinforcementLearningAgent> _agents = new();
    private readonly IEventBus _eventBus;

    public MultiAgentRLSystem(IEventBus eventBus)
    {
        _eventBus = eventBus;
    }

    public void AddAgent(ReinforcementLearningAgent agent)
    {
        _agents.Add(agent);
    }

    public async Task<int[]> SelectJointActionsAsync(double[][] states)
    {
        var tasks = states.Zip(_agents, (state, agent) => agent.SelectActionAsync(state));
        return await Task.WhenAll(tasks);
    }

    public async Task UpdateAllAgentsAsync(
        double[][] states,
        int[] actions,
        double[] rewards,
        double[][] nextStates,
        bool[] dones)
    {
        var updateTasks = new List<Task>();

        for (int i = 0; i < _agents.Count; i++)
        {
            updateTasks.Add(_agents[i].UpdateAsync(states[i], actions[i], rewards[i], nextStates[i], dones[i]));
        }

        await Task.WhenAll(updateTasks);
    }
}

/// <summary>
/// Curiosity-driven learning module
/// </summary>
public class CuriosityModule
{
    private readonly DeepNeuralNetwork _forwardModel;
    private readonly DeepNeuralNetwork _inverseModel;
    private readonly double _intrinsicRewardScale;

    public CuriosityModule(int stateSize, int actionSize, double intrinsicRewardScale = 0.1)
    {
        _intrinsicRewardScale = intrinsicRewardScale;

        // Forward model: predicts next state given current state and action
        _forwardModel = new DeepNeuralNetwork();
        _forwardModel.AddLayer(stateSize + actionSize, 128, "relu");
        _forwardModel.AddLayer(128, 64, "relu");
        _forwardModel.AddLayer(64, stateSize, "linear");

        // Inverse model: predicts action given current and next state
        _inverseModel = new DeepNeuralNetwork();
        _inverseModel.AddLayer(stateSize * 2, 128, "relu");
        _inverseModel.AddLayer(128, 64, "relu");
        _inverseModel.AddLayer(64, actionSize, "softmax");
    }

    public double ComputeIntrinsicReward(double[] state, int action, double[] nextState)
    {
        // Predict next state
        var input = state.Concat(OneHot(action, _forwardModel.Layers[0].InputSize - state.Length)).ToArray();
        var predictedNextState = _forwardModel.Predict(input);

        // Intrinsic reward is prediction error
        double predictionError = predictedNextState.Zip(nextState, (p, a) => Math.Pow(p - a, 2)).Sum();

        return _intrinsicRewardScale * predictionError;
    }

    public void Train(double[] state, int action, double[] nextState)
    {
        // Train forward model
        var forwardInput = state.Concat(OneHot(action, _forwardModel.Layers[0].InputSize - state.Length)).ToArray();
        var forwardTarget = nextState.Select(x => new[] { x }).ToArray();
        _forwardModel.Train(new[] { forwardInput }, forwardTarget.Select(t => t).ToArray(), epochs: 1);

        // Train inverse model
        var inverseInput = state.Concat(nextState).ToArray();
        var inverseTarget = OneHot(action, _inverseModel.Layers.Last().OutputSize);
        _inverseModel.Train(new[] { inverseInput }, new[] { inverseTarget }, epochs: 1);
    }

    private double[] OneHot(int index, int size)
    {
        var oneHot = new double[size];
        if (index < size)
            oneHot[index] = 1.0;
        return oneHot;
    }
}

/// <summary>
/// Hierarchical RL with options framework
/// </summary>
public class HierarchicalRLAgent
{
    private readonly ReinforcementLearningAgent _metaController;
    private readonly Dictionary<int, ReinforcementLearningAgent> _options = new();
    private int _currentOption = -1;

    public HierarchicalRLAgent(int stateSize, int numOptions, int actionsPerOption, IEventBus? eventBus = null)
    {
        // Meta-controller selects options
        _metaController = new ReinforcementLearningAgent(stateSize, numOptions, eventBus);

        // Each option is a sub-policy
        for (int i = 0; i < numOptions; i++)
        {
            _options[i] = new ReinforcementLearningAgent(stateSize, actionsPerOption, eventBus);
        }
    }

    public async Task<int> SelectActionAsync(double[] state)
    {
        // Meta-controller selects option if needed
        if (_currentOption == -1 || ShouldTerminateOption(state))
        {
            _currentOption = await _metaController.SelectActionAsync(state);
        }

        // Selected option chooses primitive action
        return await _options[_currentOption].SelectActionAsync(state);
    }

    public async Task UpdateAsync(double[] state, int action, double reward, double[] nextState, bool done)
    {
        // Update current option
        if (_currentOption >= 0)
        {
            await _options[_currentOption].UpdateAsync(state, action, reward, nextState, done);
        }

        // Update meta-controller when option terminates
        if (ShouldTerminateOption(nextState) || done)
        {
            await _metaController.UpdateAsync(state, _currentOption, reward, nextState, done);
            _currentOption = -1;
        }
    }

    private bool ShouldTerminateOption(double[] state)
    {
        // Termination condition for options (simplified)
        // In practice, could use learned termination functions
        return new Random().NextDouble() < 0.1;
    }
}

/// <summary>
/// Imitation learning from demonstrations
/// </summary>
public class ImitationLearner
{
    private readonly DeepNeuralNetwork _policy;
    private readonly List<(double[] State, int Action)> _demonstrations = new();

    public ImitationLearner(int stateSize, int actionSize)
    {
        _policy = new DeepNeuralNetwork();
        _policy.AddLayer(stateSize, 128, "relu");
        _policy.AddLayer(128, 64, "relu");
        _policy.AddLayer(64, actionSize, "softmax");
    }

    public void AddDemonstration(double[] state, int action)
    {
        _demonstrations.Add((state, action));
    }

    public void Train(int epochs = 100)
    {
        if (_demonstrations.Count == 0)
            return;

        var states = _demonstrations.Select(d => d.State).ToArray();
        var actions = _demonstrations.Select(d =>
        {
            var oneHot = new double[_policy.Layers.Last().OutputSize];
            oneHot[d.Action] = 1.0;
            return oneHot;
        }).ToArray();

        _policy.Train(states, actions, epochs);
    }

    public int PredictAction(double[] state)
    {
        var output = _policy.Predict(state);
        return output.Select((prob, i) => (prob, i)).OrderByDescending(t => t.prob).First().i;
    }

    public void Clear()
    {
        _demonstrations.Clear();
    }
}

/// <summary>
/// Reward-driven autonomous learning system
/// </summary>
public class AutonomousLearningSystem
{
    private readonly ReinforcementLearningAgent _agent;
    private readonly CuriosityModule _curiosity;
    private readonly ImitationLearner _imitationLearner;
    private readonly IEventBus _eventBus;
    private readonly IMemorySystem _memory;

    public AutonomousLearningSystem(
        int stateSize,
        int actionSize,
        IEventBus eventBus,
        IMemorySystem memory)
    {
        _agent = new ReinforcementLearningAgent(stateSize, actionSize, eventBus);
        _curiosity = new CuriosityModule(stateSize, actionSize);
        _imitationLearner = new ImitationLearner(stateSize, actionSize);
        _eventBus = eventBus;
        _memory = memory;
    }

    public async Task<int> DecideActionAsync(double[] state)
    {
        // First try imitation if available
        if (HasDemonstrations())
        {
            return _imitationLearner.PredictAction(state);
        }

        // Otherwise use RL
        return await _agent.SelectActionAsync(state);
    }

    public async Task LearnFromExperienceAsync(double[] state, int action, double externalReward, double[] nextState, bool done)
    {
        // Compute intrinsic reward from curiosity
        double intrinsicReward = _curiosity.ComputeIntrinsicReward(state, action, nextState);

        // Combined reward
        double totalReward = externalReward + intrinsicReward;

        // Update RL agent
        await _agent.UpdateAsync(state, action, totalReward, nextState, done);

        // Train curiosity module
        _curiosity.Train(state, action, nextState);

        // Store in memory
        await _memory.StoreAsync(
            $"Action {action} in state {string.Join(",", state.Select(s => s.ToString("F2")))} => reward {totalReward:F2}",
            MemoryType.Episodic,
            importance: Math.Abs(totalReward));

        // Publish event
        await _eventBus.PublishAsync(new ActionExecutedEvent
        {
            ActionType = (ActionType)action,
            Success = totalReward > 0,
            ExecutionTime = 0
        });
    }

    public void AddDemonstration(double[] state, int action)
    {
        _imitationLearner.AddDemonstration(state, action);
    }

    public void TrainFromDemonstrations(int epochs = 100)
    {
        _imitationLearner.Train(epochs);
    }

    private bool HasDemonstrations()
    {
        // Check if we have learned from demonstrations
        return false; // Placeholder
    }
}
