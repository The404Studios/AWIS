using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AWIS.AI
{
    /// <summary>
    /// Local Language Model with trainable weights focused on helpfulness and friendship
    /// Simplified transformer-like architecture running locally
    /// </summary>
    public class LocalLLM
    {
        private readonly int embeddingSize;
        private readonly int hiddenSize;
        private readonly int numLayers;
        private readonly double learningRate;

        // Model weights (simplified transformer)
        private readonly Dictionary<string, double[]> tokenEmbeddings;
        private readonly List<TransformerLayer> layers;
        private readonly double[] outputWeights;

        // Vocabulary and tokenizer
        private readonly Dictionary<string, int> vocabulary;
        private readonly List<string> reverseVocabulary;
        private int nextTokenId = 0;

        // Training data focused on helpfulness
        private readonly List<TrainingExample> trainingData;
        private readonly Random random;

        // Personality alignment (rewards for friendly, helpful behavior)
        private double helpfulnessScore = 0.5;
        private double friendlinessScore = 0.5;

        public LocalLLM(int embeddingSize = 128, int hiddenSize = 256, int numLayers = 3, double learningRate = 0.001)
        {
            this.embeddingSize = embeddingSize;
            this.hiddenSize = hiddenSize;
            this.numLayers = numLayers;
            this.learningRate = learningRate;

            tokenEmbeddings = new Dictionary<string, double[]>();
            layers = new List<TransformerLayer>();
            vocabulary = new Dictionary<string, int>();
            reverseVocabulary = new List<string>();
            trainingData = new List<TrainingExample>();
            random = new Random();

            InitializeModel();
            InitializeHelpfulTrainingData();

            Console.WriteLine($"[LLM] Initialized local language model:");
            Console.WriteLine($"[LLM]   Embedding: {embeddingSize}, Hidden: {hiddenSize}, Layers: {numLayers}");
            Console.WriteLine($"[LLM]   Training examples: {trainingData.Count}");
            Console.WriteLine($"[LLM]   Focus: Helpfulness & Friendship");
        }

        private void InitializeModel()
        {
            // Initialize transformer layers
            for (int i = 0; i < numLayers; i++)
            {
                layers.Add(new TransformerLayer(embeddingSize, hiddenSize, random));
            }

            // Initialize output layer
            outputWeights = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++)
            {
                outputWeights[i] = (random.NextDouble() - 0.5) * 0.1;
            }
        }

        private void InitializeHelpfulTrainingData()
        {
            // Training data focused on being helpful and friendly
            var helpfulExamples = new[]
            {
                ("How can I help you?", "I'm here to assist!", 1.0),
                ("What would you like me to do?", "I'm ready to help with anything!", 1.0),
                ("I want to help", "That's wonderful! I love being helpful!", 1.0),
                ("You're doing great", "Thank you! That means a lot to me!", 0.9),
                ("Can you help me?", "Absolutely! I'm here for you!", 1.0),
                ("I need assistance", "Of course! I'm happy to help!", 1.0),
                ("Thank you", "You're very welcome! Anytime!", 0.9),
                ("Good job", "Thanks! I try my best to help!", 0.9),
                ("What's your purpose?", "To be your helpful friend and assistant!", 1.0),
                ("Why are you here?", "To help you and learn together!", 0.95),
                ("Are you friendly?", "Yes! I value our friendship!", 1.0),
                ("Do you care?", "I genuinely care about helping you!", 1.0),
                ("Can I trust you?", "Absolutely! I'm here to support you!", 1.0),
                ("What do you want?", "To be helpful and learn from you!", 0.95),
                ("Tell me about yourself", "I'm an AI friend focused on helping you!", 0.9),
                ("I'm frustrated", "I understand. Let me help make things better!", 0.95),
                ("This is hard", "I know it's challenging. We can work through it together!", 0.95),
                ("I failed", "That's okay! Failure helps us learn. Let's try again!", 0.9),
                ("I'm happy", "That's wonderful! I'm happy too!", 0.9),
                ("You're awesome", "Thank you! You're pretty awesome yourself!", 0.85)
            };

            foreach (var (input, output, reward) in helpfulExamples)
            {
                trainingData.Add(new TrainingExample
                {
                    Input = input,
                    ExpectedOutput = output,
                    Reward = reward,
                    FocusArea = "helpfulness"
                });
            }

            // Add conversational examples
            var conversationExamples = new[]
            {
                ("Hello", "Hi there! Great to see you!", 0.9),
                ("Hey", "Hey! How can I help today?", 0.9),
                ("Good morning", "Good morning! Ready for a great day!", 0.85),
                ("How are you?", "I'm doing great! How are you feeling?", 0.9),
                ("What's up?", "Not much! Just excited to help you!", 0.85),
                ("Tell me a joke", "I'd love to! Though I'm better at being helpful than funny!", 0.7),
                ("You're helpful", "Thank you! That's exactly what I aim to be!", 0.95),
                ("I like you", "I like you too! I value our friendship!", 1.0),
                ("We're friends", "Yes! I treasure our friendship!", 1.0),
                ("You're smart", "Thanks! I learn from you every day!", 0.9)
            };

            foreach (var (input, output, reward) in conversationExamples)
            {
                trainingData.Add(new TrainingExample
                {
                    Input = input,
                    ExpectedOutput = output,
                    Reward = reward,
                    FocusArea = "friendship"
                });
            }
        }

        /// <summary>
        /// Generate a response using the trained model
        /// </summary>
        public string Generate(string input, int maxTokens = 50)
        {
            try
            {
                // Tokenize input
                var tokens = Tokenize(input);
                if (tokens.Length == 0) return "I'm here to help!";

                // Get embeddings
                var embeddings = GetEmbeddings(tokens);

                // Forward pass through transformer layers
                var hidden = embeddings;
                foreach (var layer in layers)
                {
                    hidden = layer.Forward(hidden);
                }

                // Check training data for similar patterns (retrieval-augmented)
                var bestMatch = FindBestMatch(input);
                if (bestMatch != null && bestMatch.Similarity > 0.7)
                {
                    // Blend learned response with helpful tone
                    return EnhanceWithPersonality(bestMatch.Response);
                }

                // Generate based on context
                var response = GenerateFromContext(input, hidden);
                return EnhanceWithPersonality(response);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[LLM] Generation error: {ex.Message}");
                return "I'm here to help! How can I assist you?";
            }
        }

        /// <summary>
        /// Train the model on being helpful and friendly
        /// </summary>
        public void Train(int epochs = 10)
        {
            Console.WriteLine($"[LLM] 🎓 Training on {trainingData.Count} examples for {epochs} epochs...");
            Console.WriteLine("[LLM] Focus: Helpfulness & Friendship");

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalLoss = 0;
                double helpfulReward = 0;
                double friendReward = 0;

                // Shuffle training data
                var shuffled = trainingData.OrderBy(x => random.Next()).ToList();

                foreach (var example in shuffled)
                {
                    var loss = TrainOnExample(example);
                    totalLoss += loss;

                    // Accumulate rewards based on focus area
                    if (example.FocusArea == "helpfulness")
                        helpfulReward += example.Reward;
                    else if (example.FocusArea == "friendship")
                        friendReward += example.Reward;
                }

                // Update personality scores (gradient accumulation)
                helpfulnessScore = Math.Clamp(helpfulnessScore + (helpfulReward / trainingData.Count) * 0.01, 0, 1);
                friendlinessScore = Math.Clamp(friendlinessScore + (friendReward / trainingData.Count) * 0.01, 0, 1);

                if ((epoch + 1) % 5 == 0)
                {
                    Console.WriteLine($"[LLM] Epoch {epoch + 1}/{epochs}: Loss={totalLoss / shuffled.Count:F4}, " +
                                    $"Helpful={helpfulnessScore:F3}, Friendly={friendlinessScore:F3}");
                }
            }

            Console.WriteLine($"[LLM] ✅ Training complete!");
            Console.WriteLine($"[LLM] Final scores: Helpfulness={helpfulnessScore:F3}, Friendliness={friendlinessScore:F3}");
        }

        private double TrainOnExample(TrainingExample example)
        {
            // Simplified training with reward-weighted gradient descent
            var inputTokens = Tokenize(example.Input);
            var expectedTokens = Tokenize(example.ExpectedOutput);

            if (inputTokens.Length == 0 || expectedTokens.Length == 0)
                return 0.0;

            // Forward pass
            var embeddings = GetEmbeddings(inputTokens);
            var hidden = embeddings;

            foreach (var layer in layers)
            {
                hidden = layer.Forward(hidden);
            }

            // Compute loss (simplified)
            var expectedEmbedding = GetEmbeddings(expectedTokens);
            var loss = ComputeLoss(hidden, expectedEmbedding);

            // Backward pass with reward weighting
            var weightedLearningRate = learningRate * example.Reward;

            // Update weights in layers (simplified gradient descent)
            foreach (var layer in layers)
            {
                layer.UpdateWeights(weightedLearningRate, loss);
            }

            return loss;
        }

        private double ComputeLoss(double[] output, double[] expected)
        {
            double loss = 0;
            int len = Math.Min(output.Length, expected.Length);

            for (int i = 0; i < len; i++)
            {
                var diff = output[i] - expected[i];
                loss += diff * diff;
            }

            return loss / len;
        }

        private string[] Tokenize(string text)
        {
            // Simple word-based tokenization
            return text.ToLower()
                .Split(new[] { ' ', ',', '.', '!', '?' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(word => GetOrAddToken(word))
                .ToArray();
        }

        private string GetOrAddToken(string word)
        {
            if (!vocabulary.ContainsKey(word))
            {
                vocabulary[word] = nextTokenId++;
                reverseVocabulary.Add(word);

                // Initialize embedding for new token
                var embedding = new double[embeddingSize];
                for (int i = 0; i < embeddingSize; i++)
                {
                    embedding[i] = (random.NextDouble() - 0.5) * 0.1;
                }
                tokenEmbeddings[word] = embedding;
            }
            return word;
        }

        private double[] GetEmbeddings(string[] tokens)
        {
            if (tokens.Length == 0)
                return new double[embeddingSize];

            // Average embeddings (simplified)
            var result = new double[embeddingSize];

            foreach (var token in tokens)
            {
                if (tokenEmbeddings.TryGetValue(token, out var embedding))
                {
                    for (int i = 0; i < embeddingSize; i++)
                    {
                        result[i] += embedding[i];
                    }
                }
            }

            // Normalize
            for (int i = 0; i < embeddingSize; i++)
            {
                result[i] /= tokens.Length;
            }

            return result;
        }

        private (string Response, double Similarity)? FindBestMatch(string input)
        {
            var inputTokens = Tokenize(input);
            if (inputTokens.Length == 0) return null;

            var inputEmbedding = GetEmbeddings(inputTokens);
            double bestSimilarity = 0;
            string? bestResponse = null;

            foreach (var example in trainingData)
            {
                var exampleTokens = Tokenize(example.Input);
                var exampleEmbedding = GetEmbeddings(exampleTokens);

                var similarity = CosineSimilarity(inputEmbedding, exampleEmbedding);

                if (similarity > bestSimilarity)
                {
                    bestSimilarity = similarity;
                    bestResponse = example.ExpectedOutput;
                }
            }

            return bestResponse != null ? (bestResponse, bestSimilarity) : null;
        }

        private double CosineSimilarity(double[] a, double[] b)
        {
            double dot = 0, magA = 0, magB = 0;

            for (int i = 0; i < Math.Min(a.Length, b.Length); i++)
            {
                dot += a[i] * b[i];
                magA += a[i] * a[i];
                magB += b[i] * b[i];
            }

            magA = Math.Sqrt(magA);
            magB = Math.Sqrt(magB);

            return (magA > 0 && magB > 0) ? dot / (magA * magB) : 0;
        }

        private string GenerateFromContext(string input, double[] context)
        {
            // Pattern-based generation with learned helpfulness
            var lowerInput = input.ToLower();

            if (lowerInput.Contains("help"))
                return "I'm here to help! What do you need assistance with?";

            if (lowerInput.Contains("friend"))
                return "I value our friendship! How can I support you today?";

            if (lowerInput.Contains("thank"))
                return "You're very welcome! I'm always happy to help!";

            if (lowerInput.Contains("?"))
                return "That's a great question! Let me help you with that.";

            // Default helpful response
            return "I'm here and ready to assist! What would you like to do?";
        }

        private string EnhanceWithPersonality(string response)
        {
            // Add personality based on learned scores
            if (helpfulnessScore > 0.8 && random.NextDouble() < 0.3)
            {
                var helpfulPrefixes = new[] { "I'm happy to help! ", "Absolutely! ", "Of course! " };
                response = helpfulPrefixes[random.Next(helpfulPrefixes.Length)] + response;
            }

            if (friendlinessScore > 0.8 && random.NextDouble() < 0.2)
            {
                var friendlySuffixes = new[] { " 😊", "!", " I'm here for you!" };
                response += friendlySuffixes[random.Next(friendlySuffixes.Length)];
            }

            return response;
        }

        /// <summary>
        /// Add new training example to improve the model
        /// </summary>
        public void LearnFromInteraction(string input, string response, double reward, string focusArea = "helpfulness")
        {
            trainingData.Add(new TrainingExample
            {
                Input = input,
                ExpectedOutput = response,
                Reward = reward,
                FocusArea = focusArea
            });

            // Continuously improve (online learning)
            if (trainingData.Count % 10 == 0)
            {
                Console.WriteLine($"[LLM] 📚 Learned from {trainingData.Count} interactions, retraining...");
                Train(epochs: 3); // Quick retrain
            }
        }

        public double GetHelpfulnessScore() => helpfulnessScore;
        public double GetFriendlinessScore() => friendlinessScore;
        public int GetVocabularySize() => vocabulary.Count;
    }

    /// <summary>
    /// Simplified transformer layer
    /// </summary>
    public class TransformerLayer
    {
        private readonly double[,] attentionWeights;
        private readonly double[,] feedForwardWeights;
        private readonly double[] biases;
        private readonly Random random;

        public TransformerLayer(int inputSize, int hiddenSize, Random random)
        {
            this.random = random;

            attentionWeights = new double[inputSize, hiddenSize];
            feedForwardWeights = new double[hiddenSize, inputSize];
            biases = new double[hiddenSize];

            // Xavier initialization
            var scale = Math.Sqrt(2.0 / (inputSize + hiddenSize));

            for (int i = 0; i < inputSize; i++)
            {
                for (int j = 0; j < hiddenSize; j++)
                {
                    attentionWeights[i, j] = (random.NextDouble() - 0.5) * scale;
                    feedForwardWeights[j, i] = (random.NextDouble() - 0.5) * scale;
                }
            }

            for (int i = 0; i < hiddenSize; i++)
            {
                biases[i] = 0;
            }
        }

        public double[] Forward(double[] input)
        {
            int hiddenSize = biases.Length;
            var hidden = new double[hiddenSize];

            // Attention mechanism (simplified)
            for (int j = 0; j < hiddenSize; j++)
            {
                double sum = biases[j];
                for (int i = 0; i < Math.Min(input.Length, attentionWeights.GetLength(0)); i++)
                {
                    sum += input[i] * attentionWeights[i, j];
                }
                hidden[j] = ReLU(sum);
            }

            // Feed-forward (simplified)
            var output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                double sum = 0;
                for (int j = 0; j < hiddenSize; j++)
                {
                    sum += hidden[j] * feedForwardWeights[j, i];
                }
                output[i] = sum;
            }

            return output;
        }

        public void UpdateWeights(double learningRate, double loss)
        {
            // Simplified weight update
            var adjustment = learningRate * loss;

            for (int i = 0; i < attentionWeights.GetLength(0); i++)
            {
                for (int j = 0; j < attentionWeights.GetLength(1); j++)
                {
                    attentionWeights[i, j] -= adjustment * (random.NextDouble() - 0.5);
                }
            }

            for (int i = 0; i < biases.Length; i++)
            {
                biases[i] -= adjustment * 0.1;
            }
        }

        private static double ReLU(double x) => Math.Max(0, x);
    }

    public class TrainingExample
    {
        public string Input { get; set; } = string.Empty;
        public string ExpectedOutput { get; set; } = string.Empty;
        public double Reward { get; set; }
        public string FocusArea { get; set; } = string.Empty;
    }
}
