using System;
using System.Collections.Generic;
using System.Runtime.Versioning;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Linq;
using AWIS.Core;
using AWIS.NLP;
using AWIS.MachineLearning;
using AWIS.AI;

namespace AWIS
{
    /// <summary>
    /// Main entry point for AWIS v8.0 - Advanced AI System
    /// Coordinates all AI/ML systems with parallel processing
    /// </summary>
    class Program
    {
        private static ParallelSystemCoordinator coordinator = null!;
        private static ParallelPerformanceMonitor performanceMonitor = null!;

        static async Task Main(string[] args)
        {
            PrintWelcomeBanner();

            // Run autonomous agent by default (or with --agent flag)
            if (args.Length == 0 || (args.Length > 0 && args[0] == "--agent"))
            {
                await RunAutonomousAgent();
                return;
            }

            // Initialize parallel coordinator for demos
            coordinator = new ParallelSystemCoordinator(Environment.ProcessorCount);
            performanceMonitor = new ParallelPerformanceMonitor();

            Console.WriteLine($"Initializing with {Environment.ProcessorCount} parallel workers...\n");

            // Run demonstration
            if (args[0] == "--demo")
            {
                await RunDemonstrationAsync();
            }
            else if (args[0] == "--full-demo")
            {
                await SystemDemo.RunFullSystemDemo();
            }
            else if (args[0] == "--test-tokenizer")
            {
                TestTokenizer();
            }
            else if (args[0] == "--benchmark")
            {
                await RunBenchmarkAsync();
            }
            else if (args[0] == "--ml-demo")
            {
                RunMLDemos();
            }
            else if (args[0] == "--menu")
            {
                ShowMenu();
            }

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadKey();
        }

        static void PrintWelcomeBanner()
        {
            Console.Clear();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("╔════════════════════════════════════════════════════════════════╗");
            Console.WriteLine("║          AWIS v8.0 - Advanced Artificial Intelligence          ║");
            Console.WriteLine("║                    Autonomous Agent Mode                       ║");
            Console.WriteLine("║                      20,000+ Lines of Code                     ║");
            Console.WriteLine("╚════════════════════════════════════════════════════════════════╝");
            Console.ResetColor();
            Console.WriteLine();
        }

        [SupportedOSPlatform("windows")]
        static async Task RunAutonomousAgent()
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("Initializing Autonomous AI Agent...\n");
            Console.ResetColor();

            AutonomousAgent? agent = null;

            try
            {
                agent = new AutonomousAgent();
                agent.Start();

                Console.WriteLine("\nAgent is now running. Type commands or speak them:");
                Console.WriteLine("  Examples:");
                Console.WriteLine("    - 'start recording'");
                Console.WriteLine("    - 'stop recording'");
                Console.WriteLine("    - 'repeat what I did'");
                Console.WriteLine("    - 'fight that crab'");
                Console.WriteLine("    - 'run away'");
                Console.WriteLine("    - 'follow player'");
                Console.WriteLine("\nType 'quit' to exit.\n");

                // Main command loop
                while (true)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.Write("> ");
                    Console.ResetColor();

                    var command = Console.ReadLine();
                    if (string.IsNullOrWhiteSpace(command))
                        continue;

                    if (command.ToLower() == "quit" || command.ToLower() == "exit")
                    {
                        Console.WriteLine("Shutting down agent...");
                        break;
                    }

                    // Process the command
                    agent.ProcessCommand(command);

                    await Task.Delay(100);
                }
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"\nError: {ex.Message}");
                Console.WriteLine("Note: This mode requires Windows and proper permissions for input control.");
                Console.ResetColor();
            }
            finally
            {
                agent?.Dispose();
            }
        }

        static void ShowMenu()
        {
            Console.WriteLine("Available Features:\n");
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("  Neural Networks:");
            Console.ResetColor();
            Console.WriteLine("    ✓ Transformers with Multi-Head Attention");
            Console.WriteLine("    ✓ Graph Neural Networks (Message Passing)");
            Console.WriteLine("    ✓ Capsule Networks with Dynamic Routing");
            Console.WriteLine("    ✓ Recurrent Networks (LSTM, GRU, Bidirectional)");
            Console.WriteLine("    ✓ Convolutional Networks (ResNet, DenseNet)");
            Console.WriteLine("    ✓ Neural ODEs & Memory-Augmented Networks");
            Console.WriteLine();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("  Generative Models:");
            Console.ResetColor();
            Console.WriteLine("    ✓ Variational Autoencoders (VAE)");
            Console.WriteLine("    ✓ Generative Adversarial Networks (GAN)");
            Console.WriteLine("    ✓ Diffusion Models (DDPM, Latent Diffusion)");
            Console.WriteLine();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("  Reinforcement Learning:");
            Console.ResetColor();
            Console.WriteLine("    ✓ PPO, SAC, TD3, A3C");
            Console.WriteLine("    ✓ Actor-Critic Methods");
            Console.WriteLine("    ✓ World Models & Predictive Coding");
            Console.WriteLine();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("  Natural Language Processing:");
            Console.ResetColor();
            Console.WriteLine("    ✓ BPE Tokenizer with Compression");
            Console.WriteLine("    ✓ WordPiece Tokenizer");
            Console.WriteLine("    ✓ Compressed Tokenizer (Huffman Coding)");
            Console.WriteLine("    ✓ SentencePiece Tokenizer");
            Console.WriteLine("    ✓ Word Embeddings & Text Summarization");
            Console.WriteLine("    ✓ NER & Dependency Parsing");
            Console.WriteLine("    ✓ Sequence-to-Sequence with Attention");
            Console.WriteLine();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("  Computer Vision:");
            Console.ResetColor();
            Console.WriteLine("    ✓ Object Detection (YOLO, R-CNN, FPN)");
            Console.WriteLine("    ✓ Image Segmentation (Watershed, Mean Shift)");
            Console.WriteLine("    ✓ Edge Detection (Canny, Sobel)");
            Console.WriteLine("    ✓ Image Filtering & Morphology");
            Console.WriteLine();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("  Machine Learning:");
            Console.ResetColor();
            Console.WriteLine("    ✓ Decision Trees & Random Forests");
            Console.WriteLine("    ✓ Gradient Boosting");
            Console.WriteLine("    ✓ Support Vector Machines");
            Console.WriteLine("    ✓ Clustering (K-Means, DBSCAN, Hierarchical)");
            Console.WriteLine("    ✓ Dimensionality Reduction (PCA, t-SNE)");
            Console.WriteLine("    ✓ Time Series Analysis (ARIMA, Forecasting)");
            Console.WriteLine();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("  Parallel Processing:");
            Console.ResetColor();
            Console.WriteLine("    ✓ Multi-threaded Task Execution");
            Console.WriteLine("    ✓ Distributed Task Coordinator");
            Console.WriteLine("    ✓ Batch Processing Pipeline");
            Console.WriteLine("    ✓ Performance Monitoring");
            Console.WriteLine();

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("\nUsage:");
            Console.ResetColor();
            Console.WriteLine("  dotnet run --demo           Run parallel processing demonstration");
            Console.WriteLine("  dotnet run --full-demo      Run complete system demonstration");
            Console.WriteLine("  dotnet run --ml-demo        Run machine learning demonstrations");
            Console.WriteLine("  dotnet run --test-tokenizer Test tokenizer with compression");
            Console.WriteLine("  dotnet run --benchmark      Run performance benchmark");
            Console.WriteLine();
        }

        static void TestTokenizer()
        {
            Console.WriteLine("=== Tokenizer Compression Demo ===\n");

            // Test data
            var corpus = new List<string>
            {
                "The quick brown fox jumps over the lazy dog",
                "Machine learning is a subset of artificial intelligence",
                "Natural language processing enables computers to understand human language",
                "Deep learning uses neural networks with multiple layers",
                "Reinforcement learning learns through trial and error"
            };

            Console.WriteLine("Training corpus:");
            foreach (var text in corpus)
            {
                Console.WriteLine($"  - {text}");
            }
            Console.WriteLine();

            // Test BPE Tokenizer
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("1. BPE Tokenizer:");
            Console.ResetColor();
            var bpe = new BPETokenizer(500);
            var sw = Stopwatch.StartNew();
            bpe.Train(corpus);
            sw.Stop();
            Console.WriteLine($"   Training time: {sw.ElapsedMilliseconds}ms");

            var testText = "Machine learning enables intelligent systems";
            var encoded = bpe.Encode(testText);
            var decoded = bpe.Decode(encoded);
            Console.WriteLine($"   Original: {testText}");
            Console.WriteLine($"   Tokens: [{string.Join(", ", encoded)}]");
            Console.WriteLine($"   Decoded: {decoded}");
            Console.WriteLine();

            // Test Compressed Tokenizer
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("2. Compressed Tokenizer (Huffman Coding):");
            Console.ResetColor();
            var compressed = new CompressedTokenizer(500);
            sw.Restart();
            compressed.Train(corpus);
            sw.Stop();
            Console.WriteLine($"   Training time: {sw.ElapsedMilliseconds}ms");

            var compressedBytes = compressed.EncodeCompressed(testText);
            var decompressed = compressed.DecodeCompressed(compressedBytes);
            var ratio = compressed.GetCompressionRatio(testText);

            Console.WriteLine($"   Original: {testText}");
            Console.WriteLine($"   Original size: {System.Text.Encoding.UTF8.GetByteCount(testText)} bytes");
            Console.WriteLine($"   Compressed size: {compressedBytes.Length} bytes");
            Console.WriteLine($"   Compression ratio: {ratio:P2}");
            Console.WriteLine($"   Decompressed: {decompressed}");
            Console.WriteLine();

            // Test WordPiece Tokenizer
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("3. WordPiece Tokenizer:");
            Console.ResetColor();
            var wordPiece = new WordPieceTokenizer();
            sw.Restart();
            wordPiece.BuildVocabulary(corpus, 1000);
            sw.Stop();
            Console.WriteLine($"   Training time: {sw.ElapsedMilliseconds}ms");

            var wpTokens = wordPiece.Tokenize(testText);
            var wpDecoded = wordPiece.Detokenize(wpTokens);
            Console.WriteLine($"   Original: {testText}");
            Console.WriteLine($"   Tokens: [{string.Join(", ", wpTokens)}]");
            Console.WriteLine($"   Decoded: {wpDecoded}");
            Console.WriteLine();

            // Test SentencePiece Tokenizer
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("4. SentencePiece Tokenizer:");
            Console.ResetColor();
            var sentencePiece = new SentencePieceTokenizer(500);
            sw.Restart();
            sentencePiece.Train(corpus);
            sw.Stop();
            Console.WriteLine($"   Training time: {sw.ElapsedMilliseconds}ms");

            var spTokens = sentencePiece.Encode(testText);
            var spDecoded = sentencePiece.Decode(spTokens);
            Console.WriteLine($"   Original: {testText}");
            Console.WriteLine($"   Tokens: [{string.Join(", ", spTokens)}]");
            Console.WriteLine($"   Decoded: {spDecoded}");
            Console.WriteLine();
        }

        static async Task RunDemonstrationAsync()
        {
            Console.WriteLine("=== AWIS Parallel Processing Demonstration ===\n");

            // Simulate parallel AI tasks
            var tasks = new Dictionary<string, Func<Task<string>>>
            {
                ["NLP Processing"] = async () =>
                {
                    await Task.Delay(100);
                    return "NLP: Processed 1000 sentences";
                },
                ["Computer Vision"] = async () =>
                {
                    await Task.Delay(150);
                    return "CV: Analyzed 500 images";
                },
                ["Speech Recognition"] = async () =>
                {
                    await Task.Delay(120);
                    return "Speech: Transcribed 100 audio clips";
                },
                ["Reinforcement Learning"] = async () =>
                {
                    await Task.Delay(200);
                    return "RL: Trained agent for 1000 episodes";
                },
                ["Neural Network Training"] = async () =>
                {
                    await Task.Delay(180);
                    return "NN: Completed 50 epochs";
                }
            };

            Console.WriteLine("Executing AI systems in parallel...\n");
            var sw = Stopwatch.StartNew();

            var results = await coordinator.ExecuteNamedTasksAsync(tasks);

            sw.Stop();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("Results:");
            Console.ResetColor();
            foreach (var result in results)
            {
                Console.WriteLine($"  ✓ {result.Key}: {result.Value}");
            }

            Console.WriteLine($"\nTotal execution time: {sw.ElapsedMilliseconds}ms");
            Console.WriteLine($"(Sequential would have taken ~{tasks.Count * 150}ms)");
            Console.WriteLine($"Speedup: {(tasks.Count * 150.0 / sw.ElapsedMilliseconds):F2}x");
        }

        static async Task RunBenchmarkAsync()
        {
            Console.WriteLine("=== Performance Benchmark ===\n");

            int dataSize = 10000;
            var data = Enumerable.Range(0, dataSize).ToList();

            // Sequential processing
            Console.WriteLine("1. Sequential Processing...");
            var sw = Stopwatch.StartNew();
            var sequentialResults = data.Select(x => ProcessItem(x)).ToList();
            sw.Stop();
            var sequentialTime = sw.ElapsedMilliseconds;
            Console.WriteLine($"   Time: {sequentialTime}ms");

            // Parallel processing
            Console.WriteLine("\n2. Parallel Processing...");
            sw.Restart();
            var parallelResults = coordinator.ExecuteParallel(data, x => ProcessItem(x));
            sw.Stop();
            var parallelTime = sw.ElapsedMilliseconds;
            Console.WriteLine($"   Time: {parallelTime}ms");

            // Batch processing
            Console.WriteLine("\n3. Batch Processing (batch size: 100)...");
            var batchProcessor = new BatchProcessor<int, int>(100, batch =>
            {
                return batch.Select(x => ProcessItem(x)).ToList();
            });
            sw.Restart();
            var batchResults = batchProcessor.Process(data);
            sw.Stop();
            var batchTime = sw.ElapsedMilliseconds;
            Console.WriteLine($"   Time: {batchTime}ms");

            // Summary
            Console.WriteLine("\n=== Summary ===");
            Console.WriteLine($"Data size: {dataSize:N0} items");
            Console.WriteLine($"Processor count: {Environment.ProcessorCount}");
            Console.WriteLine($"\nSequential: {sequentialTime}ms");
            Console.WriteLine($"Parallel:   {parallelTime}ms (Speedup: {(double)sequentialTime / parallelTime:F2}x)");
            Console.WriteLine($"Batch:      {batchTime}ms (Speedup: {(double)sequentialTime / batchTime:F2}x)");
        }

        static int ProcessItem(int x)
        {
            // Simulate some work
            double result = x;
            for (int i = 0; i < 100; i++)
            {
                result = Math.Sqrt(result + 1);
            }
            return (int)result;
        }

        static void RunMLDemos()
        {
            Console.WriteLine("=== Machine Learning Demonstrations ===\n");

            // Demo 1: Neural Network
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("1. Deep Neural Network Training:");
            Console.ResetColor();
            var nn = new DeepNeuralNetwork();
            nn.AddLayer(2, 4, "relu");
            nn.AddLayer(4, 1, "sigmoid");

            var X = new double[][] {
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 }
            };
            var y = new double[][] {
                new double[] { 0 },
                new double[] { 1 },
                new double[] { 1 },
                new double[] { 0 }
            };

            nn.Train(X, y, epochs: 50);
            Console.WriteLine("Predictions:");
            foreach (var input in X)
            {
                var pred = nn.Predict(input);
                Console.WriteLine($"  Input: [{input[0]}, {input[1]}] -> Output: {pred[0]:F3}");
            }
            Console.WriteLine();

            // Demo 2: K-Means Clustering
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("2. K-Means Clustering:");
            Console.ResetColor();
            var random = new Random();
            var clusterData = Enumerable.Range(0, 100).Select(_ =>
                new double[] { random.NextDouble() * 10, random.NextDouble() * 10 }
            ).ToArray();

            var kmeans = new KMeans(k: 3);
            var labels = kmeans.Fit(clusterData);
            Console.WriteLine($"  Clustered {clusterData.Length} points into 3 clusters");
            Console.WriteLine($"  Cluster distribution:");
            for (int i = 0; i < 3; i++)
                Console.WriteLine($"    Cluster {i}: {labels.Count(l => l == i)} points");
            Console.WriteLine();

            // Demo 3: Q-Learning
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("3. Q-Learning Agent:");
            Console.ResetColor();
            var qAgent = new QLearningAgent(learningRate: 0.1, discountFactor: 0.9, explorationRate: 0.2);

            Console.WriteLine("  Training for 100 episodes...");
            for (int episode = 0; episode < 100; episode++)
            {
                string state = "start";
                for (int step = 0; step < 10; step++)
                {
                    int action = qAgent.ChooseAction(state, numActions: 4);
                    double reward = random.NextDouble();
                    string nextState = $"state_{action}";
                    qAgent.Learn(state, action, reward, nextState, numActions: 4);
                    state = nextState;
                }
            }
            Console.WriteLine("  Training complete! Q-table has " + qAgent.GetQTable().Count + " states learned.");
            Console.WriteLine();

            // Demo 4: Linear Regression
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("4. Linear Regression:");
            Console.ResetColor();
            var lr = new LinearRegression();
            var X_lr = Enumerable.Range(0, 20).Select(i => new double[] { i }).ToArray();
            var y_lr = X_lr.Select(x => 2 * x[0] + 3 + (random.NextDouble() - 0.5)).ToArray();

            lr.Fit(X_lr, y_lr, epochs: 100, learningRate: 0.01);
            Console.WriteLine("  Fitted line to noisy data");
            Console.WriteLine($"  Sample predictions:");
            for (int i = 0; i < 5; i++)
            {
                var test = new double[] { i * 4 };
                Console.WriteLine($"    X={test[0]:F0} -> Y={lr.Predict(test):F2} (true: {2 * test[0] + 3:F2})");
            }
            Console.WriteLine();

            // Demo 5: Decision Tree
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("5. Decision Tree Classification:");
            Console.ResetColor();
            var dt = new DecisionTree(maxDepth: 5);
            var X_dt = new double[100][];
            var y_dt = new int[100];
            for (int i = 0; i < 100; i++)
            {
                X_dt[i] = new double[] { random.NextDouble() * 10, random.NextDouble() * 10 };
                y_dt[i] = X_dt[i][0] + X_dt[i][1] > 10 ? 1 : 0;
            }

            dt.Train(X_dt, y_dt);
            int correct = 0;
            for (int i = 0; i < 20; i++)
            {
                var test = new double[] { random.NextDouble() * 10, random.NextDouble() * 10 };
                var pred = dt.Predict(test);
                var actual = test[0] + test[1] > 10 ? 1 : 0;
                if (pred == actual) correct++;
            }
            Console.WriteLine($"  Accuracy on test set: {correct / 20.0:P0}");
            Console.WriteLine();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine("✓ All ML demos completed successfully!");
            Console.ResetColor();
        }
    }
}
