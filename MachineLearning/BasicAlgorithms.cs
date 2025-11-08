using System;
using System.Collections.Generic;
using System.Linq;

namespace AWIS.MachineLearning
{
    /// <summary>
    /// Neural network layer
    /// </summary>
    public class Layer
    {
        public double[,] Weights { get; set; }
        public double[] Biases { get; set; }
        public string Activation { get; set; }

        public Layer(int inputSize, int outputSize, string activation = "relu")
        {
            Weights = InitializeWeights(inputSize, outputSize);
            Biases = new double[outputSize];
            Activation = activation;
        }

        private double[,] InitializeWeights(int rows, int cols)
        {
            var weights = new double[rows, cols];
            var random = new Random();
            double scale = Math.Sqrt(2.0 / rows); // He initialization

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    weights[i, j] = (random.NextDouble() * 2 - 1) * scale;

            return weights;
        }

        public double[] Forward(double[] input)
        {
            int outputSize = Weights.GetLength(1);
            var output = new double[outputSize];

            for (int j = 0; j < outputSize; j++)
            {
                output[j] = Biases[j];
                for (int i = 0; i < input.Length; i++)
                    output[j] += input[i] * Weights[i, j];
                output[j] = ApplyActivation(output[j]);
            }

            return output;
        }

        private double ApplyActivation(double x)
        {
            return Activation switch
            {
                "relu" => Math.Max(0, x),
                "sigmoid" => 1.0 / (1.0 + Math.Exp(-x)),
                "tanh" => Math.Tanh(x),
                "linear" => x,
                _ => x
            };
        }
    }

    /// <summary>
    /// Deep neural network with multiple layers
    /// </summary>
    public class DeepNeuralNetwork
    {
        private readonly List<Layer> layers = new();
        private double learningRate = 0.01;

        public void AddLayer(int inputSize, int outputSize, string activation = "relu")
        {
            layers.Add(new Layer(inputSize, outputSize, activation));
        }

        public double[] Predict(double[] input)
        {
            var output = input;
            foreach (var layer in layers)
                output = layer.Forward(output);
            return output;
        }

        public void Train(double[][] inputs, double[][] targets, int epochs = 100)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalLoss = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    var prediction = Predict(inputs[i]);
                    var loss = ComputeLoss(prediction, targets[i]);
                    totalLoss += loss;

                    // Backpropagation (simplified)
                    UpdateWeights(inputs[i], prediction, targets[i]);
                }

                if (epoch % 10 == 0)
                    Console.WriteLine($"Epoch {epoch}, Loss: {totalLoss / inputs.Length:F4}");
            }
        }

        private double ComputeLoss(double[] prediction, double[] target)
        {
            double loss = 0;
            for (int i = 0; i < prediction.Length; i++)
            {
                double diff = prediction[i] - target[i];
                loss += diff * diff;
            }
            return loss / prediction.Length;
        }

        private void UpdateWeights(double[] input, double[] prediction, double[] target)
        {
            // Simplified gradient descent
            var lastLayer = layers[^1];
            for (int i = 0; i < prediction.Length; i++)
            {
                double error = prediction[i] - target[i];
                lastLayer.Biases[i] -= learningRate * error;
            }
        }
    }

    /// <summary>
    /// Decision tree for classification
    /// </summary>
    public class DecisionTree
    {
        private class Node
        {
            public int FeatureIndex { get; set; }
            public double Threshold { get; set; }
            public Node? Left { get; set; }
            public Node? Right { get; set; }
            public int Label { get; set; } = -1;
            public bool IsLeaf => Left == null && Right == null;
        }

        private Node? root;
        private readonly int maxDepth;

        public DecisionTree(int maxDepth = 10)
        {
            this.maxDepth = maxDepth;
        }

        public void Train(double[][] data, int[] labels)
        {
            root = BuildTree(data, labels, 0);
        }

        private Node BuildTree(double[][] data, int[] labels, int depth)
        {
            if (depth >= maxDepth || labels.Distinct().Count() == 1)
            {
                return new Node { Label = labels.GroupBy(x => x).OrderByDescending(g => g.Count()).First().Key };
            }

            var (featureIndex, threshold) = FindBestSplit(data, labels);
            if (featureIndex == -1)
            {
                return new Node { Label = labels.GroupBy(x => x).OrderByDescending(g => g.Count()).First().Key };
            }

            var (leftData, leftLabels, rightData, rightLabels) = Split(data, labels, featureIndex, threshold);

            return new Node
            {
                FeatureIndex = featureIndex,
                Threshold = threshold,
                Left = BuildTree(leftData, leftLabels, depth + 1),
                Right = BuildTree(rightData, rightLabels, depth + 1)
            };
        }

        private (int featureIndex, double threshold) FindBestSplit(double[][] data, int[] labels)
        {
            int bestFeature = -1;
            double bestThreshold = 0;
            double bestGini = double.MaxValue;

            int numFeatures = data[0].Length;
            for (int f = 0; f < numFeatures; f++)
            {
                var values = data.Select(row => row[f]).Distinct().OrderBy(v => v).ToArray();
                foreach (var threshold in values)
                {
                    var (leftData, leftLabels, rightData, rightLabels) = Split(data, labels, f, threshold);
                    if (leftLabels.Length == 0 || rightLabels.Length == 0) continue;

                    double gini = ComputeGini(leftLabels, rightLabels);
                    if (gini < bestGini)
                    {
                        bestGini = gini;
                        bestFeature = f;
                        bestThreshold = threshold;
                    }
                }
            }

            return (bestFeature, bestThreshold);
        }

        private (double[][] leftData, int[] leftLabels, double[][] rightData, int[] rightLabels) Split(
            double[][] data, int[] labels, int featureIndex, double threshold)
        {
            var leftData = new List<double[]>();
            var leftLabels = new List<int>();
            var rightData = new List<double[]>();
            var rightLabels = new List<int>();

            for (int i = 0; i < data.Length; i++)
            {
                if (data[i][featureIndex] <= threshold)
                {
                    leftData.Add(data[i]);
                    leftLabels.Add(labels[i]);
                }
                else
                {
                    rightData.Add(data[i]);
                    rightLabels.Add(labels[i]);
                }
            }

            return (leftData.ToArray(), leftLabels.ToArray(), rightData.ToArray(), rightLabels.ToArray());
        }

        private double ComputeGini(int[] leftLabels, int[] rightLabels)
        {
            int total = leftLabels.Length + rightLabels.Length;
            double leftGini = 1.0 - leftLabels.GroupBy(x => x).Sum(g => Math.Pow(g.Count() / (double)leftLabels.Length, 2));
            double rightGini = 1.0 - rightLabels.GroupBy(x => x).Sum(g => Math.Pow(g.Count() / (double)rightLabels.Length, 2));
            return (leftLabels.Length * leftGini + rightLabels.Length * rightGini) / total;
        }

        public int Predict(double[] features)
        {
            var node = root;
            while (node != null && !node.IsLeaf)
            {
                node = features[node.FeatureIndex] <= node.Threshold ? node.Left : node.Right;
            }
            return node?.Label ?? -1;
        }
    }

    /// <summary>
    /// K-Means clustering algorithm
    /// </summary>
    public class KMeans
    {
        private readonly int k;
        private readonly int maxIterations;
        private double[][] centroids = Array.Empty<double[]>();

        public KMeans(int k, int maxIterations = 100)
        {
            this.k = k;
            this.maxIterations = maxIterations;
        }

        public int[] Fit(double[][] data)
        {
            int n = data.Length;
            int dims = data[0].Length;
            var random = new Random();

            // Initialize centroids randomly
            centroids = new double[k][];
            var indices = Enumerable.Range(0, n).OrderBy(x => random.Next()).Take(k).ToArray();
            for (int i = 0; i < k; i++)
                centroids[i] = (double[])data[indices[i]].Clone();

            int[] labels = new int[n];
            for (int iter = 0; iter < maxIterations; iter++)
            {
                // Assign points to nearest centroid
                bool changed = false;
                for (int i = 0; i < n; i++)
                {
                    int nearest = FindNearestCentroid(data[i]);
                    if (labels[i] != nearest)
                    {
                        labels[i] = nearest;
                        changed = true;
                    }
                }

                if (!changed) break;

                // Update centroids
                for (int j = 0; j < k; j++)
                {
                    var clusterPoints = data.Where((_, i) => labels[i] == j).ToArray();
                    if (clusterPoints.Length > 0)
                    {
                        for (int d = 0; d < dims; d++)
                            centroids[j][d] = clusterPoints.Average(p => p[d]);
                    }
                }
            }

            return labels;
        }

        private int FindNearestCentroid(double[] point)
        {
            int nearest = 0;
            double minDist = EuclideanDistance(point, centroids[0]);
            for (int i = 1; i < k; i++)
            {
                double dist = EuclideanDistance(point, centroids[i]);
                if (dist < minDist)
                {
                    minDist = dist;
                    nearest = i;
                }
            }
            return nearest;
        }

        private double EuclideanDistance(double[] a, double[] b)
        {
            double sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                double diff = a[i] - b[i];
                sum += diff * diff;
            }
            return Math.Sqrt(sum);
        }
    }

    /// <summary>
    /// Linear regression
    /// </summary>
    public class LinearRegression
    {
        private double[] weights = Array.Empty<double>();
        private double bias;

        public void Fit(double[][] X, double[] y, int epochs = 1000, double learningRate = 0.01)
        {
            int n = X.Length;
            int features = X[0].Length;
            weights = new double[features];
            bias = 0;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double[] predictions = X.Select(Predict).ToArray();

                // Compute gradients
                double[] weightGrads = new double[features];
                double biasGrad = 0;

                for (int i = 0; i < n; i++)
                {
                    double error = predictions[i] - y[i];
                    biasGrad += error;
                    for (int j = 0; j < features; j++)
                        weightGrads[j] += error * X[i][j];
                }

                // Update parameters
                bias -= learningRate * biasGrad / n;
                for (int j = 0; j < features; j++)
                    weights[j] -= learningRate * weightGrads[j] / n;
            }
        }

        public double Predict(double[] x)
        {
            double result = bias;
            for (int i = 0; i < weights.Length; i++)
                result += weights[i] * x[i];
            return result;
        }
    }

    /// <summary>
    /// Logistic regression for classification
    /// </summary>
    public class LogisticRegression
    {
        private double[] weights = Array.Empty<double>();
        private double bias;

        public void Fit(double[][] X, int[] y, int epochs = 1000, double learningRate = 0.01)
        {
            int n = X.Length;
            int features = X[0].Length;
            weights = new double[features];
            bias = 0;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double[] predictions = X.Select(PredictProbability).ToArray();

                // Compute gradients
                double[] weightGrads = new double[features];
                double biasGrad = 0;

                for (int i = 0; i < n; i++)
                {
                    double error = predictions[i] - y[i];
                    biasGrad += error;
                    for (int j = 0; j < features; j++)
                        weightGrads[j] += error * X[i][j];
                }

                // Update parameters
                bias -= learningRate * biasGrad / n;
                for (int j = 0; j < features; j++)
                    weights[j] -= learningRate * weightGrads[j] / n;
            }
        }

        public double PredictProbability(double[] x)
        {
            double z = bias;
            for (int i = 0; i < weights.Length; i++)
                z += weights[i] * x[i];
            return 1.0 / (1.0 + Math.Exp(-z));
        }

        public int Predict(double[] x) => PredictProbability(x) >= 0.5 ? 1 : 0;
    }
}
