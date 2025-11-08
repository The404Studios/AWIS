using System;
using System.Collections.Generic;
using System.Linq;

namespace AWIS.MachineLearning;

/// <summary>
/// Random Forest Classifier - Ensemble of decision trees
/// </summary>
public class RandomForest
{
    private readonly List<DecisionTree> _trees = new();
    private readonly int _numTrees;
    private readonly int _maxDepth;
    private readonly int _minSamples;
    private readonly double _featureFraction;
    private readonly Random _random = new();

    public RandomForest(int numTrees = 100, int maxDepth = 10, int minSamples = 2, double featureFraction = 0.7)
    {
        _numTrees = numTrees;
        _maxDepth = maxDepth;
        _minSamples = minSamples;
        _featureFraction = featureFraction;
    }

    public void Train(double[][] data, int[] labels)
    {
        _trees.Clear();

        for (int i = 0; i < _numTrees; i++)
        {
            // Bootstrap sampling
            var (bootData, bootLabels) = BootstrapSample(data, labels);

            // Feature subsampling
            var numFeatures = (int)(data[0].Length * _featureFraction);
            var selectedFeatures = Enumerable.Range(0, data[0].Length)
                .OrderBy(_ => _random.Next())
                .Take(numFeatures)
                .ToArray();

            // Create and train tree
            var tree = new DecisionTree(_maxDepth, _minSamples, selectedFeatures);
            tree.Train(bootData, bootLabels);
            _trees.Add(tree);
        }
    }

    public int Predict(double[] features)
    {
        // Majority voting
        var votes = new Dictionary<int, int>();

        foreach (var tree in _trees)
        {
            var prediction = tree.Predict(features);
            votes[prediction] = votes.GetValueOrDefault(prediction, 0) + 1;
        }

        return votes.OrderByDescending(kvp => kvp.Value).First().Key;
    }

    public Dictionary<int, double> PredictProba(double[] features)
    {
        var votes = new Dictionary<int, int>();

        foreach (var tree in _trees)
        {
            var prediction = tree.Predict(features);
            votes[prediction] = votes.GetValueOrDefault(prediction, 0) + 1;
        }

        return votes.ToDictionary(kvp => kvp.Key, kvp => (double)kvp.Value / _trees.Count);
    }

    public double GetFeatureImportance(int featureIndex, double[][] data, int[] labels)
    {
        // Out-of-bag error approach
        double totalImportance = 0;

        foreach (var tree in _trees)
        {
            totalImportance += tree.GetFeatureImportance(featureIndex);
        }

        return totalImportance / _trees.Count;
    }

    private (double[][], int[]) BootstrapSample(double[][] data, int[] labels)
    {
        var n = data.Length;
        var bootData = new double[n][];
        var bootLabels = new int[n];

        for (int i = 0; i < n; i++)
        {
            var idx = _random.Next(n);
            bootData[i] = data[idx];
            bootLabels[i] = labels[idx];
        }

        return (bootData, bootLabels);
    }
}

/// <summary>
/// Gradient Boosting Machine for regression and classification
/// </summary>
public class GradientBoostingMachine
{
    private readonly List<WeakLearner> _learners = new();
    private readonly double _learningRate;
    private readonly int _numIterations;
    private readonly int _maxDepth;
    private double _initialPrediction;

    public GradientBoostingMachine(int numIterations = 100, double learningRate = 0.1, int maxDepth = 3)
    {
        _numIterations = numIterations;
        _learningRate = learningRate;
        _maxDepth = maxDepth;
    }

    public void Train(double[][] X, double[] y)
    {
        // Initialize with mean
        _initialPrediction = y.Average();
        var predictions = Enumerable.Repeat(_initialPrediction, y.Length).ToArray();

        for (int iter = 0; iter < _numIterations; iter++)
        {
            // Compute residuals (negative gradient)
            var residuals = new double[y.Length];
            for (int i = 0; i < y.Length; i++)
            {
                residuals[i] = y[i] - predictions[i];
            }

            // Fit weak learner to residuals
            var learner = new WeakLearner(_maxDepth);
            learner.Fit(X, residuals);
            _learners.Add(learner);

            // Update predictions
            for (int i = 0; i < y.Length; i++)
            {
                predictions[i] += _learningRate * learner.Predict(X[i]);
            }

            // Early stopping if residuals are small
            var mse = residuals.Select(r => r * r).Average();
            if (mse < 0.001)
                break;
        }
    }

    public double Predict(double[] x)
    {
        double prediction = _initialPrediction;

        foreach (var learner in _learners)
        {
            prediction += _learningRate * learner.Predict(x);
        }

        return prediction;
    }

    public double[] PredictBatch(double[][] X)
    {
        return X.Select(x => Predict(x)).ToArray();
    }

    /// <summary>
    /// Simple regression tree as weak learner
    /// </summary>
    private class WeakLearner
    {
        private Node? _root;
        private readonly int _maxDepth;

        public WeakLearner(int maxDepth)
        {
            _maxDepth = maxDepth;
        }

        public void Fit(double[][] X, double[] y)
        {
            _root = BuildTree(X, y, 0);
        }

        private Node BuildTree(double[][] X, double[] y, int depth)
        {
            if (depth >= _maxDepth || y.Length < 2)
            {
                return new Node { Value = y.Average(), IsLeaf = true };
            }

            // Find best split
            int bestFeature = -1;
            double bestThreshold = 0;
            double bestGain = double.MinValue;

            for (int feature = 0; feature < X[0].Length; feature++)
            {
                var values = X.Select((x, i) => (x[feature], i)).OrderBy(t => t.Item1).ToArray();

                for (int i = 1; i < values.Length; i++)
                {
                    if (Math.Abs(values[i].Item1 - values[i - 1].Item1) < 1e-10)
                        continue;

                    double threshold = (values[i].Item1 + values[i - 1].Item1) / 2;

                    var leftIndices = values.Take(i).Select(t => t.i).ToArray();
                    var rightIndices = values.Skip(i).Select(t => t.i).ToArray();

                    if (leftIndices.Length == 0 || rightIndices.Length == 0)
                        continue;

                    double gain = ComputeVarianceReduction(y, leftIndices, rightIndices);

                    if (gain > bestGain)
                    {
                        bestGain = gain;
                        bestFeature = feature;
                        bestThreshold = threshold;
                    }
                }
            }

            if (bestFeature == -1)
            {
                return new Node { Value = y.Average(), IsLeaf = true };
            }

            // Split data
            var leftMask = X.Select(x => x[bestFeature] <= bestThreshold).ToArray();
            var leftX = X.Where((x, i) => leftMask[i]).ToArray();
            var leftY = y.Where((_, i) => leftMask[i]).ToArray();
            var rightX = X.Where((x, i) => !leftMask[i]).ToArray();
            var rightY = y.Where((_, i) => !leftMask[i]).ToArray();

            return new Node
            {
                FeatureIndex = bestFeature,
                Threshold = bestThreshold,
                Left = BuildTree(leftX, leftY, depth + 1),
                Right = BuildTree(rightX, rightY, depth + 1),
                IsLeaf = false
            };
        }

        private double ComputeVarianceReduction(double[] y, int[] leftIndices, int[] rightIndices)
        {
            var totalVariance = Variance(y);
            var leftY = leftIndices.Select(i => y[i]).ToArray();
            var rightY = rightIndices.Select(i => y[i]).ToArray();

            var leftVariance = Variance(leftY);
            var rightVariance = Variance(rightY);

            var weightedVariance = (leftY.Length * leftVariance + rightY.Length * rightVariance) / y.Length;

            return totalVariance - weightedVariance;
        }

        private double Variance(double[] values)
        {
            if (values.Length == 0) return 0;
            var mean = values.Average();
            return values.Select(v => Math.Pow(v - mean, 2)).Average();
        }

        public double Predict(double[] x)
        {
            var node = _root;
            while (node != null && !node.IsLeaf)
            {
                node = x[node.FeatureIndex] <= node.Threshold ? node.Left : node.Right;
            }
            return node?.Value ?? 0;
        }

        private class Node
        {
            public int FeatureIndex { get; set; }
            public double Threshold { get; set; }
            public Node? Left { get; set; }
            public Node? Right { get; set; }
            public double Value { get; set; }
            public bool IsLeaf { get; set; }
        }
    }
}

/// <summary>
/// Support Vector Machine (SVM) with SMO algorithm
/// </summary>
public class SVM
{
    private double[] _alpha = Array.Empty<double>();
    private double _b;
    private double[][] _X = Array.Empty<double[]>();
    private int[] _y = Array.Empty<int>();
    private readonly double _C; // Regularization parameter
    private readonly double _tol; // Tolerance
    private readonly int _maxIter;
    private readonly string _kernel;
    private readonly double _gamma; // RBF kernel parameter

    public SVM(double C = 1.0, string kernel = "linear", double gamma = 0.1, double tol = 0.001, int maxIter = 1000)
    {
        _C = C;
        _kernel = kernel;
        _gamma = gamma;
        _tol = tol;
        _maxIter = maxIter;
    }

    public void Train(double[][] X, int[] y)
    {
        _X = X;
        _y = y;
        int n = X.Length;

        _alpha = new double[n];
        _b = 0;

        // SMO algorithm
        for (int iter = 0; iter < _maxIter; iter++)
        {
            int numChangedAlphas = 0;

            for (int i = 0; i < n; i++)
            {
                double Ei = ComputeError(i);

                if ((_y[i] * Ei < -_tol && _alpha[i] < _C) ||
                    (_y[i] * Ei > _tol && _alpha[i] > 0))
                {
                    // Select j randomly
                    int j = i;
                    while (j == i)
                        j = new Random().Next(n);

                    double Ej = ComputeError(j);

                    double alphaIOld = _alpha[i];
                    double alphaJOld = _alpha[j];

                    // Compute L and H
                    double L, H;
                    if (_y[i] != _y[j])
                    {
                        L = Math.Max(0, _alpha[j] - _alpha[i]);
                        H = Math.Min(_C, _C + _alpha[j] - _alpha[i]);
                    }
                    else
                    {
                        L = Math.Max(0, _alpha[i] + _alpha[j] - _C);
                        H = Math.Min(_C, _alpha[i] + _alpha[j]);
                    }

                    if (Math.Abs(L - H) < 1e-10)
                        continue;

                    // Compute eta
                    double eta = 2 * Kernel(_X[i], _X[j]) - Kernel(_X[i], _X[i]) - Kernel(_X[j], _X[j]);

                    if (eta >= 0)
                        continue;

                    // Update alpha[j]
                    _alpha[j] = alphaJOld - _y[j] * (Ei - Ej) / eta;
                    _alpha[j] = Math.Max(L, Math.Min(H, _alpha[j]));

                    if (Math.Abs(_alpha[j] - alphaJOld) < 1e-5)
                        continue;

                    // Update alpha[i]
                    _alpha[i] = alphaIOld + _y[i] * _y[j] * (alphaJOld - _alpha[j]);

                    // Update b
                    double b1 = _b - Ei - _y[i] * (_alpha[i] - alphaIOld) * Kernel(_X[i], _X[i])
                                - _y[j] * (_alpha[j] - alphaJOld) * Kernel(_X[i], _X[j]);

                    double b2 = _b - Ej - _y[i] * (_alpha[i] - alphaIOld) * Kernel(_X[i], _X[j])
                                - _y[j] * (_alpha[j] - alphaJOld) * Kernel(_X[j], _X[j]);

                    if (_alpha[i] > 0 && _alpha[i] < _C)
                        _b = b1;
                    else if (_alpha[j] > 0 && _alpha[j] < _C)
                        _b = b2;
                    else
                        _b = (b1 + b2) / 2;

                    numChangedAlphas++;
                }
            }

            if (numChangedAlphas == 0)
                break;
        }
    }

    private double ComputeError(int i)
    {
        double prediction = 0;
        for (int j = 0; j < _X.Length; j++)
        {
            prediction += _alpha[j] * _y[j] * Kernel(_X[j], _X[i]);
        }
        prediction += _b;

        return prediction - _y[i];
    }

    private double Kernel(double[] x1, double[] x2)
    {
        if (_kernel == "linear")
        {
            return x1.Zip(x2, (a, b) => a * b).Sum();
        }
        else if (_kernel == "rbf")
        {
            double sum = x1.Zip(x2, (a, b) => Math.Pow(a - b, 2)).Sum();
            return Math.Exp(-_gamma * sum);
        }
        else if (_kernel == "polynomial")
        {
            double dot = x1.Zip(x2, (a, b) => a * b).Sum();
            return Math.Pow(dot + 1, 3); // degree 3 polynomial
        }

        return 0;
    }

    public int Predict(double[] x)
    {
        double sum = 0;
        for (int i = 0; i < _X.Length; i++)
        {
            sum += _alpha[i] * _y[i] * Kernel(_X[i], x);
        }
        sum += _b;

        return sum >= 0 ? 1 : -1;
    }

    public double DecisionFunction(double[] x)
    {
        double sum = 0;
        for (int i = 0; i < _X.Length; i++)
        {
            sum += _alpha[i] * _y[i] * Kernel(_X[i], x);
        }
        return sum + _b;
    }

    public int[] GetSupportVectors()
    {
        return _alpha.Select((alpha, i) => (alpha, i))
                     .Where(t => t.alpha > 0)
                     .Select(t => t.i)
                     .ToArray();
    }
}

/// <summary>
/// Principal Component Analysis (PCA) for dimensionality reduction
/// </summary>
public class PCA
{
    private double[][] _components = Array.Empty<double[]>();
    private double[] _mean = Array.Empty<double>();
    private double[] _explainedVariance = Array.Empty<double>();

    public void Fit(double[][] X, int nComponents)
    {
        int n = X.Length;
        int d = X[0].Length;

        // Compute mean
        _mean = new double[d];
        for (int j = 0; j < d; j++)
        {
            _mean[j] = X.Average(x => x[j]);
        }

        // Center data
        var centered = X.Select(x => x.Zip(_mean, (xi, mi) => xi - mi).ToArray()).ToArray();

        // Compute covariance matrix
        var cov = new double[d, d];
        for (int i = 0; i < d; i++)
        {
            for (int j = 0; j < d; j++)
            {
                cov[i, j] = centered.Sum(x => x[i] * x[j]) / n;
            }
        }

        // Compute eigenvectors and eigenvalues using power iteration
        _components = new double[nComponents][];
        _explainedVariance = new double[nComponents];

        for (int k = 0; k < nComponents; k++)
        {
            var (eigenvalue, eigenvector) = PowerIteration(cov, d);
            _components[k] = eigenvector;
            _explainedVariance[k] = eigenvalue;

            // Deflate covariance matrix
            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j < d; j++)
                {
                    cov[i, j] -= eigenvalue * eigenvector[i] * eigenvector[j];
                }
            }
        }
    }

    private (double eigenvalue, double[] eigenvector) PowerIteration(double[,] matrix, int size)
    {
        var random = new Random();
        var v = Enumerable.Range(0, size).Select(_ => random.NextDouble()).ToArray();

        // Normalize
        double norm = Math.Sqrt(v.Sum(x => x * x));
        v = v.Select(x => x / norm).ToArray();

        for (int iter = 0; iter < 100; iter++)
        {
            // Multiply matrix by vector
            var newV = new double[size];
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    newV[i] += matrix[i, j] * v[j];
                }
            }

            // Normalize
            norm = Math.Sqrt(newV.Sum(x => x * x));
            v = newV.Select(x => x / norm).ToArray();
        }

        // Compute eigenvalue (Rayleigh quotient)
        double eigenvalue = 0;
        for (int i = 0; i < size; i++)
        {
            for (int j = 0; j < size; j++)
            {
                eigenvalue += v[i] * matrix[i, j] * v[j];
            }
        }

        return (eigenvalue, v);
    }

    public double[][] Transform(double[][] X)
    {
        // Center and project
        return X.Select(x =>
        {
            var centered = x.Zip(_mean, (xi, mi) => xi - mi).ToArray();
            return _components.Select(component =>
                centered.Zip(component, (c, comp) => c * comp).Sum()
            ).ToArray();
        }).ToArray();
    }

    public double[] GetExplainedVarianceRatio()
    {
        double total = _explainedVariance.Sum();
        return _explainedVariance.Select(v => v / total).ToArray();
    }
}

/// <summary>
/// t-SNE (t-Distributed Stochastic Neighbor Embedding) for visualization
/// </summary>
public class TSNE
{
    private readonly int _nComponents;
    private readonly double _perplexity;
    private readonly double _learningRate;
    private readonly int _nIter;

    public TSNE(int nComponents = 2, double perplexity = 30, double learningRate = 200, int nIter = 1000)
    {
        _nComponents = nComponents;
        _perplexity = perplexity;
        _learningRate = learningRate;
        _nIter = nIter;
    }

    public double[][] FitTransform(double[][] X)
    {
        int n = X.Length;

        // Compute pairwise affinities
        var P = ComputeAffinities(X);

        // Initialize embedding randomly
        var Y = InitializeEmbedding(n, _nComponents);

        // Gradient descent
        var velocity = new double[n][];
        for (int i = 0; i < n; i++)
            velocity[i] = new double[_nComponents];

        for (int iter = 0; iter < _nIter; iter++)
        {
            // Compute Q (low-dimensional affinities)
            var Q = ComputeLowDimAffinities(Y);

            // Compute gradient
            var grad = new double[n][];
            for (int i = 0; i < n; i++)
            {
                grad[i] = new double[_nComponents];
                for (int j = 0; j < n; j++)
                {
                    if (i == j) continue;

                    double factor = (P[i, j] - Q[i, j]) * (1 + SquaredDistance(Y[i], Y[j]));

                    for (int d = 0; d < _nComponents; d++)
                    {
                        grad[i][d] += 4 * factor * (Y[i][d] - Y[j][d]);
                    }
                }
            }

            // Update with momentum
            double momentum = iter < 250 ? 0.5 : 0.8;
            for (int i = 0; i < n; i++)
            {
                for (int d = 0; d < _nComponents; d++)
                {
                    velocity[i][d] = momentum * velocity[i][d] - _learningRate * grad[i][d];
                    Y[i][d] += velocity[i][d];
                }
            }

            // Center embedding
            var mean = new double[_nComponents];
            for (int d = 0; d < _nComponents; d++)
                mean[d] = Y.Average(y => y[d]);

            for (int i = 0; i < n; i++)
                for (int d = 0; d < _nComponents; d++)
                    Y[i][d] -= mean[d];
        }

        return Y;
    }

    private double[,] ComputeAffinities(double[][] X)
    {
        int n = X.Length;
        var P = new double[n, n];

        // Compute pairwise distances
        var distances = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                distances[i, j] = Math.Sqrt(SquaredDistance(X[i], X[j]));
            }
        }

        // Compute conditional probabilities
        for (int i = 0; i < n; i++)
        {
            // Binary search for beta
            double beta = 1.0;
            var Pi = new double[n];

            for (int iter = 0; iter < 50; iter++)
            {
                double sum = 0;
                for (int j = 0; j < n; j++)
                {
                    if (i == j) continue;
                    Pi[j] = Math.Exp(-distances[i, j] * distances[i, j] * beta);
                    sum += Pi[j];
                }

                // Normalize
                for (int j = 0; j < n; j++)
                    if (i != j) Pi[j] /= sum;

                // Compute perplexity
                double H = 0;
                for (int j = 0; j < n; j++)
                    if (i != j && Pi[j] > 1e-12)
                        H -= Pi[j] * Math.Log(Pi[j]);

                double perplexity = Math.Exp(H);

                if (Math.Abs(perplexity - _perplexity) < 1e-5)
                    break;

                if (perplexity > _perplexity)
                    beta *= 2;
                else
                    beta /= 2;
            }

            for (int j = 0; j < n; j++)
                P[i, j] = Pi[j];
        }

        // Symmetrize
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                P[i, j] = (P[i, j] + P[j, i]) / (2 * n);

        return P;
    }

    private double[,] ComputeLowDimAffinities(double[][] Y)
    {
        int n = Y.Length;
        var Q = new double[n, n];
        double sum = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                Q[i, j] = 1 / (1 + SquaredDistance(Y[i], Y[j]));
                sum += Q[i, j];
            }
        }

        // Normalize
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                Q[i, j] /= sum;

        return Q;
    }

    private double[][] InitializeEmbedding(int n, int d)
    {
        var random = new Random();
        var Y = new double[n][];
        for (int i = 0; i < n; i++)
        {
            Y[i] = new double[d];
            for (int j = 0; j < d; j++)
                Y[i][j] = (random.NextDouble() - 0.5) * 0.0001;
        }
        return Y;
    }

    private double SquaredDistance(double[] a, double[] b)
    {
        return a.Zip(b, (x, y) => Math.Pow(x - y, 2)).Sum();
    }
}
