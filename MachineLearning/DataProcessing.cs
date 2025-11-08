using System;
using System.Collections.Generic;
using System.Linq;

namespace AWIS.MachineLearning
{

/// <summary>
/// Data preprocessing and feature engineering pipeline
/// </summary>
public class DataPreprocessor
{
    /// <summary>
    /// Standardize features to have mean=0 and std=1
    /// </summary>
    public class StandardScaler
    {
        private double[] _mean = Array.Empty<double>();
        private double[] _std = Array.Empty<double>();

        public void Fit(double[][] X)
        {
            int nFeatures = X[0].Length;
            _mean = new double[nFeatures];
            _std = new double[nFeatures];

            for (int j = 0; j < nFeatures; j++)
            {
                var values = X.Select(x => x[j]).ToArray();
                _mean[j] = values.Average();
                _std[j] = Math.Sqrt(values.Select(v => Math.Pow(v - _mean[j], 2)).Average());

                if (_std[j] < 1e-10)
                    _std[j] = 1.0; // Avoid division by zero
            }
        }

        public double[][] Transform(double[][] X)
        {
            return X.Select(x =>
                x.Select((xi, j) => (xi - _mean[j]) / _std[j]).ToArray()
            ).ToArray();
        }

        public double[][] FitTransform(double[][] X)
        {
            Fit(X);
            return Transform(X);
        }

        public double[][] InverseTransform(double[][] X)
        {
            return X.Select(x =>
                x.Select((xi, j) => xi * _std[j] + _mean[j]).ToArray()
            ).ToArray();
        }
    }

    /// <summary>
    /// Scale features to [0, 1] range
    /// </summary>
    public class MinMaxScaler
    {
        private double[] _min = Array.Empty<double>();
        private double[] _max = Array.Empty<double>();

        public void Fit(double[][] X)
        {
            int nFeatures = X[0].Length;
            _min = new double[nFeatures];
            _max = new double[nFeatures];

            for (int j = 0; j < nFeatures; j++)
            {
                var values = X.Select(x => x[j]).ToArray();
                _min[j] = values.Min();
                _max[j] = values.Max();

                if (Math.Abs(_max[j] - _min[j]) < 1e-10)
                    _max[j] = _min[j] + 1.0; // Avoid division by zero
            }
        }

        public double[][] Transform(double[][] X)
        {
            return X.Select(x =>
                x.Select((xi, j) => (xi - _min[j]) / (_max[j] - _min[j])).ToArray()
            ).ToArray();
        }

        public double[][] FitTransform(double[][] X)
        {
            Fit(X);
            return Transform(X);
        }
    }

    /// <summary>
    /// Normalize features to unit norm
    /// </summary>
    public class Normalizer
    {
        public static double[][] Transform(double[][] X, string norm = "l2")
        {
            return X.Select(x =>
            {
                double normValue = norm switch
                {
                    "l1" => x.Sum(Math.Abs),
                    "l2" => Math.Sqrt(x.Sum(xi => xi * xi)),
                    "max" => x.Max(Math.Abs),
                    _ => 1.0
                };

                if (normValue < 1e-10)
                    normValue = 1.0;

                return x.Select(xi => xi / normValue).ToArray();
            }).ToArray();
        }
    }

    /// <summary>
    /// Handle missing values
    /// </summary>
    public class SimpleImputer
    {
        private double[] _fillValues = Array.Empty<double>();
        private readonly string _strategy;

        public SimpleImputer(string strategy = "mean")
        {
            _strategy = strategy;
        }

        public void Fit(double[][] X)
        {
            int nFeatures = X[0].Length;
            _fillValues = new double[nFeatures];

            for (int j = 0; j < nFeatures; j++)
            {
                var values = X.Select(x => x[j]).Where(v => !double.IsNaN(v)).ToArray();

                _fillValues[j] = _strategy switch
                {
                    "mean" => values.Average(),
                    "median" => Median(values),
                    "most_frequent" => MostFrequent(values),
                    _ => 0
                };
            }
        }

        public double[][] Transform(double[][] X)
        {
            return X.Select(x =>
                x.Select((xi, j) => double.IsNaN(xi) ? _fillValues[j] : xi).ToArray()
            ).ToArray();
        }

        public double[][] FitTransform(double[][] X)
        {
            Fit(X);
            return Transform(X);
        }

        private double Median(double[] values)
        {
            var sorted = values.OrderBy(v => v).ToArray();
            int n = sorted.Length;
            return n % 2 == 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2 : sorted[n / 2];
        }

        private double MostFrequent(double[] values)
        {
            return values.GroupBy(v => v)
                         .OrderByDescending(g => g.Count())
                         .First()
                         .Key;
        }
    }

    /// <summary>
    /// Encode categorical variables
    /// </summary>
    public class OneHotEncoder
    {
        private Dictionary<int, Dictionary<double, int>> _categories = new();
        private int _totalFeatures;

        public void Fit(double[][] X)
        {
            _categories.Clear();
            _totalFeatures = 0;

            for (int j = 0; j < X[0].Length; j++)
            {
                var uniqueValues = X.Select(x => x[j]).Distinct().OrderBy(v => v).ToArray();
                var categoryMap = uniqueValues.Select((v, i) => (v, i)).ToDictionary(t => t.v, t => t.i);

                _categories[j] = categoryMap;
                _totalFeatures += categoryMap.Count;
            }
        }

        public double[][] Transform(double[][] X)
        {
            return X.Select(x =>
            {
                var encoded = new List<double>();

                for (int j = 0; j < x.Length; j++)
                {
                    var categoryMap = _categories[j];
                    var oneHot = new double[categoryMap.Count];

                    if (categoryMap.TryGetValue(x[j], out int index))
                        oneHot[index] = 1.0;

                    encoded.AddRange(oneHot);
                }

                return encoded.ToArray();
            }).ToArray();
        }

        public double[][] FitTransform(double[][] X)
        {
            Fit(X);
            return Transform(X);
        }
    }

    /// <summary>
    /// Label encoding for categorical targets
    /// </summary>
    public class LabelEncoder
    {
        private Dictionary<double, int> _labelMap = new();
        private Dictionary<int, double> _inverseLabelMap = new();

        public void Fit(double[] y)
        {
            var uniqueLabels = y.Distinct().OrderBy(v => v).ToArray();
            _labelMap = uniqueLabels.Select((v, i) => (v, i)).ToDictionary(t => t.v, t => t.i);
            _inverseLabelMap = _labelMap.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
        }

        public int[] Transform(double[] y)
        {
            return y.Select(yi => _labelMap.TryGetValue(yi, out int label) ? label : 0).ToArray();
        }

        public int[] FitTransform(double[] y)
        {
            Fit(y);
            return Transform(y);
        }

        public double[] InverseTransform(int[] y)
        {
            return y.Select(yi => _inverseLabelMap.TryGetValue(yi, out double label) ? label : 0.0).ToArray();
        }
    }
}

/// <summary>
/// Feature engineering utilities
/// </summary>
public class FeatureEngineering
{
    /// <summary>
    /// Create polynomial features
    /// </summary>
    public class PolynomialFeatures
    {
        private readonly int _degree;
        private readonly bool _includeInteractions;

        public PolynomialFeatures(int degree = 2, bool includeInteractions = true)
        {
            _degree = degree;
            _includeInteractions = includeInteractions;
        }

        public double[][] Transform(double[][] X)
        {
            return X.Select(x =>
            {
                var features = new List<double> { 1.0 }; // Bias term

                // Original features
                features.AddRange(x);

                // Polynomial features
                for (int d = 2; d <= _degree; d++)
                {
                    // Generate all combinations of degree d
                    var combinations = GenerateCombinations(x.Length, d, _includeInteractions);

                    foreach (var combo in combinations)
                    {
                        double product = 1.0;
                        foreach (var index in combo)
                        {
                            product *= x[index];
                        }
                        features.Add(product);
                    }
                }

                return features.ToArray();
            }).ToArray();
        }

        private List<int[]> GenerateCombinations(int n, int degree, bool includeInteractions)
        {
            var combinations = new List<int[]>();

            if (includeInteractions)
            {
                GenerateCombinationsWithReplacement(combinations, new int[degree], 0, n, 0, degree);
            }
            else
            {
                // Only pure powers (x^degree)
                for (int i = 0; i < n; i++)
                {
                    combinations.Add(Enumerable.Repeat(i, degree).ToArray());
                }
            }

            return combinations;
        }

        private void GenerateCombinationsWithReplacement(List<int[]> result, int[] current, int start, int n, int index, int degree)
        {
            if (index == degree)
            {
                result.Add(current.ToArray());
                return;
            }

            for (int i = start; i < n; i++)
            {
                current[index] = i;
                GenerateCombinationsWithReplacement(result, current, i, n, index + 1, degree);
            }
        }
    }

    /// <summary>
    /// Create interaction features
    /// </summary>
    public static double[][] CreateInteractionFeatures(double[][] X)
    {
        return X.Select(x =>
        {
            var features = new List<double>(x);

            // Add all pairwise interactions
            for (int i = 0; i < x.Length; i++)
            {
                for (int j = i + 1; j < x.Length; j++)
                {
                    features.Add(x[i] * x[j]);
                }
            }

            return features.ToArray();
        }).ToArray();
    }

    /// <summary>
    /// Binning/discretization of continuous features
    /// </summary>
    public class KBinsDiscretizer
    {
        private readonly int _nBins;
        private double[][] _binEdges = Array.Empty<double[]>();

        public KBinsDiscretizer(int nBins = 5)
        {
            _nBins = nBins;
        }

        public void Fit(double[][] X)
        {
            int nFeatures = X[0].Length;
            _binEdges = new double[nFeatures][];

            for (int j = 0; j < nFeatures; j++)
            {
                var values = X.Select(x => x[j]).OrderBy(v => v).ToArray();
                _binEdges[j] = new double[_nBins + 1];

                // Equal frequency binning
                for (int b = 0; b <= _nBins; b++)
                {
                    int index = (int)((double)b / _nBins * (values.Length - 1));
                    _binEdges[j][b] = values[index];
                }
            }
        }

        public double[][] Transform(double[][] X)
        {
            return X.Select(x =>
            {
                var binned = new double[x.Length];

                for (int j = 0; j < x.Length; j++)
                {
                    // Find which bin this value falls into
                    int bin = 0;
                    for (int b = 0; b < _nBins; b++)
                    {
                        if (x[j] >= _binEdges[j][b] && x[j] <= _binEdges[j][b + 1])
                        {
                            bin = b;
                            break;
                        }
                    }
                    binned[j] = bin;
                }

                return binned;
            }).ToArray();
        }

        public double[][] FitTransform(double[][] X)
        {
            Fit(X);
            return Transform(X);
        }
    }

    /// <summary>
    /// Statistical feature extraction
    /// </summary>
    public static double[] ExtractStatisticalFeatures(double[] timeSeries)
    {
        var features = new List<double>();

        // Basic statistics
        features.Add(timeSeries.Average());                                    // Mean
        features.Add(timeSeries.Max());                                        // Max
        features.Add(timeSeries.Min());                                        // Min
        features.Add(Math.Sqrt(timeSeries.Select(x => x * x).Average()));     // RMS
        features.Add(ComputeStdDev(timeSeries));                              // Std Dev
        features.Add(ComputeSkewness(timeSeries));                            // Skewness
        features.Add(ComputeKurtosis(timeSeries));                            // Kurtosis

        // Derivative features
        var diffs = timeSeries.Zip(timeSeries.Skip(1), (a, b) => b - a).ToArray();
        if (diffs.Length > 0)
        {
            features.Add(diffs.Average());                                     // Mean derivative
            features.Add(ComputeStdDev(diffs));                               // Std dev of derivative
        }

        // Percentiles
        var sorted = timeSeries.OrderBy(x => x).ToArray();
        features.Add(sorted[(int)(sorted.Length * 0.25)]);                    // 25th percentile
        features.Add(sorted[(int)(sorted.Length * 0.50)]);                    // Median
        features.Add(sorted[(int)(sorted.Length * 0.75)]);                    // 75th percentile

        return features.ToArray();
    }

    private static double ComputeStdDev(double[] values)
    {
        double mean = values.Average();
        return Math.Sqrt(values.Select(v => Math.Pow(v - mean, 2)).Average());
    }

    private static double ComputeSkewness(double[] values)
    {
        double mean = values.Average();
        double stdDev = ComputeStdDev(values);
        if (stdDev < 1e-10) return 0;

        return values.Select(v => Math.Pow((v - mean) / stdDev, 3)).Average();
    }

    private static double ComputeKurtosis(double[] values)
    {
        double mean = values.Average();
        double stdDev = ComputeStdDev(values);
        if (stdDev < 1e-10) return 0;

        return values.Select(v => Math.Pow((v - mean) / stdDev, 4)).Average() - 3;
    }
}

/// <summary>
/// Data augmentation for increasing training set size
/// </summary>
public class DataAugmentation
{
    private readonly Random _random = new();

    /// <summary>
    /// Add Gaussian noise to data
    /// </summary>
    public double[][] AddGaussianNoise(double[][] X, double sigma = 0.1)
    {
        return X.Select(x =>
            x.Select(xi => xi + SampleGaussian(0, sigma)).ToArray()
        ).ToArray();
    }

    /// <summary>
    /// SMOTE - Synthetic Minority Over-sampling Technique
    /// </summary>
    public (double[][], int[]) SMOTE(double[][] X, int[] y, int kNeighbors = 5, double samplingStrategy = 1.0)
    {
        // Find minority and majority classes
        var classCounts = y.GroupBy(yi => yi).ToDictionary(g => g.Key, g => g.Count());
        int minorityClass = classCounts.OrderBy(kvp => kvp.Value).First().Key;
        int majorityCount = classCounts.OrderByDescending(kvp => kvp.Value).First().Value;

        // Calculate number of synthetic samples to generate
        int minorityCount = classCounts[minorityClass];
        int syntheticCount = (int)((majorityCount - minorityCount) * samplingStrategy);

        // Get minority class samples
        var minorityIndices = y.Select((yi, i) => (yi, i))
                               .Where(t => t.yi == minorityClass)
                               .Select(t => t.i)
                               .ToArray();

        var minoritySamples = minorityIndices.Select(i => X[i]).ToArray();

        // Generate synthetic samples
        var syntheticSamples = new List<double[]>();
        var syntheticLabels = new List<int>();

        for (int i = 0; i < syntheticCount; i++)
        {
            // Random minority sample
            var sample = minoritySamples[_random.Next(minoritySamples.Length)];

            // Find k nearest neighbors
            var neighbors = FindKNearestNeighbors(sample, minoritySamples, kNeighbors);

            // Random neighbor
            var neighbor = neighbors[_random.Next(neighbors.Length)];

            // Generate synthetic sample
            double lambda = _random.NextDouble();
            var synthetic = sample.Zip(neighbor, (s, n) => s + lambda * (n - s)).ToArray();

            syntheticSamples.Add(synthetic);
            syntheticLabels.Add(minorityClass);
        }

        // Combine original and synthetic data
        var newX = X.Concat(syntheticSamples).ToArray();
        var newY = y.Concat(syntheticLabels).ToArray();

        return (newX, newY);
    }

    private double[][] FindKNearestNeighbors(double[] sample, double[][] candidates, int k)
    {
        var distances = candidates
            .Select(c => (Candidate: c, Distance: EuclideanDistance(sample, c)))
            .OrderBy(t => t.Distance)
            .Skip(1) // Skip the sample itself
            .Take(k)
            .Select(t => t.Candidate)
            .ToArray();

        return distances;
    }

    private double EuclideanDistance(double[] a, double[] b)
    {
        return Math.Sqrt(a.Zip(b, (x, y) => Math.Pow(x - y, 2)).Sum());
    }

    private double SampleGaussian(double mean, double stdDev)
    {
        // Box-Muller transform
        double u1 = 1.0 - _random.NextDouble();
        double u2 = 1.0 - _random.NextDouble();
        double z = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        return mean + stdDev * z;
    }
}

/// <summary>
/// Feature selection methods
/// </summary>
public class FeatureSelector
{
    /// <summary>
    /// Select k best features based on variance
    /// </summary>
    public static int[] SelectKBestByVariance(double[][] X, int k)
    {
        int nFeatures = X[0].Length;
        var variances = new double[nFeatures];

        for (int j = 0; j < nFeatures; j++)
        {
            var values = X.Select(x => x[j]).ToArray();
            double mean = values.Average();
            variances[j] = values.Select(v => Math.Pow(v - mean, 2)).Average();
        }

        return variances
            .Select((v, i) => (Variance: v, Index: i))
            .OrderByDescending(t => t.Variance)
            .Take(k)
            .Select(t => t.Index)
            .ToArray();
    }

    /// <summary>
    /// Remove low variance features
    /// </summary>
    public static int[] RemoveLowVarianceFeatures(double[][] X, double threshold = 0.01)
    {
        int nFeatures = X[0].Length;
        var selectedFeatures = new List<int>();

        for (int j = 0; j < nFeatures; j++)
        {
            var values = X.Select(x => x[j]).ToArray();
            double mean = values.Average();
            double variance = values.Select(v => Math.Pow(v - mean, 2)).Average();

            if (variance >= threshold)
                selectedFeatures.Add(j);
        }

        return selectedFeatures.ToArray();
    }

    /// <summary>
    /// Select features based on correlation with target
    /// </summary>
    public static int[] SelectByCorrelation(double[][] X, double[] y, int k)
    {
        int nFeatures = X[0].Length;
        var correlations = new double[nFeatures];

        for (int j = 0; j < nFeatures; j++)
        {
            var xCol = X.Select(x => x[j]).ToArray();
            correlations[j] = Math.Abs(ComputeCorrelation(xCol, y));
        }

        return correlations
            .Select((c, i) => (Correlation: c, Index: i))
            .OrderByDescending(t => t.Correlation)
            .Take(k)
            .Select(t => t.Index)
            .ToArray();
    }

    private static double ComputeCorrelation(double[] x, double[] y)
    {
        double meanX = x.Average();
        double meanY = y.Average();

        double numerator = x.Zip(y, (xi, yi) => (xi - meanX) * (yi - meanY)).Sum();
        double denominator = Math.Sqrt(
            x.Sum(xi => Math.Pow(xi - meanX, 2)) *
            y.Sum(yi => Math.Pow(yi - meanY, 2))
        );

        return denominator > 1e-10 ? numerator / denominator : 0;
    }
}
}
