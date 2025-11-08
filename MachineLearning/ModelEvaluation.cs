using System;
using System.Collections.Generic;
using System.Linq;

namespace AWIS.MachineLearning;

/// <summary>
/// Comprehensive model evaluation metrics
/// </summary>
public class ModelEvaluator
{
    /// <summary>
    /// Compute confusion matrix for classification
    /// </summary>
    public static ConfusionMatrix ComputeConfusionMatrix(int[] yTrue, int[] yPred, int numClasses)
    {
        var matrix = new int[numClasses, numClasses];

        for (int i = 0; i < yTrue.Length; i++)
        {
            matrix[yTrue[i], yPred[i]]++;
        }

        return new ConfusionMatrix(matrix, numClasses);
    }

    /// <summary>
    /// Compute classification metrics (precision, recall, F1)
    /// </summary>
    public static ClassificationMetrics ComputeClassificationMetrics(int[] yTrue, int[] yPred)
    {
        var numClasses = Math.Max(yTrue.Max(), yPred.Max()) + 1;
        var cm = ComputeConfusionMatrix(yTrue, yPred, numClasses);

        var precision = new double[numClasses];
        var recall = new double[numClasses];
        var f1 = new double[numClasses];

        for (int i = 0; i < numClasses; i++)
        {
            double tp = cm.Matrix[i, i];
            double fp = Enumerable.Range(0, numClasses).Where(j => j != i).Sum(j => cm.Matrix[j, i]);
            double fn = Enumerable.Range(0, numClasses).Where(j => j != i).Sum(j => cm.Matrix[i, j]);

            precision[i] = tp / (tp + fp);
            recall[i] = tp / (tp + fn);
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]);

            if (double.IsNaN(precision[i])) precision[i] = 0;
            if (double.IsNaN(recall[i])) recall[i] = 0;
            if (double.IsNaN(f1[i])) f1[i] = 0;
        }

        return new ClassificationMetrics
        {
            Accuracy = cm.Accuracy,
            Precision = precision,
            Recall = recall,
            F1Score = f1,
            MacroPrecision = precision.Average(),
            MacroRecall = recall.Average(),
            MacroF1 = f1.Average(),
            ConfusionMatrix = cm
        };
    }

    /// <summary>
    /// Compute regression metrics (MSE, RMSE, MAE, R²)
    /// </summary>
    public static RegressionMetrics ComputeRegressionMetrics(double[] yTrue, double[] yPred)
    {
        double n = yTrue.Length;

        // Mean Squared Error
        double mse = yTrue.Zip(yPred, (t, p) => Math.Pow(t - p, 2)).Average();

        // Root Mean Squared Error
        double rmse = Math.Sqrt(mse);

        // Mean Absolute Error
        double mae = yTrue.Zip(yPred, (t, p) => Math.Abs(t - p)).Average();

        // R² Score
        double yMean = yTrue.Average();
        double ssTot = yTrue.Sum(y => Math.Pow(y - yMean, 2));
        double ssRes = yTrue.Zip(yPred, (t, p) => Math.Pow(t - p, 2)).Sum();
        double r2 = 1 - (ssRes / ssTot);

        // Mean Absolute Percentage Error
        double mape = yTrue.Zip(yPred, (t, p) =>
            Math.Abs((t - p) / (Math.Abs(t) + 1e-10))).Average() * 100;

        return new RegressionMetrics
        {
            MSE = mse,
            RMSE = rmse,
            MAE = mae,
            R2Score = r2,
            MAPE = mape
        };
    }

    /// <summary>
    /// Compute ROC curve and AUC
    /// </summary>
    public static ROCCurve ComputeROC(double[] yTrue, double[] scores)
    {
        // Sort by score descending
        var pairs = yTrue.Zip(scores, (t, s) => (True: t, Score: s))
                        .OrderByDescending(p => p.Score)
                        .ToArray();

        var tprs = new List<double>();
        var fprs = new List<double>();
        var thresholds = new List<double>();

        int positives = (int)yTrue.Sum();
        int negatives = yTrue.Length - positives;

        int tp = 0, fp = 0;

        for (int i = 0; i < pairs.Length; i++)
        {
            if (pairs[i].True == 1)
                tp++;
            else
                fp++;

            double tpr = (double)tp / positives;
            double fpr = (double)fp / negatives;

            tprs.Add(tpr);
            fprs.Add(fpr);
            thresholds.Add(pairs[i].Score);
        }

        // Compute AUC using trapezoidal rule
        double auc = 0;
        for (int i = 1; i < fprs.Count; i++)
        {
            auc += (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2;
        }

        return new ROCCurve
        {
            FPR = fprs.ToArray(),
            TPR = tprs.ToArray(),
            Thresholds = thresholds.ToArray(),
            AUC = auc
        };
    }

    /// <summary>
    /// K-Fold Cross Validation
    /// </summary>
    public static CrossValidationResult KFoldCrossValidation<TModel>(
        TModel model,
        double[][] X,
        double[] y,
        int k = 5,
        Func<TModel, double[][], double[], double[][], double[], double> evaluator
    ) where TModel : class
    {
        int n = X.Length;
        int foldSize = n / k;
        var scores = new double[k];
        var random = new Random(42);

        // Shuffle indices
        var indices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).ToArray();

        for (int fold = 0; fold < k; fold++)
        {
            // Split data
            var testStart = fold * foldSize;
            var testEnd = fold == k - 1 ? n : (fold + 1) * foldSize;

            var testIndices = indices.Skip(testStart).Take(testEnd - testStart).ToArray();
            var trainIndices = indices.Where(i => !testIndices.Contains(i)).ToArray();

            var XTrain = trainIndices.Select(i => X[i]).ToArray();
            var yTrain = trainIndices.Select(i => y[i]).ToArray();
            var XTest = testIndices.Select(i => X[i]).ToArray();
            var yTest = testIndices.Select(i => y[i]).ToArray();

            // Evaluate
            scores[fold] = evaluator(model, XTrain, yTrain, XTest, yTest);
        }

        return new CrossValidationResult
        {
            Scores = scores,
            Mean = scores.Average(),
            StdDev = Math.Sqrt(scores.Select(s => Math.Pow(s - scores.Average(), 2)).Average())
        };
    }

    /// <summary>
    /// Learning curve analysis
    /// </summary>
    public static LearningCurve ComputeLearningCurve<TModel>(
        TModel model,
        double[][] X,
        double[] y,
        int[] trainingSizes,
        Func<TModel, double[][], double[], double[][], double[], double> evaluator
    ) where TModel : class
    {
        var trainScores = new double[trainingSizes.Length];
        var valScores = new double[trainingSizes.Length];
        var random = new Random(42);

        // Shuffle data
        var indices = Enumerable.Range(0, X.Length).OrderBy(_ => random.Next()).ToArray();
        var XShuffled = indices.Select(i => X[i]).ToArray();
        var yShuffled = indices.Select(i => y[i]).ToArray();

        for (int i = 0; i < trainingSizes.Length; i++)
        {
            int size = trainingSizes[i];

            // Use first 80% for training, last 20% for validation
            int trainEnd = (int)(size * 0.8);

            var XTrain = XShuffled.Take(trainEnd).ToArray();
            var yTrain = yShuffled.Take(trainEnd).ToArray();
            var XVal = XShuffled.Skip(trainEnd).Take(size - trainEnd).ToArray();
            var yVal = yShuffled.Skip(trainEnd).Take(size - trainEnd).ToArray();

            trainScores[i] = evaluator(model, XTrain, yTrain, XTrain, yTrain);
            valScores[i] = evaluator(model, XTrain, yTrain, XVal, yVal);
        }

        return new LearningCurve
        {
            TrainingSizes = trainingSizes,
            TrainScores = trainScores,
            ValidationScores = valScores
        };
    }
}

/// <summary>
/// Confusion matrix with computed metrics
/// </summary>
public class ConfusionMatrix
{
    public int[,] Matrix { get; }
    public int NumClasses { get; }

    public ConfusionMatrix(int[,] matrix, int numClasses)
    {
        Matrix = matrix;
        NumClasses = numClasses;
    }

    public double Accuracy
    {
        get
        {
            double correct = 0;
            double total = 0;

            for (int i = 0; i < NumClasses; i++)
            {
                for (int j = 0; j < NumClasses; j++)
                {
                    if (i == j)
                        correct += Matrix[i, j];
                    total += Matrix[i, j];
                }
            }

            return correct / total;
        }
    }

    public void Print()
    {
        Console.WriteLine("Confusion Matrix:");
        Console.Write("     ");
        for (int j = 0; j < NumClasses; j++)
            Console.Write($"{j,5}");
        Console.WriteLine();

        for (int i = 0; i < NumClasses; i++)
        {
            Console.Write($"{i,3}: ");
            for (int j = 0; j < NumClasses; j++)
            {
                Console.Write($"{Matrix[i, j],5}");
            }
            Console.WriteLine();
        }

        Console.WriteLine($"Accuracy: {Accuracy:F4}");
    }
}

/// <summary>
/// Classification metrics
/// </summary>
public class ClassificationMetrics
{
    public double Accuracy { get; set; }
    public double[] Precision { get; set; } = Array.Empty<double>();
    public double[] Recall { get; set; } = Array.Empty<double>();
    public double[] F1Score { get; set; } = Array.Empty<double>();
    public double MacroPrecision { get; set; }
    public double MacroRecall { get; set; }
    public double MacroF1 { get; set; }
    public ConfusionMatrix? ConfusionMatrix { get; set; }

    public void Print()
    {
        Console.WriteLine("Classification Metrics:");
        Console.WriteLine($"Accuracy: {Accuracy:F4}");
        Console.WriteLine($"Macro Precision: {MacroPrecision:F4}");
        Console.WriteLine($"Macro Recall: {MacroRecall:F4}");
        Console.WriteLine($"Macro F1: {MacroF1:F4}");

        Console.WriteLine("\nPer-Class Metrics:");
        for (int i = 0; i < Precision.Length; i++)
        {
            Console.WriteLine($"Class {i}: Precision={Precision[i]:F4}, Recall={Recall[i]:F4}, F1={F1Score[i]:F4}");
        }
    }
}

/// <summary>
/// Regression metrics
/// </summary>
public class RegressionMetrics
{
    public double MSE { get; set; }
    public double RMSE { get; set; }
    public double MAE { get; set; }
    public double R2Score { get; set; }
    public double MAPE { get; set; }

    public void Print()
    {
        Console.WriteLine("Regression Metrics:");
        Console.WriteLine($"MSE: {MSE:F4}");
        Console.WriteLine($"RMSE: {RMSE:F4}");
        Console.WriteLine($"MAE: {MAE:F4}");
        Console.WriteLine($"R² Score: {R2Score:F4}");
        Console.WriteLine($"MAPE: {MAPE:F2}%");
    }
}

/// <summary>
/// ROC curve data
/// </summary>
public class ROCCurve
{
    public double[] FPR { get; set; } = Array.Empty<double>();
    public double[] TPR { get; set; } = Array.Empty<double>();
    public double[] Thresholds { get; set; } = Array.Empty<double>();
    public double AUC { get; set; }

    public void Print()
    {
        Console.WriteLine($"ROC AUC: {AUC:F4}");
    }
}

/// <summary>
/// Cross-validation results
/// </summary>
public class CrossValidationResult
{
    public double[] Scores { get; set; } = Array.Empty<double>();
    public double Mean { get; set; }
    public double StdDev { get; set; }

    public void Print()
    {
        Console.WriteLine("Cross-Validation Results:");
        Console.WriteLine($"Scores: {string.Join(", ", Scores.Select(s => $"{s:F4}"))}");
        Console.WriteLine($"Mean: {Mean:F4} ± {StdDev:F4}");
    }
}

/// <summary>
/// Learning curve data
/// </summary>
public class LearningCurve
{
    public int[] TrainingSizes { get; set; } = Array.Empty<int>();
    public double[] TrainScores { get; set; } = Array.Empty<double>();
    public double[] ValidationScores { get; set; } = Array.Empty<double>();

    public void Print()
    {
        Console.WriteLine("Learning Curve:");
        for (int i = 0; i < TrainingSizes.Length; i++)
        {
            Console.WriteLine($"Size {TrainingSizes[i]}: Train={TrainScores[i]:F4}, Val={ValidationScores[i]:F4}");
        }
    }
}

/// <summary>
/// Hyperparameter tuning with grid search
/// </summary>
public class GridSearchCV<TModel> where TModel : class
{
    private readonly Dictionary<string, object[]> _paramGrid;
    private readonly Func<Dictionary<string, object>, TModel> _modelFactory;
    private readonly Func<TModel, double[][], double[], double[][], double[], double> _evaluator;

    public GridSearchCV(
        Dictionary<string, object[]> paramGrid,
        Func<Dictionary<string, object>, TModel> modelFactory,
        Func<TModel, double[][], double[], double[][], double[], double> evaluator)
    {
        _paramGrid = paramGrid;
        _modelFactory = modelFactory;
        _evaluator = evaluator;
    }

    public GridSearchResult<TModel> Fit(double[][] X, double[] y, int cv = 5)
    {
        var results = new List<(Dictionary<string, object> Params, double Score)>();

        // Generate all combinations
        var paramCombinations = GenerateParameterCombinations(_paramGrid);

        foreach (var paramCombo in paramCombinations)
        {
            var model = _modelFactory(paramCombo);

            // Cross-validation
            var cvResult = ModelEvaluator.KFoldCrossValidation(model, X, y, cv, _evaluator);

            results.Add((paramCombo, cvResult.Mean));

            Console.WriteLine($"Params: {string.Join(", ", paramCombo.Select(kvp => $"{kvp.Key}={kvp.Value}"))} => Score: {cvResult.Mean:F4}");
        }

        var best = results.OrderByDescending(r => r.Score).First();

        return new GridSearchResult<TModel>
        {
            BestParams = best.Params,
            BestScore = best.Score,
            BestModel = _modelFactory(best.Params),
            AllResults = results
        };
    }

    private List<Dictionary<string, object>> GenerateParameterCombinations(Dictionary<string, object[]> paramGrid)
    {
        var combinations = new List<Dictionary<string, object>> { new Dictionary<string, object>() };

        foreach (var param in paramGrid)
        {
            var newCombinations = new List<Dictionary<string, object>>();

            foreach (var combo in combinations)
            {
                foreach (var value in param.Value)
                {
                    var newCombo = new Dictionary<string, object>(combo)
                    {
                        [param.Key] = value
                    };
                    newCombinations.Add(newCombo);
                }
            }

            combinations = newCombinations;
        }

        return combinations;
    }
}

/// <summary>
/// Grid search results
/// </summary>
public class GridSearchResult<TModel> where TModel : class
{
    public Dictionary<string, object> BestParams { get; set; } = new();
    public double BestScore { get; set; }
    public TModel? BestModel { get; set; }
    public List<(Dictionary<string, object> Params, double Score)> AllResults { get; set; } = new();

    public void Print()
    {
        Console.WriteLine("Grid Search Results:");
        Console.WriteLine($"Best Score: {BestScore:F4}");
        Console.WriteLine($"Best Params: {string.Join(", ", BestParams.Select(kvp => $"{kvp.Key}={kvp.Value}"))}");
    }
}

/// <summary>
/// Feature importance analyzer
/// </summary>
public class FeatureImportanceAnalyzer
{
    /// <summary>
    /// Compute permutation feature importance
    /// </summary>
    public static double[] PermutationImportance<TModel>(
        TModel model,
        double[][] XTest,
        double[] yTest,
        Func<TModel, double[][], double[], double> scorer,
        int nRepeats = 10)
    {
        int nFeatures = XTest[0].Length;
        var importance = new double[nFeatures];
        var random = new Random();

        // Baseline score
        double baselineScore = scorer(model, XTest, yTest);

        for (int feature = 0; feature < nFeatures; feature++)
        {
            var featureScores = new double[nRepeats];

            for (int repeat = 0; repeat < nRepeats; repeat++)
            {
                // Create copy and permute feature
                var XPermuted = XTest.Select(x => x.ToArray()).ToArray();

                // Shuffle feature column
                var featureValues = XPermuted.Select(x => x[feature]).ToArray();
                for (int i = featureValues.Length - 1; i > 0; i--)
                {
                    int j = random.Next(i + 1);
                    (featureValues[i], featureValues[j]) = (featureValues[j], featureValues[i]);
                }

                for (int i = 0; i < XPermuted.Length; i++)
                    XPermuted[i][feature] = featureValues[i];

                // Score with permuted feature
                featureScores[repeat] = scorer(model, XPermuted, yTest);
            }

            // Importance is decrease in score
            importance[feature] = baselineScore - featureScores.Average();
        }

        return importance;
    }

    /// <summary>
    /// Print feature importance
    /// </summary>
    public static void PrintImportance(double[] importance, string[]? featureNames = null)
    {
        Console.WriteLine("Feature Importance:");

        var indices = Enumerable.Range(0, importance.Length)
            .OrderByDescending(i => importance[i])
            .ToArray();

        for (int i = 0; i < indices.Length; i++)
        {
            int idx = indices[i];
            string name = featureNames?[idx] ?? $"Feature {idx}";
            Console.WriteLine($"{name}: {importance[idx]:F4}");
        }
    }
}
