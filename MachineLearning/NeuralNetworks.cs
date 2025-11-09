using System;
using System.Collections.Generic;
using System.Linq;

namespace AWIS.MachineLearning
{
    /// <summary>
    /// Convolutional layer for CNNs
    /// </summary>
    public class ConvLayer
    {
        public double[,,,] Filters { get; set; } // [numFilters, height, width, channels]
        public double[] Biases { get; set; }
        private readonly int stride;
        private readonly int padding;

        public ConvLayer(int numFilters, int filterSize, int inputChannels, int stride = 1, int padding = 0)
        {
            this.stride = stride;
            this.padding = padding;
            Filters = InitializeFilters(numFilters, filterSize, filterSize, inputChannels);
            Biases = new double[numFilters];
        }

        private double[,,,] InitializeFilters(int num, int h, int w, int c)
        {
            var filters = new double[num, h, w, c];
            var random = new Random();
            double scale = Math.Sqrt(2.0 / (h * w * c));

            for (int n = 0; n < num; n++)
                for (int i = 0; i < h; i++)
                    for (int j = 0; j < w; j++)
                        for (int k = 0; k < c; k++)
                            filters[n, i, j, k] = (random.NextDouble() * 2 - 1) * scale;

            return filters;
        }

        public double[,,] Forward(double[,,] input)
        {
            int inputH = input.GetLength(0);
            int inputW = input.GetLength(1);
            int inputC = input.GetLength(2);
            int filterH = Filters.GetLength(1);
            int filterW = Filters.GetLength(2);
            int numFilters = Filters.GetLength(0);

            int outputH = (inputH + 2 * padding - filterH) / stride + 1;
            int outputW = (inputW + 2 * padding - filterW) / stride + 1;
            var output = new double[outputH, outputW, numFilters];

            for (int f = 0; f < numFilters; f++)
            {
                for (int oh = 0; oh < outputH; oh++)
                {
                    for (int ow = 0; ow < outputW; ow++)
                    {
                        double sum = Biases[f];
                        for (int fh = 0; fh < filterH; fh++)
                        {
                            for (int fw = 0; fw < filterW; fw++)
                            {
                                for (int c = 0; c < inputC; c++)
                                {
                                    int ih = oh * stride + fh - padding;
                                    int iw = ow * stride + fw - padding;
                                    if (ih >= 0 && ih < inputH && iw >= 0 && iw < inputW)
                                        sum += input[ih, iw, c] * Filters[f, fh, fw, c];
                                }
                            }
                        }
                        output[oh, ow, f] = Math.Max(0, sum); // ReLU
                    }
                }
            }

            return output;
        }
    }

    /// <summary>
    /// Max pooling layer
    /// </summary>
    public class MaxPoolLayer
    {
        private readonly int poolSize;
        private readonly int stride;

        public MaxPoolLayer(int poolSize = 2, int stride = 2)
        {
            this.poolSize = poolSize;
            this.stride = stride;
        }

        public double[,,] Forward(double[,,] input)
        {
            int inputH = input.GetLength(0);
            int inputW = input.GetLength(1);
            int channels = input.GetLength(2);

            int outputH = (inputH - poolSize) / stride + 1;
            int outputW = (inputW - poolSize) / stride + 1;
            var output = new double[outputH, outputW, channels];

            for (int c = 0; c < channels; c++)
            {
                for (int oh = 0; oh < outputH; oh++)
                {
                    for (int ow = 0; ow < outputW; ow++)
                    {
                        double max = double.MinValue;
                        for (int ph = 0; ph < poolSize; ph++)
                        {
                            for (int pw = 0; pw < poolSize; pw++)
                            {
                                int ih = oh * stride + ph;
                                int iw = ow * stride + pw;
                                if (input[ih, iw, c] > max)
                                    max = input[ih, iw, c];
                            }
                        }
                        output[oh, ow, c] = max;
                    }
                }
            }

            return output;
        }
    }

    /// <summary>
    /// LSTM cell for recurrent networks
    /// </summary>
    public class LSTMCell
    {
        private double[,] Wf = null!, Wi = null!, Wc = null!, Wo = null!; // Weights
        private double[] bf = null!, bi = null!, bc = null!, bo = null!;  // Biases
        private readonly int hiddenSize;

        public LSTMCell(int inputSize, int hiddenSize)
        {
            this.hiddenSize = hiddenSize;
            InitializeWeights(inputSize, hiddenSize);
        }

        private void InitializeWeights(int inputSize, int hiddenSize)
        {
            var random = new Random();
            double scale = Math.Sqrt(2.0 / (inputSize + hiddenSize));

            Wf = RandomMatrix(inputSize + hiddenSize, hiddenSize, scale, random);
            Wi = RandomMatrix(inputSize + hiddenSize, hiddenSize, scale, random);
            Wc = RandomMatrix(inputSize + hiddenSize, hiddenSize, scale, random);
            Wo = RandomMatrix(inputSize + hiddenSize, hiddenSize, scale, random);

            bf = new double[hiddenSize];
            bi = new double[hiddenSize];
            bc = new double[hiddenSize];
            bo = new double[hiddenSize];
        }

        private double[,] RandomMatrix(int rows, int cols, double scale, Random random)
        {
            var matrix = new double[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = (random.NextDouble() * 2 - 1) * scale;
            return matrix;
        }

        public (double[] h, double[] c) Forward(double[] x, double[] hPrev, double[] cPrev)
        {
            // Concatenate input and previous hidden state
            var combined = x.Concat(hPrev).ToArray();

            // Forget gate
            var ft = Sigmoid(MatMul(combined, Wf).Zip(bf, (a, b) => a + b).ToArray());

            // Input gate
            var it = Sigmoid(MatMul(combined, Wi).Zip(bi, (a, b) => a + b).ToArray());

            // Candidate cell state
            var cTilde = Tanh(MatMul(combined, Wc).Zip(bc, (a, b) => a + b).ToArray());

            // New cell state
            var c = ft.Zip(cPrev, (f, c) => f * c).Zip(it.Zip(cTilde, (i, ct) => i * ct), (a, b) => a + b).ToArray();

            // Output gate
            var ot = Sigmoid(MatMul(combined, Wo).Zip(bo, (a, b) => a + b).ToArray());

            // New hidden state
            var h = ot.Zip(Tanh(c), (o, tc) => o * tc).ToArray();

            return (h, c);
        }

        private double[] MatMul(double[] vec, double[,] mat)
        {
            int rows = mat.GetLength(0);
            int cols = mat.GetLength(1);
            var result = new double[cols];

            for (int j = 0; j < cols; j++)
                for (int i = 0; i < rows; i++)
                    result[j] += vec[i] * mat[i, j];

            return result;
        }

        private double[] Sigmoid(double[] x) => x.Select(v => 1.0 / (1.0 + Math.Exp(-v))).ToArray();
        private double[] Tanh(double[] x) => x.Select(Math.Tanh).ToArray();
    }

    /// <summary>
    /// Recurrent Neural Network with LSTM cells
    /// </summary>
    public class RNN
    {
        private readonly List<LSTMCell> cells = new();
        private readonly int hiddenSize;

        public RNN(int inputSize, int hiddenSize, int numLayers = 1)
        {
            this.hiddenSize = hiddenSize;
            for (int i = 0; i < numLayers; i++)
            {
                int size = i == 0 ? inputSize : hiddenSize;
                cells.Add(new LSTMCell(size, hiddenSize));
            }
        }

        public double[][] Forward(double[][] sequence)
        {
            int seqLen = sequence.Length;
            var outputs = new double[seqLen][];
            var h = Enumerable.Repeat(new double[hiddenSize], cells.Count).ToArray();
            var c = Enumerable.Repeat(new double[hiddenSize], cells.Count).ToArray();

            for (int t = 0; t < seqLen; t++)
            {
                var input = sequence[t];
                for (int layer = 0; layer < cells.Count; layer++)
                {
                    (h[layer], c[layer]) = cells[layer].Forward(input, h[layer], c[layer]);
                    input = h[layer];
                }
                outputs[t] = h[^1];
            }

            return outputs;
        }
    }

    /// <summary>
    /// Attention mechanism
    /// </summary>
    public class Attention
    {
        private readonly int hiddenSize;
        private double[,] Wq, Wk, Wv;

        public Attention(int hiddenSize)
        {
            this.hiddenSize = hiddenSize;
            var random = new Random();
            double scale = Math.Sqrt(2.0 / hiddenSize);

            Wq = RandomMatrix(hiddenSize, hiddenSize, scale, random);
            Wk = RandomMatrix(hiddenSize, hiddenSize, scale, random);
            Wv = RandomMatrix(hiddenSize, hiddenSize, scale, random);
        }

        private double[,] RandomMatrix(int rows, int cols, double scale, Random random)
        {
            var matrix = new double[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    matrix[i, j] = (random.NextDouble() * 2 - 1) * scale;
            return matrix;
        }

        public double[][] Forward(double[][] queries, double[][] keys, double[][] values)
        {
            int seqLen = queries.Length;
            var output = new double[seqLen][];

            // Compute Q, K, V
            var Q = queries.Select(q => MatMul(q, Wq)).ToArray();
            var K = keys.Select(k => MatMul(k, Wk)).ToArray();
            var V = values.Select(v => MatMul(v, Wv)).ToArray();

            // Scaled dot-product attention
            for (int i = 0; i < seqLen; i++)
            {
                var scores = K.Select(k => DotProduct(Q[i], k) / Math.Sqrt(hiddenSize)).ToArray();
                var attentionWeights = Softmax(scores);

                output[i] = new double[hiddenSize];
                for (int j = 0; j < seqLen; j++)
                    for (int d = 0; d < hiddenSize; d++)
                        output[i][d] += attentionWeights[j] * V[j][d];
            }

            return output;
        }

        private double[] MatMul(double[] vec, double[,] mat)
        {
            int cols = mat.GetLength(1);
            var result = new double[cols];
            for (int j = 0; j < cols; j++)
                for (int i = 0; i < vec.Length; i++)
                    result[j] += vec[i] * mat[i, j];
            return result;
        }

        private double DotProduct(double[] a, double[] b)
        {
            return a.Zip(b, (x, y) => x * y).Sum();
        }

        private double[] Softmax(double[] x)
        {
            double max = x.Max();
            var exp = x.Select(v => Math.Exp(v - max)).ToArray();
            double sum = exp.Sum();
            return exp.Select(e => e / sum).ToArray();
        }
    }

    /// <summary>
    /// Transformer encoder layer
    /// </summary>
    public class TransformerEncoder
    {
        private readonly Attention attention;
        private readonly DeepNeuralNetwork feedForward;
        private readonly int hiddenSize;

        public TransformerEncoder(int hiddenSize, int ffnDim = 2048)
        {
            this.hiddenSize = hiddenSize;
            this.attention = new Attention(hiddenSize);

            feedForward = new DeepNeuralNetwork();
            feedForward.AddLayer(hiddenSize, ffnDim, "relu");
            feedForward.AddLayer(ffnDim, hiddenSize, "linear");
        }

        public double[][] Forward(double[][] input)
        {
            // Multi-head attention (simplified to single head)
            var attentionOut = attention.Forward(input, input, input);

            // Add & Norm (simplified - just add)
            var residual1 = input.Zip(attentionOut, (a, b) => a.Zip(b, (x, y) => x + y).ToArray()).ToArray();

            // Feed-forward
            var ffnOut = residual1.Select(x => feedForward.Predict(x)).ToArray();

            // Add & Norm
            var output = residual1.Zip(ffnOut, (a, b) => a.Zip(b, (x, y) => x + y).ToArray()).ToArray();

            return output;
        }
    }

    /// <summary>
    /// Autoencoder for dimensionality reduction
    /// </summary>
    public class Autoencoder
    {
        private readonly DeepNeuralNetwork encoder;
        private readonly DeepNeuralNetwork decoder;

        public Autoencoder(int inputDim, int encodingDim, int[] hiddenLayers)
        {
            // Encoder
            encoder = new DeepNeuralNetwork();
            int prevDim = inputDim;
            foreach (var dim in hiddenLayers)
            {
                encoder.AddLayer(prevDim, dim, "relu");
                prevDim = dim;
            }
            encoder.AddLayer(prevDim, encodingDim, "linear");

            // Decoder (mirror of encoder)
            decoder = new DeepNeuralNetwork();
            prevDim = encodingDim;
            for (int i = hiddenLayers.Length - 1; i >= 0; i--)
            {
                decoder.AddLayer(prevDim, hiddenLayers[i], "relu");
                prevDim = hiddenLayers[i];
            }
            decoder.AddLayer(prevDim, inputDim, "linear");
        }

        public double[] Encode(double[] input) => encoder.Predict(input);
        public double[] Decode(double[] encoding) => decoder.Predict(encoding);
        public double[] Reconstruct(double[] input) => Decode(Encode(input));

        public void Train(double[][] data, int epochs = 100)
        {
            // Train encoder and decoder together
            var targets = data; // Autoencoder tries to reconstruct input
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalLoss = 0;
                foreach (var sample in data)
                {
                    var reconstruction = Reconstruct(sample);
                    var loss = sample.Zip(reconstruction, (a, b) => Math.Pow(a - b, 2)).Sum();
                    totalLoss += loss;
                }

                if (epoch % 10 == 0)
                    Console.WriteLine($"Autoencoder Epoch {epoch}, Loss: {totalLoss / data.Length:F4}");
            }
        }
    }

    /// <summary>
    /// Variational Autoencoder (VAE)
    /// </summary>
    public class VAE
    {
        private readonly DeepNeuralNetwork encoder;
        private readonly DeepNeuralNetwork muNet;
        private readonly DeepNeuralNetwork logVarNet;
        private readonly DeepNeuralNetwork decoder;
        private readonly Random random = new();

        public VAE(int inputDim, int latentDim, int hiddenDim = 256)
        {
            // Encoder
            encoder = new DeepNeuralNetwork();
            encoder.AddLayer(inputDim, hiddenDim, "relu");
            encoder.AddLayer(hiddenDim, hiddenDim, "relu");

            // Mean and log variance networks
            muNet = new DeepNeuralNetwork();
            muNet.AddLayer(hiddenDim, latentDim, "linear");

            logVarNet = new DeepNeuralNetwork();
            logVarNet.AddLayer(hiddenDim, latentDim, "linear");

            // Decoder
            decoder = new DeepNeuralNetwork();
            decoder.AddLayer(latentDim, hiddenDim, "relu");
            decoder.AddLayer(hiddenDim, hiddenDim, "relu");
            decoder.AddLayer(hiddenDim, inputDim, "sigmoid");
        }

        public (double[] mu, double[] logVar, double[] z) Encode(double[] input)
        {
            var h = encoder.Predict(input);
            var mu = muNet.Predict(h);
            var logVar = logVarNet.Predict(h);

            // Reparameterization trick
            var z = mu.Zip(logVar, (m, lv) =>
            {
                double std = Math.Exp(0.5 * lv);
                double eps = SampleNormal();
                return m + std * eps;
            }).ToArray();

            return (mu, logVar, z);
        }

        public double[] Decode(double[] z) => decoder.Predict(z);

        public double[] Generate()
        {
            // Sample from standard normal
            int latentDim = decoder.Predict(new double[10]).Length; // Get latent dim
            var z = Enumerable.Range(0, 10).Select(_ => SampleNormal()).ToArray();
            return Decode(z);
        }

        private double SampleNormal()
        {
            // Box-Muller transform
            double u1 = 1.0 - random.NextDouble();
            double u2 = 1.0 - random.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        }
    }

    /// <summary>
    /// Batch normalization layer
    /// </summary>
    public class BatchNorm
    {
        private double[] runningMean;
        private double[] runningVar;
        private readonly double momentum = 0.9;
        private readonly double epsilon = 1e-5;

        public BatchNorm(int numFeatures)
        {
            runningMean = new double[numFeatures];
            runningVar = Enumerable.Repeat(1.0, numFeatures).ToArray();
        }

        public double[][] Forward(double[][] batch, bool training = true)
        {
            int batchSize = batch.Length;
            int numFeatures = batch[0].Length;
            var normalized = new double[batchSize][];

            if (training)
            {
                // Compute batch statistics
                var batchMean = new double[numFeatures];
                var batchVar = new double[numFeatures];

                for (int f = 0; f < numFeatures; f++)
                {
                    batchMean[f] = batch.Average(x => x[f]);
                    batchVar[f] = batch.Average(x => Math.Pow(x[f] - batchMean[f], 2));

                    // Update running statistics
                    runningMean[f] = momentum * runningMean[f] + (1 - momentum) * batchMean[f];
                    runningVar[f] = momentum * runningVar[f] + (1 - momentum) * batchVar[f];
                }

                // Normalize
                for (int i = 0; i < batchSize; i++)
                {
                    normalized[i] = new double[numFeatures];
                    for (int f = 0; f < numFeatures; f++)
                    {
                        normalized[i][f] = (batch[i][f] - batchMean[f]) / Math.Sqrt(batchVar[f] + epsilon);
                    }
                }
            }
            else
            {
                // Use running statistics
                for (int i = 0; i < batchSize; i++)
                {
                    normalized[i] = new double[numFeatures];
                    for (int f = 0; f < numFeatures; f++)
                    {
                        normalized[i][f] = (batch[i][f] - runningMean[f]) / Math.Sqrt(runningVar[f] + epsilon);
                    }
                }
            }

            return normalized;
        }
    }

    /// <summary>
    /// Dropout layer for regularization
    /// </summary>
    public class Dropout
    {
        private readonly double dropRate;
        private readonly Random random = new();

        public Dropout(double dropRate = 0.5)
        {
            this.dropRate = dropRate;
        }

        public double[] Forward(double[] input, bool training = true)
        {
            if (!training) return input;

            var output = new double[input.Length];
            double scale = 1.0 / (1.0 - dropRate);

            for (int i = 0; i < input.Length; i++)
            {
                if (random.NextDouble() > dropRate)
                    output[i] = input[i] * scale;
            }

            return output;
        }
    }
}
