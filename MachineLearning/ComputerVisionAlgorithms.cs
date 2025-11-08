using System;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;

namespace AWIS.MachineLearning
{
    /// <summary>
    /// Edge detection algorithms
    /// </summary>
    public class EdgeDetection
    {
        public static double[,] SobelX = new double[,] { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
        public static double[,] SobelY = new double[,] { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };

        public static double[,] Sobel(double[,] image)
        {
            int height = image.GetLength(0);
            int width = image.GetLength(1);
            var result = new double[height, width];

            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    double gx = 0, gy = 0;
                    for (int ky = -1; ky <= 1; ky++)
                    {
                        for (int kx = -1; kx <= 1; kx++)
                        {
                            gx += image[y + ky, x + kx] * SobelX[ky + 1, kx + 1];
                            gy += image[y + ky, x + kx] * SobelY[ky + 1, kx + 1];
                        }
                    }
                    result[y, x] = Math.Sqrt(gx * gx + gy * gy);
                }
            }
            return result;
        }

        public static double[,] Canny(double[,] image, double lowThreshold = 0.05, double highThreshold = 0.15)
        {
            // Step 1: Gaussian blur
            var blurred = GaussianBlur(image, 5, 1.4);

            // Step 2: Gradient calculation
            int height = blurred.GetLength(0);
            int width = blurred.GetLength(1);
            var magnitude = new double[height, width];
            var direction = new double[height, width];

            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    double gx = 0, gy = 0;
                    for (int ky = -1; ky <= 1; ky++)
                    {
                        for (int kx = -1; kx <= 1; kx++)
                        {
                            gx += blurred[y + ky, x + kx] * SobelX[ky + 1, kx + 1];
                            gy += blurred[y + ky, x + kx] * SobelY[ky + 1, kx + 1];
                        }
                    }
                    magnitude[y, x] = Math.Sqrt(gx * gx + gy * gy);
                    direction[y, x] = Math.Atan2(gy, gx);
                }
            }

            // Step 3: Non-maximum suppression
            var suppressed = NonMaximumSuppression(magnitude, direction);

            // Step 4: Double threshold and edge tracking
            return DoubleThreshold(suppressed, lowThreshold, highThreshold);
        }

        private static double[,] GaussianBlur(double[,] image, int kernelSize = 5, double sigma = 1.4)
        {
            int height = image.GetLength(0);
            int width = image.GetLength(1);
            var kernel = CreateGaussianKernel(kernelSize, sigma);
            var result = new double[height, width];
            int half = kernelSize / 2;

            for (int y = half; y < height - half; y++)
            {
                for (int x = half; x < width - half; x++)
                {
                    double sum = 0;
                    for (int ky = -half; ky <= half; ky++)
                    {
                        for (int kx = -half; kx <= half; kx++)
                        {
                            sum += image[y + ky, x + kx] * kernel[ky + half, kx + half];
                        }
                    }
                    result[y, x] = sum;
                }
            }
            return result;
        }

        private static double[,] CreateGaussianKernel(int size, double sigma)
        {
            var kernel = new double[size, size];
            int half = size / 2;
            double sum = 0;

            for (int y = -half; y <= half; y++)
            {
                for (int x = -half; x <= half; x++)
                {
                    double value = Math.Exp(-(x * x + y * y) / (2 * sigma * sigma));
                    kernel[y + half, x + half] = value;
                    sum += value;
                }
            }

            // Normalize
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                    kernel[y, x] /= sum;

            return kernel;
        }

        private static double[,] NonMaximumSuppression(double[,] magnitude, double[,] direction)
        {
            int height = magnitude.GetLength(0);
            int width = magnitude.GetLength(1);
            var result = new double[height, width];

            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    double angle = direction[y, x] * 180 / Math.PI;
                    if (angle < 0) angle += 180;

                    double q = 0, r = 0;

                    // 0 degrees
                    if ((angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180))
                    {
                        q = magnitude[y, x + 1];
                        r = magnitude[y, x - 1];
                    }
                    // 45 degrees
                    else if (angle >= 22.5 && angle < 67.5)
                    {
                        q = magnitude[y + 1, x - 1];
                        r = magnitude[y - 1, x + 1];
                    }
                    // 90 degrees
                    else if (angle >= 67.5 && angle < 112.5)
                    {
                        q = magnitude[y + 1, x];
                        r = magnitude[y - 1, x];
                    }
                    // 135 degrees
                    else if (angle >= 112.5 && angle < 157.5)
                    {
                        q = magnitude[y - 1, x - 1];
                        r = magnitude[y + 1, x + 1];
                    }

                    if (magnitude[y, x] >= q && magnitude[y, x] >= r)
                        result[y, x] = magnitude[y, x];
                }
            }
            return result;
        }

        private static double[,] DoubleThreshold(double[,] image, double low, double high)
        {
            int height = image.GetLength(0);
            int width = image.GetLength(1);
            var result = new double[height, width];

            double maxVal = 0;
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    if (image[y, x] > maxVal) maxVal = image[y, x];

            double lowThresh = maxVal * low;
            double highThresh = maxVal * high;

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    if (image[y, x] >= highThresh)
                        result[y, x] = 1.0;
                    else if (image[y, x] >= lowThresh)
                        result[y, x] = 0.5; // Weak edge
                }
            }

            // Edge tracking by hysteresis
            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    if (result[y, x] == 0.5)
                    {
                        bool hasStrongNeighbor = false;
                        for (int dy = -1; dy <= 1; dy++)
                        {
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                if (result[y + dy, x + dx] == 1.0)
                                {
                                    hasStrongNeighbor = true;
                                    break;
                                }
                            }
                            if (hasStrongNeighbor) break;
                        }
                        result[y, x] = hasStrongNeighbor ? 1.0 : 0.0;
                    }
                }
            }

            return result;
        }
    }

    /// <summary>
    /// Harris corner detection
    /// </summary>
    public class CornerDetection
    {
        public static List<(int x, int y, double response)> HarrisCorners(double[,] image, double threshold = 0.01)
        {
            int height = image.GetLength(0);
            int width = image.GetLength(1);
            var corners = new List<(int, int, double)>();

            // Compute gradients
            var Ix = new double[height, width];
            var Iy = new double[height, width];

            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    Ix[y, x] = (image[y, x + 1] - image[y, x - 1]) / 2.0;
                    Iy[y, x] = (image[y + 1, x] - image[y - 1, x]) / 2.0;
                }
            }

            // Compute products
            var Ixx = new double[height, width];
            var Iyy = new double[height, width];
            var Ixy = new double[height, width];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Ixx[y, x] = Ix[y, x] * Ix[y, x];
                    Iyy[y, x] = Iy[y, x] * Iy[y, x];
                    Ixy[y, x] = Ix[y, x] * Iy[y, x];
                }
            }

            // Gaussian weighting
            int windowSize = 3;
            double k = 0.04;

            for (int y = windowSize; y < height - windowSize; y++)
            {
                for (int x = windowSize; x < width - windowSize; x++)
                {
                    double sumIxx = 0, sumIyy = 0, sumIxy = 0;

                    for (int wy = -windowSize; wy <= windowSize; wy++)
                    {
                        for (int wx = -windowSize; wx <= windowSize; wx++)
                        {
                            sumIxx += Ixx[y + wy, x + wx];
                            sumIyy += Iyy[y + wy, x + wx];
                            sumIxy += Ixy[y + wy, x + wx];
                        }
                    }

                    // Harris response
                    double det = sumIxx * sumIyy - sumIxy * sumIxy;
                    double trace = sumIxx + sumIyy;
                    double response = det - k * trace * trace;

                    if (response > threshold)
                        corners.Add((x, y, response));
                }
            }

            // Non-maximum suppression
            var filtered = NonMaxSuppression(corners, 5);
            return filtered.OrderByDescending(c => c.response).Take(100).ToList();
        }

        private static List<(int x, int y, double response)> NonMaxSuppression(
            List<(int x, int y, double response)> corners, int radius)
        {
            var result = new List<(int, int, double)>();
            var sorted = corners.OrderByDescending(c => c.response).ToList();

            foreach (var corner in sorted)
            {
                bool isMax = true;
                foreach (var existing in result)
                {
                    double dist = Math.Sqrt(Math.Pow(corner.x - existing.x, 2) + Math.Pow(corner.y - existing.y, 2));
                    if (dist < radius)
                    {
                        isMax = false;
                        break;
                    }
                }
                if (isMax) result.Add(corner);
            }

            return result;
        }
    }

    /// <summary>
    /// Feature descriptor (simplified SIFT)
    /// </summary>
    public class FeatureDescriptor
    {
        public static double[] ComputeDescriptor(double[,] image, int x, int y, int patchSize = 16)
        {
            int height = image.GetLength(0);
            int width = image.GetLength(1);
            var descriptor = new List<double>();

            int half = patchSize / 2;
            int bins = 8; // Number of orientation bins

            for (int py = -half; py < half; py += 4)
            {
                for (int px = -half; px < half; px += 4)
                {
                    var histogram = new double[bins];

                    for (int dy = 0; dy < 4; dy++)
                    {
                        for (int dx = 0; dx < 4; dx++)
                        {
                            int iy = y + py + dy;
                            int ix = x + px + dx;

                            if (iy >= 1 && iy < height - 1 && ix >= 1 && ix < width - 1)
                            {
                                double gx = image[iy, ix + 1] - image[iy, ix - 1];
                                double gy = image[iy + 1, ix] - image[iy - 1, ix];
                                double magnitude = Math.Sqrt(gx * gx + gy * gy);
                                double angle = Math.Atan2(gy, gx) + Math.PI;

                                int bin = (int)(angle / (2 * Math.PI) * bins) % bins;
                                histogram[bin] += magnitude;
                            }
                        }
                    }

                    descriptor.AddRange(histogram);
                }
            }

            // Normalize
            double norm = Math.Sqrt(descriptor.Sum(d => d * d));
            if (norm > 0)
                descriptor = descriptor.Select(d => d / norm).ToList();

            return descriptor.ToArray();
        }

        public static double DescriptorDistance(double[] desc1, double[] desc2)
        {
            return Math.Sqrt(desc1.Zip(desc2, (a, b) => Math.Pow(a - b, 2)).Sum());
        }
    }

    /// <summary>
    /// Object detection using sliding window
    /// </summary>
    public class ObjectDetector
    {
        private readonly DeepNeuralNetwork classifier;
        private readonly int windowSize;

        public ObjectDetector(int windowSize = 64)
        {
            this.windowSize = windowSize;
            classifier = new DeepNeuralNetwork();
            // Initialize classifier
            classifier.AddLayer(windowSize * windowSize, 128, "relu");
            classifier.AddLayer(128, 64, "relu");
            classifier.AddLayer(64, 2, "sigmoid"); // Binary: object vs background
        }

        public List<(int x, int y, int width, int height, double confidence)> Detect(double[,] image, int stride = 8)
        {
            int height = image.GetLength(0);
            int width = image.GetLength(1);
            var detections = new List<(int, int, int, int, double)>();

            for (int y = 0; y <= height - windowSize; y += stride)
            {
                for (int x = 0; x <= width - windowSize; x += stride)
                {
                    var window = ExtractWindow(image, x, y, windowSize);
                    var features = FlattenWindow(window);
                    var prediction = classifier.Predict(features);

                    double confidence = prediction[1]; // Probability of object
                    if (confidence > 0.7)
                    {
                        detections.Add((x, y, windowSize, windowSize, confidence));
                    }
                }
            }

            // Non-maximum suppression
            return NMS(detections, 0.5);
        }

        private double[,] ExtractWindow(double[,] image, int x, int y, int size)
        {
            var window = new double[size, size];
            for (int wy = 0; wy < size; wy++)
                for (int wx = 0; wx < size; wx++)
                    window[wy, wx] = image[y + wy, x + wx];
            return window;
        }

        private double[] FlattenWindow(double[,] window)
        {
            int size = window.GetLength(0);
            var flattened = new double[size * size];
            for (int y = 0; y < size; y++)
                for (int x = 0; x < size; x++)
                    flattened[y * size + x] = window[y, x];
            return flattened;
        }

        private List<(int x, int y, int w, int h, double conf)> NMS(
            List<(int x, int y, int w, int h, double conf)> boxes, double iouThreshold)
        {
            var result = new List<(int, int, int, int, double)>();
            var sorted = boxes.OrderByDescending(b => b.conf).ToList();

            while (sorted.Count > 0)
            {
                var best = sorted[0];
                result.Add(best);
                sorted.RemoveAt(0);

                sorted = sorted.Where(box =>
                {
                    double iou = ComputeIoU(best, box);
                    return iou < iouThreshold;
                }).ToList();
            }

            return result;
        }

        private double ComputeIoU((int x, int y, int w, int h, double conf) box1,
                                  (int x, int y, int w, int h, double conf) box2)
        {
            int x1 = Math.Max(box1.x, box2.x);
            int y1 = Math.Max(box1.y, box2.y);
            int x2 = Math.Min(box1.x + box1.w, box2.x + box2.w);
            int y2 = Math.Min(box1.y + box1.h, box2.y + box2.h);

            int intersection = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
            int union = box1.w * box1.h + box2.w * box2.h - intersection;

            return union > 0 ? intersection / (double)union : 0;
        }
    }

    /// <summary>
    /// Image segmentation
    /// </summary>
    public class Segmentation
    {
        public static int[,] Threshold(double[,] image, double threshold = 0.5)
        {
            int height = image.GetLength(0);
            int width = image.GetLength(1);
            var result = new int[height, width];

            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    result[y, x] = image[y, x] > threshold ? 1 : 0;

            return result;
        }

        public static int[,] RegionGrowing(double[,] image, int seedX, int seedY, double threshold = 0.1)
        {
            int height = image.GetLength(0);
            int width = image.GetLength(1);
            var segmented = new int[height, width];
            var visited = new bool[height, width];
            var queue = new Queue<(int x, int y)>();

            double seedValue = image[seedY, seedX];
            queue.Enqueue((seedX, seedY));
            visited[seedY, seedX] = true;

            while (queue.Count > 0)
            {
                var (x, y) = queue.Dequeue();
                segmented[y, x] = 1;

                // Check 4-connected neighbors
                foreach (var (dx, dy) in new[] { (-1, 0), (1, 0), (0, -1), (0, 1) })
                {
                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx >= 0 && nx < width && ny >= 0 && ny < height && !visited[ny, nx])
                    {
                        visited[ny, nx] = true;
                        if (Math.Abs(image[ny, nx] - seedValue) < threshold)
                        {
                            queue.Enqueue((nx, ny));
                        }
                    }
                }
            }

            return segmented;
        }
    }

    /// <summary>
    /// Histogram of Oriented Gradients (HOG)
    /// </summary>
    public class HOG
    {
        public static double[] ComputeHOG(double[,] image, int cellSize = 8, int bins = 9)
        {
            int height = image.GetLength(0);
            int width = image.GetLength(1);
            var features = new List<double>();

            // Compute gradients
            var magnitude = new double[height, width];
            var angle = new double[height, width];

            for (int y = 1; y < height - 1; y++)
            {
                for (int x = 1; x < width - 1; x++)
                {
                    double gx = image[y, x + 1] - image[y, x - 1];
                    double gy = image[y + 1, x] - image[y - 1, x];
                    magnitude[y, x] = Math.Sqrt(gx * gx + gy * gy);
                    angle[y, x] = Math.Atan2(gy, gx);
                }
            }

            // Create histograms for each cell
            for (int cy = 0; cy < height / cellSize; cy++)
            {
                for (int cx = 0; cx < width / cellSize; cx++)
                {
                    var histogram = new double[bins];

                    for (int py = 0; py < cellSize; py++)
                    {
                        for (int px = 0; px < cellSize; px++)
                        {
                            int y = cy * cellSize + py;
                            int x = cx * cellSize + px;

                            if (y < height && x < width)
                            {
                                double ang = (angle[y, x] + Math.PI) / (2 * Math.PI) * bins;
                                int bin = ((int)ang) % bins;
                                histogram[bin] += magnitude[y, x];
                            }
                        }
                    }

                    features.AddRange(histogram);
                }
            }

            // Normalize
            double norm = Math.Sqrt(features.Sum(f => f * f));
            if (norm > 0)
                features = features.Select(f => f / norm).ToList();

            return features.ToArray();
        }
    }
}
