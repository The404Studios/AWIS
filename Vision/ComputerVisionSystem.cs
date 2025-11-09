using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.Versioning;

namespace AWIS.Vision
{
    /// <summary>
    /// Detected object in an image
    /// </summary>
    public class DetectedObject
    {
        public string Label { get; set; } = string.Empty;
        public Rectangle BoundingBox { get; set; }
        public double Confidence { get; set; }
        public Dictionary<string, object> Attributes { get; set; } = new();
    }

    /// <summary>
    /// Result of text extraction (OCR)
    /// </summary>
    public class TextExtractionResult
    {
        public string Text { get; set; } = string.Empty;
        public List<(string text, Rectangle bounds, double confidence)> Words { get; set; } = new();
        public double AverageConfidence { get; set; }
    }

    /// <summary>
    /// Advanced computer vision system for image analysis
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class AdvancedComputerVision
    {
        private readonly int screenWidth;
        private readonly int screenHeight;

        public AdvancedComputerVision(int screenWidth = 1920, int screenHeight = 1080)
        {
            this.screenWidth = screenWidth;
            this.screenHeight = screenHeight;
        }

        /// <summary>
        /// Captures a screenshot
        /// </summary>
        public Bitmap CaptureScreen()
        {
            var bitmap = new Bitmap(screenWidth, screenHeight);
            using var graphics = Graphics.FromImage(bitmap);
            graphics.CopyFromScreen(0, 0, 0, 0, bitmap.Size);
            return bitmap;
        }

        /// <summary>
        /// Captures a specific region of the screen
        /// </summary>
        public Bitmap CaptureScreen(Rectangle region)
        {
            var bitmap = new Bitmap(region.Width, region.Height);
            using var graphics = Graphics.FromImage(bitmap);
            graphics.CopyFromScreen(region.X, region.Y, 0, 0, bitmap.Size);
            return bitmap;
        }

        /// <summary>
        /// Detects objects in an image (simulated detection)
        /// </summary>
        public List<DetectedObject> DetectObjects(Bitmap image, double confidenceThreshold = 0.5)
        {
            var detectedObjects = new List<DetectedObject>();

            // Analyze image colors and create simple detections
            var colorRegions = AnalyzeColorRegions(image);

            foreach (var region in colorRegions)
            {
                var obj = new DetectedObject
                {
                    Label = DetermineObjectType(region.color),
                    BoundingBox = region.bounds,
                    Confidence = region.confidence
                };

                if (obj.Confidence >= confidenceThreshold)
                {
                    obj.Attributes["Color"] = region.color.Name;
                    obj.Attributes["Area"] = region.bounds.Width * region.bounds.Height;
                    detectedObjects.Add(obj);
                }
            }

            return detectedObjects;
        }

        /// <summary>
        /// Extracts text from an image (simulated OCR)
        /// </summary>
        public TextExtractionResult ExtractText(Bitmap image)
        {
            // This is a simplified simulation
            // In a real implementation, you would use Tesseract or similar
            var result = new TextExtractionResult
            {
                Text = "Simulated OCR text extraction",
                AverageConfidence = 0.85
            };

            // Simulate word detection
            result.Words.Add(("Simulated", new Rectangle(10, 10, 100, 20), 0.90));
            result.Words.Add(("OCR", new Rectangle(120, 10, 50, 20), 0.85));
            result.Words.Add(("text", new Rectangle(180, 10, 60, 20), 0.80));
            result.Words.Add(("extraction", new Rectangle(250, 10, 120, 20), 0.85));

            return result;
        }

        /// <summary>
        /// Finds a specific color in an image
        /// </summary>
        public List<Rectangle> FindColorRegions(Bitmap image, Color targetColor, int tolerance = 30)
        {
            var regions = new List<Rectangle>();
            var visited = new bool[image.Width, image.Height];

            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    if (visited[x, y]) continue;

                    var pixel = image.GetPixel(x, y);
                    if (IsColorMatch(pixel, targetColor, tolerance))
                    {
                        var region = FloodFillRegion(image, x, y, targetColor, tolerance, visited);
                        if (region.Width > 5 && region.Height > 5) // Minimum size
                        {
                            regions.Add(region);
                        }
                    }
                }
            }

            return regions;
        }

        /// <summary>
        /// Detects edges in an image using a simple algorithm
        /// </summary>
        public Bitmap DetectEdges(Bitmap image)
        {
            var result = new Bitmap(image.Width, image.Height);

            for (int y = 1; y < image.Height - 1; y++)
            {
                for (int x = 1; x < image.Width - 1; x++)
                {
                    // Simple Sobel-like edge detection
                    var center = image.GetPixel(x, y);
                    var right = image.GetPixel(x + 1, y);
                    var bottom = image.GetPixel(x, y + 1);

                    var dx = Math.Abs(center.R - right.R);
                    var dy = Math.Abs(center.R - bottom.R);
                    var edgeStrength = Math.Min(255, dx + dy);

                    result.SetPixel(x, y, Color.FromArgb(edgeStrength, edgeStrength, edgeStrength));
                }
            }

            return result;
        }

        /// <summary>
        /// Analyzes dominant colors in an image
        /// </summary>
        public Dictionary<Color, int> AnalyzeDominantColors(Bitmap image, int maxColors = 10)
        {
            var colorCounts = new Dictionary<Color, int>();

            // Sample pixels (not every pixel for performance)
            int step = Math.Max(1, image.Width / 100);

            for (int y = 0; y < image.Height; y += step)
            {
                for (int x = 0; x < image.Width; x += step)
                {
                    var pixel = image.GetPixel(x, y);
                    var quantized = QuantizeColor(pixel);

                    if (colorCounts.ContainsKey(quantized))
                    {
                        colorCounts[quantized]++;
                    }
                    else
                    {
                        colorCounts[quantized] = 1;
                    }
                }
            }

            return colorCounts
                .OrderByDescending(kvp => kvp.Value)
                .Take(maxColors)
                .ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }

        private List<(Color color, Rectangle bounds, double confidence)> AnalyzeColorRegions(Bitmap image)
        {
            var regions = new List<(Color, Rectangle, double)>();
            var dominantColors = AnalyzeDominantColors(image, 5);

            foreach (var colorPair in dominantColors)
            {
                var colorRegions = FindColorRegions(image, colorPair.Key, 50);
                foreach (var bounds in colorRegions.Take(3)) // Top 3 regions per color
                {
                    var confidence = 0.5 + (colorPair.Value / 1000.0); // Based on frequency
                    regions.Add((colorPair.Key, bounds, Math.Min(0.95, confidence)));
                }
            }

            return regions;
        }

        private string DetermineObjectType(Color color)
        {
            // Simple color-based object type detection
            var hue = color.GetHue();
            var saturation = color.GetSaturation();
            var brightness = color.GetBrightness();

            if (saturation < 0.2)
            {
                return brightness > 0.8 ? "UI_Element_Light" :
                       brightness < 0.2 ? "UI_Element_Dark" : "UI_Element_Gray";
            }

            return hue switch
            {
                >= 0 and < 30 => "Button_Red",
                >= 30 and < 90 => "Button_Yellow",
                >= 90 and < 150 => "Button_Green",
                >= 150 and < 210 => "Button_Cyan",
                >= 210 and < 270 => "Button_Blue",
                >= 270 and < 330 => "Button_Purple",
                _ => "Button_Red"
            };
        }

        private bool IsColorMatch(Color c1, Color c2, int tolerance)
        {
            return Math.Abs(c1.R - c2.R) <= tolerance &&
                   Math.Abs(c1.G - c2.G) <= tolerance &&
                   Math.Abs(c1.B - c2.B) <= tolerance;
        }

        private Rectangle FloodFillRegion(Bitmap image, int startX, int startY, Color targetColor,
                                         int tolerance, bool[,] visited)
        {
            var queue = new Queue<(int x, int y)>();
            queue.Enqueue((startX, startY));
            visited[startX, startY] = true;

            int minX = startX, maxX = startX;
            int minY = startY, maxY = startY;

            while (queue.Count > 0 && queue.Count < 10000) // Limit flood fill size
            {
                var (x, y) = queue.Dequeue();

                minX = Math.Min(minX, x);
                maxX = Math.Max(maxX, x);
                minY = Math.Min(minY, y);
                maxY = Math.Max(maxY, y);

                // Check neighbors
                foreach (var (dx, dy) in new[] { (-1, 0), (1, 0), (0, -1), (0, 1) })
                {
                    int nx = x + dx, ny = y + dy;
                    if (nx >= 0 && nx < image.Width && ny >= 0 && ny < image.Height && !visited[nx, ny])
                    {
                        var pixel = image.GetPixel(nx, ny);
                        if (IsColorMatch(pixel, targetColor, tolerance))
                        {
                            visited[nx, ny] = true;
                            queue.Enqueue((nx, ny));
                        }
                    }
                }
            }

            return new Rectangle(minX, minY, maxX - minX + 1, maxY - minY + 1);
        }

        private Color QuantizeColor(Color color)
        {
            // Quantize to reduce color space
            int step = 32;
            return Color.FromArgb(
                (color.R / step) * step,
                (color.G / step) * step,
                (color.B / step) * step
            );
        }
    }

    /// <summary>
    /// Simpler computer vision for basic operations
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class ComputerVision
    {
        private readonly AdvancedComputerVision advanced;

        public ComputerVision()
        {
            advanced = new AdvancedComputerVision();
        }

        public Bitmap CaptureScreen() => advanced.CaptureScreen();

        public List<DetectedObject> DetectObjects(Bitmap image) => advanced.DetectObjects(image);

        public string ExtractText(Bitmap image) => advanced.ExtractText(image).Text;

        public List<Rectangle> FindColor(Bitmap image, Color color) =>
            advanced.FindColorRegions(image, color);
    }
}
