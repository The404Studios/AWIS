using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Threading;
using System.Threading.Tasks;
using WindowsInput;
using WindowsInput.Native;

namespace AWIS.Input
{
    /// <summary>
    /// Humanized mouse and keyboard controller with natural movements
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class HumanizedInputController
    {
        private readonly InputSimulator inputSimulator;
        private readonly Random random;
        private Point currentMousePosition;

        // Humanization parameters
        private const int MIN_MOVE_DELAY = 10;
        private const int MAX_MOVE_DELAY = 30;
        private const int MIN_CLICK_DELAY = 50;
        private const int MAX_CLICK_DELAY = 150;
        private const double OVERSHOOT_PROBABILITY = 0.15;
        private const int OVERSHOOT_PIXELS = 5;

        public HumanizedInputController()
        {
            inputSimulator = new InputSimulator();
            random = new Random();
            UpdateCurrentMousePosition();
        }

        /// <summary>
        /// Move mouse to target with humanized Bezier curve path
        /// </summary>
        public async Task MoveMouse(int targetX, int targetY, double speedFactor = 1.0)
        {
            UpdateCurrentMousePosition();
            var start = currentMousePosition;
            var target = new Point(targetX, targetY);

            // Apply random overshoot
            if (random.NextDouble() < OVERSHOOT_PROBABILITY)
            {
                var overshootX = target.X + random.Next(-OVERSHOOT_PIXELS, OVERSHOOT_PIXELS);
                var overshootY = target.Y + random.Next(-OVERSHOOT_PIXELS, OVERSHOOT_PIXELS);
                await MoveAlongBezierCurve(start, new Point(overshootX, overshootY), speedFactor);
                await Task.Delay(random.Next(20, 50));
                start = new Point(overshootX, overshootY);
            }

            // Move to final target
            await MoveAlongBezierCurve(start, target, speedFactor);
            currentMousePosition = target;
        }

        /// <summary>
        /// Move mouse along a Bezier curve for natural path
        /// </summary>
        private async Task MoveAlongBezierCurve(Point start, Point end, double speedFactor)
        {
            // Generate control points for Bezier curve
            var controlPoint1 = new Point(
                start.X + (end.X - start.X) / 3 + random.Next(-30, 30),
                start.Y + (end.Y - start.Y) / 3 + random.Next(-30, 30)
            );

            var controlPoint2 = new Point(
                start.X + 2 * (end.X - start.X) / 3 + random.Next(-30, 30),
                start.Y + 2 * (end.Y - start.Y) / 3 + random.Next(-30, 30)
            );

            // Calculate distance for step count
            double distance = Math.Sqrt(Math.Pow(end.X - start.X, 2) + Math.Pow(end.Y - start.Y, 2));
            int steps = Math.Max(10, (int)(distance / (5 * speedFactor)));

            for (int i = 0; i <= steps; i++)
            {
                double t = i / (double)steps;
                var point = CalculateBezierPoint(t, start, controlPoint1, controlPoint2, end);

                SetCursorPos(point.X, point.Y);

                // Add random micro-delays for human-like movement
                int delay = (int)(random.Next(MIN_MOVE_DELAY, MAX_MOVE_DELAY) / speedFactor);
                await Task.Delay(delay);
            }
        }

        /// <summary>
        /// Calculate point on cubic Bezier curve
        /// </summary>
        private Point CalculateBezierPoint(double t, Point p0, Point p1, Point p2, Point p3)
        {
            double u = 1 - t;
            double tt = t * t;
            double uu = u * u;
            double uuu = uu * u;
            double ttt = tt * t;

            int x = (int)(uuu * p0.X + 3 * uu * t * p1.X + 3 * u * tt * p2.X + ttt * p3.X);
            int y = (int)(uuu * p0.Y + 3 * uu * t * p1.Y + 3 * u * tt * p2.Y + ttt * p3.Y);

            return new Point(x, y);
        }

        /// <summary>
        /// Humanized mouse click with random delay
        /// </summary>
        public async Task Click(MouseButton button = MouseButton.Left)
        {
            // Pre-click delay
            await Task.Delay(random.Next(MIN_CLICK_DELAY, MAX_CLICK_DELAY));

            // Perform click
            switch (button)
            {
                case MouseButton.Left:
                    inputSimulator.Mouse.LeftButtonDown();
                    await Task.Delay(random.Next(50, 120)); // Hold time
                    inputSimulator.Mouse.LeftButtonUp();
                    break;
                case MouseButton.Right:
                    inputSimulator.Mouse.RightButtonDown();
                    await Task.Delay(random.Next(50, 120));
                    inputSimulator.Mouse.RightButtonUp();
                    break;
                case MouseButton.Middle:
                    inputSimulator.Mouse.MiddleButtonDown();
                    await Task.Delay(random.Next(50, 120));
                    inputSimulator.Mouse.MiddleButtonUp();
                    break;
            }

            // Post-click delay
            await Task.Delay(random.Next(MIN_CLICK_DELAY, MAX_CLICK_DELAY));
        }

        /// <summary>
        /// Humanized double click
        /// </summary>
        public async Task DoubleClick()
        {
            await Click(MouseButton.Left);
            await Task.Delay(random.Next(50, 150));
            await Click(MouseButton.Left);
        }

        /// <summary>
        /// Type text with human-like delays
        /// </summary>
        public async Task TypeText(string text)
        {
            foreach (char c in text)
            {
                inputSimulator.Keyboard.TextEntry(c);
                await Task.Delay(random.Next(50, 150)); // Typing speed variation
            }
        }

        /// <summary>
        /// Press a key with humanized timing
        /// </summary>
        public async Task PressKey(VirtualKeyCode key, int holdDuration = 0)
        {
            inputSimulator.Keyboard.KeyDown(key);
            await Task.Delay(holdDuration > 0 ? holdDuration : random.Next(50, 120));
            inputSimulator.Keyboard.KeyUp(key);
            await Task.Delay(random.Next(30, 80));
        }

        /// <summary>
        /// Press multiple keys simultaneously (for shortcuts)
        /// </summary>
        public async Task PressKeys(params VirtualKeyCode[] keys)
        {
            // Press all keys down
            foreach (var key in keys)
            {
                inputSimulator.Keyboard.KeyDown(key);
                await Task.Delay(random.Next(10, 30));
            }

            await Task.Delay(random.Next(50, 100));

            // Release all keys up (in reverse order)
            foreach (var key in keys.Reverse())
            {
                inputSimulator.Keyboard.KeyUp(key);
                await Task.Delay(random.Next(10, 30));
            }
        }

        /// <summary>
        /// Scroll mouse wheel with humanized behavior
        /// </summary>
        public async Task Scroll(int amount, bool horizontal = false)
        {
            int scrolls = Math.Abs(amount);
            int direction = amount > 0 ? 1 : -1;

            for (int i = 0; i < scrolls; i++)
            {
                if (horizontal)
                {
                    inputSimulator.Mouse.HorizontalScroll(direction);
                }
                else
                {
                    inputSimulator.Mouse.VerticalScroll(direction);
                }
                await Task.Delay(random.Next(50, 150));
            }
        }

        private void UpdateCurrentMousePosition()
        {
            if (GetCursorPos(out POINT lpPoint))
            {
                currentMousePosition = new Point(lpPoint.X, lpPoint.Y);
            }
        }

        // Win32 API imports
        [DllImport("user32.dll")]
        private static extern bool SetCursorPos(int X, int Y);

        [DllImport("user32.dll")]
        private static extern bool GetCursorPos(out POINT lpPoint);

        [StructLayout(LayoutKind.Sequential)]
        private struct POINT
        {
            public int X;
            public int Y;
        }
    }

    public enum MouseButton
    {
        Left,
        Right,
        Middle
    }
}
