using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;
using System.Threading;
using System.Threading.Tasks;

namespace AWIS.Input
{
    /// <summary>
    /// Humanized mouse and keyboard controller with natural movements using Win32 APIs
    /// </summary>
    [SupportedOSPlatform("windows")]
    public class HumanizedInputController
    {
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

            // Perform click using Win32 mouse events
            uint downFlag = 0, upFlag = 0;
            switch (button)
            {
                case MouseButton.Left:
                    downFlag = MOUSEEVENTF_LEFTDOWN;
                    upFlag = MOUSEEVENTF_LEFTUP;
                    break;
                case MouseButton.Right:
                    downFlag = MOUSEEVENTF_RIGHTDOWN;
                    upFlag = MOUSEEVENTF_RIGHTUP;
                    break;
                case MouseButton.Middle:
                    downFlag = MOUSEEVENTF_MIDDLEDOWN;
                    upFlag = MOUSEEVENTF_MIDDLEUP;
                    break;
            }

            // Mouse down
            mouse_event(downFlag, 0, 0, 0, 0);
            await Task.Delay(random.Next(50, 120)); // Hold time

            // Mouse up
            mouse_event(upFlag, 0, 0, 0, 0);

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
                // Send character using keybd_event
                short vk = VkKeyScan(c);
                byte virtualKey = (byte)(vk & 0xFF);

                keybd_event(virtualKey, 0, 0, 0); // Key down
                await Task.Delay(random.Next(10, 30));
                keybd_event(virtualKey, 0, KEYEVENTF_KEYUP, 0); // Key up

                await Task.Delay(random.Next(50, 150)); // Typing speed variation
            }
        }

        /// <summary>
        /// Press a key with humanized timing
        /// </summary>
        public async Task PressKey(byte virtualKey, int holdDuration = 0)
        {
            keybd_event(virtualKey, 0, 0, 0); // Key down
            await Task.Delay(holdDuration > 0 ? holdDuration : random.Next(50, 120));
            keybd_event(virtualKey, 0, KEYEVENTF_KEYUP, 0); // Key up
            await Task.Delay(random.Next(30, 80));
        }

        /// <summary>
        /// Press multiple keys simultaneously (for shortcuts)
        /// </summary>
        public async Task PressKeys(params byte[] keys)
        {
            // Press all keys down
            foreach (var key in keys)
            {
                keybd_event(key, 0, 0, 0);
                await Task.Delay(random.Next(10, 30));
            }

            await Task.Delay(random.Next(50, 100));

            // Release all keys up (in reverse order)
            foreach (var key in keys.Reverse())
            {
                keybd_event(key, 0, KEYEVENTF_KEYUP, 0);
                await Task.Delay(random.Next(10, 30));
            }
        }

        /// <summary>
        /// Scroll mouse wheel with humanized behavior
        /// </summary>
        public async Task Scroll(int amount, bool horizontal = false)
        {
            int scrolls = Math.Abs(amount);
            int direction = amount > 0 ? 120 : -120; // WHEEL_DELTA

            for (int i = 0; i < scrolls; i++)
            {
                if (horizontal)
                {
                    mouse_event(MOUSEEVENTF_HWHEEL, 0, 0, (uint)direction, 0);
                }
                else
                {
                    mouse_event(MOUSEEVENTF_WHEEL, 0, 0, (uint)direction, 0);
                }
                await Task.Delay(random.Next(50, 150));
            }
        }

        /// <summary>
        /// Move mouse for camera control (yaw/pitch axis)
        /// </summary>
        /// <param name="yaw">Horizontal rotation (-1.0 to 1.0, negative = left, positive = right)</param>
        /// <param name="pitch">Vertical rotation (-1.0 to 1.0, negative = down, positive = up)</param>
        /// <param name="sensitivity">Sensitivity multiplier (default 100 pixels per unit)</param>
        /// <param name="duration">Duration in milliseconds to apply the movement</param>
        public async Task MoveAxis(double yaw, double pitch, double sensitivity = 100.0, int duration = 100)
        {
            UpdateCurrentMousePosition();

            // Calculate total pixel movement
            int totalDeltaX = (int)(yaw * sensitivity);
            int totalDeltaY = (int)(-pitch * sensitivity); // Negative because screen Y is inverted

            // Calculate steps for smooth movement
            int steps = Math.Max(5, duration / 20); // At least 5 steps
            int stepDelay = duration / steps;

            for (int i = 0; i < steps; i++)
            {
                // Calculate movement for this step with some randomness
                double progress = (i + 1.0) / steps;
                int stepX = (int)(totalDeltaX * progress) - (int)(totalDeltaX * (i / (double)steps));
                int stepY = (int)(totalDeltaY * progress) - (int)(totalDeltaY * (i / (double)steps));

                // Add micro-variations for human-like jitter
                stepX += random.Next(-1, 2);
                stepY += random.Next(-1, 2);

                // Move relative to current position
                if (GetCursorPos(out POINT current))
                {
                    SetCursorPos(current.X + stepX, current.Y + stepY);
                }

                await Task.Delay(stepDelay + random.Next(-5, 5));
            }
        }

        /// <summary>
        /// Continuous axis input (for holding a direction)
        /// </summary>
        public async Task HoldAxis(double yaw, double pitch, int holdDuration, double sensitivity = 100.0)
        {
            int elapsed = 0;
            int frameTime = 16; // ~60 FPS

            while (elapsed < holdDuration)
            {
                await MoveAxis(yaw, pitch, sensitivity, frameTime);
                elapsed += frameTime;
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

        [DllImport("user32.dll")]
        private static extern void mouse_event(uint dwFlags, uint dx, uint dy, uint dwData, int dwExtraInfo);

        [DllImport("user32.dll")]
        private static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, int dwExtraInfo);

        [DllImport("user32.dll")]
        private static extern short VkKeyScan(char ch);

        // Mouse event flags
        private const uint MOUSEEVENTF_LEFTDOWN = 0x0002;
        private const uint MOUSEEVENTF_LEFTUP = 0x0004;
        private const uint MOUSEEVENTF_RIGHTDOWN = 0x0008;
        private const uint MOUSEEVENTF_RIGHTUP = 0x0010;
        private const uint MOUSEEVENTF_MIDDLEDOWN = 0x0020;
        private const uint MOUSEEVENTF_MIDDLEUP = 0x0040;
        private const uint MOUSEEVENTF_WHEEL = 0x0800;
        private const uint MOUSEEVENTF_HWHEEL = 0x01000;

        // Keyboard event flags
        private const uint KEYEVENTF_KEYUP = 0x0002;

        [StructLayout(LayoutKind.Sequential)]
        private struct POINT
        {
            public int X;
            public int Y;
        }

        // Complete Virtual Key Codes
        public static class VK
        {
            // Mouse buttons
            public const byte LBUTTON = 0x01;
            public const byte RBUTTON = 0x02;
            public const byte CANCEL = 0x03;
            public const byte MBUTTON = 0x04;
            public const byte XBUTTON1 = 0x05;
            public const byte XBUTTON2 = 0x06;

            // Control keys
            public const byte BACK = 0x08;
            public const byte TAB = 0x09;
            public const byte CLEAR = 0x0C;
            public const byte RETURN = 0x0D;
            public const byte ENTER = 0x0D;
            public const byte SHIFT = 0x10;
            public const byte CONTROL = 0x11;
            public const byte CTRL = 0x11;
            public const byte MENU = 0x12;
            public const byte ALT = 0x12;
            public const byte PAUSE = 0x13;
            public const byte CAPITAL = 0x14;
            public const byte CAPSLOCK = 0x14;
            public const byte ESCAPE = 0x1B;
            public const byte ESC = 0x1B;
            public const byte SPACE = 0x20;

            // Navigation keys
            public const byte PRIOR = 0x21;
            public const byte PAGEUP = 0x21;
            public const byte NEXT = 0x22;
            public const byte PAGEDOWN = 0x22;
            public const byte END = 0x23;
            public const byte HOME = 0x24;
            public const byte LEFT = 0x25;
            public const byte UP = 0x26;
            public const byte RIGHT = 0x27;
            public const byte DOWN = 0x28;
            public const byte SELECT = 0x29;
            public const byte PRINT = 0x2A;
            public const byte EXECUTE = 0x2B;
            public const byte SNAPSHOT = 0x2C;
            public const byte PRINTSCREEN = 0x2C;
            public const byte INSERT = 0x2D;
            public const byte DELETE = 0x2E;
            public const byte HELP = 0x2F;

            // Number keys (top row)
            public const byte VK_0 = 0x30;
            public const byte VK_1 = 0x31;
            public const byte VK_2 = 0x32;
            public const byte VK_3 = 0x33;
            public const byte VK_4 = 0x34;
            public const byte VK_5 = 0x35;
            public const byte VK_6 = 0x36;
            public const byte VK_7 = 0x37;
            public const byte VK_8 = 0x38;
            public const byte VK_9 = 0x39;

            // Letter keys A-Z
            public const byte A = 0x41;
            public const byte B = 0x42;
            public const byte C = 0x43;
            public const byte D = 0x44;
            public const byte E = 0x45;
            public const byte F = 0x46;
            public const byte G = 0x47;
            public const byte H = 0x48;
            public const byte I = 0x49;
            public const byte J = 0x4A;
            public const byte K = 0x4B;
            public const byte L = 0x4C;
            public const byte M = 0x4D;
            public const byte N = 0x4E;
            public const byte O = 0x4F;
            public const byte P = 0x50;
            public const byte Q = 0x51;
            public const byte R = 0x52;
            public const byte S = 0x53;
            public const byte T = 0x54;
            public const byte U = 0x55;
            public const byte V = 0x56;
            public const byte W = 0x57;
            public const byte X = 0x58;
            public const byte Y = 0x59;
            public const byte Z = 0x5A;

            // Windows keys
            public const byte LWIN = 0x5B;
            public const byte RWIN = 0x5C;
            public const byte APPS = 0x5D;
            public const byte SLEEP = 0x5F;

            // Numpad keys
            public const byte NUMPAD0 = 0x60;
            public const byte NUMPAD1 = 0x61;
            public const byte NUMPAD2 = 0x62;
            public const byte NUMPAD3 = 0x63;
            public const byte NUMPAD4 = 0x64;
            public const byte NUMPAD5 = 0x65;
            public const byte NUMPAD6 = 0x66;
            public const byte NUMPAD7 = 0x67;
            public const byte NUMPAD8 = 0x68;
            public const byte NUMPAD9 = 0x69;
            public const byte MULTIPLY = 0x6A;
            public const byte ADD = 0x6B;
            public const byte SEPARATOR = 0x6C;
            public const byte SUBTRACT = 0x6D;
            public const byte DECIMAL = 0x6E;
            public const byte DIVIDE = 0x6F;

            // Function keys
            public const byte F1 = 0x70;
            public const byte F2 = 0x71;
            public const byte F3 = 0x72;
            public const byte F4 = 0x73;
            public const byte F5 = 0x74;
            public const byte F6 = 0x75;
            public const byte F7 = 0x76;
            public const byte F8 = 0x77;
            public const byte F9 = 0x78;
            public const byte F10 = 0x79;
            public const byte F11 = 0x7A;
            public const byte F12 = 0x7B;
            public const byte F13 = 0x7C;
            public const byte F14 = 0x7D;
            public const byte F15 = 0x7E;
            public const byte F16 = 0x7F;
            public const byte F17 = 0x80;
            public const byte F18 = 0x81;
            public const byte F19 = 0x82;
            public const byte F20 = 0x83;
            public const byte F21 = 0x84;
            public const byte F22 = 0x85;
            public const byte F23 = 0x86;
            public const byte F24 = 0x87;

            // Lock keys
            public const byte NUMLOCK = 0x90;
            public const byte SCROLL = 0x91;
            public const byte SCROLLLOCK = 0x91;

            // Shift/Control/Alt modifiers (left/right specific)
            public const byte LSHIFT = 0xA0;
            public const byte RSHIFT = 0xA1;
            public const byte LCONTROL = 0xA2;
            public const byte LCTRL = 0xA2;
            public const byte RCONTROL = 0xA3;
            public const byte RCTRL = 0xA3;
            public const byte LMENU = 0xA4;
            public const byte LALT = 0xA4;
            public const byte RMENU = 0xA5;
            public const byte RALT = 0xA5;

            // Browser keys
            public const byte BROWSER_BACK = 0xA6;
            public const byte BROWSER_FORWARD = 0xA7;
            public const byte BROWSER_REFRESH = 0xA8;
            public const byte BROWSER_STOP = 0xA9;
            public const byte BROWSER_SEARCH = 0xAA;
            public const byte BROWSER_FAVORITES = 0xAB;
            public const byte BROWSER_HOME = 0xAC;

            // Volume keys
            public const byte VOLUME_MUTE = 0xAD;
            public const byte VOLUME_DOWN = 0xAE;
            public const byte VOLUME_UP = 0xAF;

            // Media keys
            public const byte MEDIA_NEXT_TRACK = 0xB0;
            public const byte MEDIA_PREV_TRACK = 0xB1;
            public const byte MEDIA_STOP = 0xB2;
            public const byte MEDIA_PLAY_PAUSE = 0xB3;

            // Application keys
            public const byte LAUNCH_MAIL = 0xB4;
            public const byte LAUNCH_MEDIA_SELECT = 0xB5;
            public const byte LAUNCH_APP1 = 0xB6;
            public const byte LAUNCH_APP2 = 0xB7;

            // OEM keys (punctuation, varies by keyboard layout)
            public const byte OEM_1 = 0xBA;        // ';:' for US
            public const byte SEMICOLON = 0xBA;
            public const byte OEM_PLUS = 0xBB;     // '+' any country
            public const byte PLUS = 0xBB;
            public const byte OEM_COMMA = 0xBC;    // ',' any country
            public const byte COMMA = 0xBC;
            public const byte OEM_MINUS = 0xBD;    // '-' any country
            public const byte MINUS = 0xBD;
            public const byte OEM_PERIOD = 0xBE;   // '.' any country
            public const byte PERIOD = 0xBE;
            public const byte OEM_2 = 0xBF;        // '/?' for US
            public const byte SLASH = 0xBF;
            public const byte OEM_3 = 0xC0;        // '`~' for US
            public const byte TILDE = 0xC0;
            public const byte OEM_4 = 0xDB;        // '[{' for US
            public const byte OPENBRACKET = 0xDB;
            public const byte OEM_5 = 0xDC;        // '\|' for US
            public const byte BACKSLASH = 0xDC;
            public const byte OEM_6 = 0xDD;        // ']}' for US
            public const byte CLOSEBRACKET = 0xDD;
            public const byte OEM_7 = 0xDE;        // ''"' for US
            public const byte QUOTE = 0xDE;
            public const byte OEM_8 = 0xDF;
        }
    }

    public enum MouseButton
    {
        Left,
        Right,
        Middle
    }
}
