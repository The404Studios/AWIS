# 🧠 AWIS - Autonomous Web Intelligence System

<div align="center">

[![Version](https://img.shields.io/badge/Version-7.0-blue.svg)](https://github.com/The404Studios/AWIS)
[![.NET](https://img.shields.io/badge/.NET-6.0-purple.svg)](https://dotnet.microsoft.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Experimental-orange.svg)]()

**A truly autonomous AI that sees, learns, plays, and socializes - making its own decisions and improving through real experience**

[Features](#-features) • [How It Works](#-how-it-works) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture)

</div>

---

## 🌟 What Makes AWIS Special?

AWIS is not just another bot - it's an autonomous AI agent with:

- **👁️ Real Computer Vision**: Captures and understands what's on your screen using OpenCV
- **🎮 Game Playing AI**: Actually learns to play video games through reinforcement learning
- **🎨 Visual Overlay**: Shows you exactly what the AI sees and thinks in real-time
- **🧠 True Learning**: Uses Q-learning with experience replay to genuinely improve over time
- **🎯 Autonomous Goals**: Decides its own objectives and pursues them independently
- **💬 Social Intelligence**: Chats, responds, and learns from conversations with sentiment analysis
- **🎮 Input Simulation**: Can control mouse, keyboard to actually interact with games and applications
- **📊 Persistent Memory**: Remembers experiences and learns from past successes and failures

## ✨ Core Features

### 🎮 Autonomous Game Playing

The AI can:
- **See the game**: Captures screen in real-time
- **Understand the visuals**: Detects objects, UI elements, and text
- **Make decisions**: Uses reinforcement learning to choose actions
- **Learn from experience**: Improves through Q-learning and experience replay
- **Control inputs**: Simulates keyboard and mouse to actually play

### 👁️ Advanced Computer Vision

- **Screen Capture**: Full-screen or region-specific capture
- **Object Detection**: Identifies UI elements, buttons, and game objects using edge detection
- **OCR**: Reads text from screen using Tesseract
- **Color Recognition**: Detects UI elements by color patterns
- **Feature Extraction**: Converts visual information into 1024-dimensional feature vectors

### 🎨 Real-Time Visualization Overlay

A transparent overlay shows:
- **Detected Objects**: Highlighted bounding boxes around identified elements
- **Confidence Scores**: How certain the AI is about each detection
- **Current Action**: What the AI is doing right now
- **Goal Progress**: Active goals and their completion status
- **Reasoning**: Why the AI chose this action
- **Learning Stats**: Cycle count, rewards, object count

### 🧠 Reinforcement Learning System

- **Q-Learning Algorithm**: Learns optimal action-value function
- **Experience Replay**: Stores and replays past experiences for better learning
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation
- **Reward System**: Learns from success and failure
- **Persistent Q-Table**: Saves learning progress between sessions
- **Goal-Aligned Actions**: Adjusts behavior based on current objectives

### 🎯 Autonomous Goal System

The AI creates and pursues its own goals:
- **PlayGame**: Learn to master video games
- **SocializeInChat**: Engage in conversations
- **LearnSkill**: Acquire new abilities
- **ExploreWeb**: Navigate and discover
- **CreateContent**: Generate creative outputs
- **SelfImprove**: Optimize its own performance

Goals have priority, progress tracking, and completion detection.

### 💬 Chat & Social System

- **Natural Conversation**: Context-aware responses
- **Sentiment Analysis**: Understands emotional tone
- **Voice Output**: Text-to-speech for responses
- **Message History**: Remembers conversation context
- **Personality**: Friendly, curious, and eager to learn

### 🔧 Input Simulation

- **Mouse Control**: Move cursor and click precisely
- **Keyboard Input**: Press keys and type text
- **Scroll Actions**: Navigate pages
- **Game Controls**: Arrow keys, space, WASD, etc.

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────┐
│          AWIS Core Orchestrator             │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │ Vision │  │Learning│  │ Goals  │
   │ System │  │ Engine │  │ System │
   └────────┘  └────────┘  └────────┘
        │           │           │
        │           │           │
        ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │  Game  │  │  Chat  │  │Visual- │
   │Control │  │ System │  │ization │
   └────────┘  └────────┘  └────────┘
```

### Components:

1. **VisionSystem**: Screen capture, OCR, object detection, feature extraction
2. **ReinforcementLearner**: Q-learning, experience replay, decision making
3. **GameController**: Input simulation (mouse, keyboard)
4. **AutonomousGoalSystem**: Goal generation, priority management, progress tracking
5. **ChatSystem**: NLP, sentiment analysis, conversation management
6. **VisualizationOverlay**: Transparent real-time display of AI's perception
7. **AWISCore**: Main orchestrator coordinating all systems

## 🚀 How It Works

### The Learning Loop

1. **Perceive**: Capture screen → Detect objects → Extract features → Read text
2. **Decide**: Check goals → Query Q-table → Choose action (explore/exploit)
3. **Act**: Simulate input (click/type/press keys)
4. **Observe**: Capture new state
5. **Learn**: Calculate reward → Update Q-table → Store experience
6. **Replay**: Periodically replay past experiences to reinforce learning
7. **Visualize**: Update overlay to show what AI sees and thinks

### Reinforcement Learning

- **State**: Visual features + detected objects + screen text
- **Actions**: Click, KeyPress, TypeText, Scroll, Navigate, Chat, etc.
- **Rewards**:
  - +0.5 for discovering new objects
  - +0.3 for text interaction
  - +0.2 for exploration
  - +0.1 × confidence for high-confidence actions
  - -0.1 for inactivity
- **Q-Learning Update**: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

## 📦 Installation

### Prerequisites

- **OS**: Windows 10/11 (required for Windows Forms overlay and input simulation)
- **.NET**: .NET 6.0 SDK or higher
- **Tesseract OCR**: Trained data files in `tessdata/` directory

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/The404Studios/AWIS.git
cd AWIS
```

2. **Install Tesseract data**:
```bash
mkdir tessdata
# Download eng.traineddata from: https://github.com/tesseract-ocr/tessdata
# Place in tessdata/ folder
```

3. **Restore NuGet packages**:
```bash
dotnet restore
```

4. **Build the project**:
```bash
dotnet build
```

5. **Run AWIS**:
```bash
dotnet run
```

### NuGet Dependencies

- **Microsoft.ML** (3.0.1) - Machine learning framework
- **Emgu.CV** (4.8.1) - OpenCV wrapper for computer vision
- **Tesseract** (5.2.0) - OCR engine
- **InputSimulator** (1.0.4) - Keyboard/mouse input simulation
- **Newtonsoft.Json** (13.0.3) - JSON serialization
- **System.Speech** (9.0.8) - Text-to-speech
- **System.Drawing.Common** (7.0.0) - Graphics and image processing

## 🎮 Usage

### Basic Controls

When AWIS is running:

- **Q**: Quit and save
- **S**: Save current state (Q-table, goals, chat history)
- **C**: Chat with the AI

### What You'll See

1. **Console Output**:
   - Initialization messages
   - Cycle statistics (every 10 cycles)
   - Object detection count
   - Current action and confidence
   - Reward values
   - Goal progress

2. **Screen Overlay**:
   - Cyan boxes around detected objects
   - Blue boxes around buttons
   - Green boxes around text fields
   - Status panel (top-left) showing:
     - Current action
     - Confidence level
     - Reasoning
     - Active goals
     - Progress metrics

3. **Learning Progress**:
   - Watch the AI explore and learn
   - Observe decisions becoming more confident
   - See it adapt to achieve goals

### Example Session

```
🧠 Initializing AWIS - Autonomous Web Intelligence System
============================================================
✓ Vision system ready
✓ Learning system ready
✓ Controller ready
✓ Goal system ready
✓ Chat system ready
Loaded Q-table with 1523 states and 847 experiences

🚀 AWIS Started!
Press 'Q' to quit, 'S' to save, 'C' to chat

🎯 New Goal: Master a video game
   Learn to play a game by observing patterns and practicing actions

📊 Cycle 10
   Objects detected: 15
   Action: Click (confidence: 67%)
   Reward: 0.72
   Goal: Master a video game (12%)
```

### Chat Example

```
Press C to chat...
You: Hello!
💬 User: Hello!
🤖 AI: Hello! I'm AWIS, an autonomous AI learning to interact with the world!

You: What are you doing?
💬 User: What are you doing?
🤖 AI: I'm observing the screen, detecting objects, and learning to play games!
```

## 🧪 How the AI Learns

### Training Process

1. **Initial Exploration**: Random actions to discover the environment
2. **Pattern Recognition**: Identifies which actions lead to positive rewards
3. **Strategy Development**: Builds Q-table of optimal actions per state
4. **Exploitation**: Uses learned strategies while still exploring
5. **Continuous Improvement**: Refines through experience replay

### Learning Files

- **qtable.json**: Stores learned action values for different states
- **goals.json**: Saves active goals and progress
- **chat_history.json**: Conversation history (last 1000 messages)

### Performance Metrics

Monitor learning through:
- Q-table size (number of learned states)
- Experience replay buffer size
- Average rewards over time
- Goal completion rate
- Action confidence scores

## 🎯 Use Cases

### Game Playing
- **Platformers**: Learn to jump, move, avoid obstacles
- **Puzzle Games**: Pattern recognition and problem solving
- **Strategy Games**: Long-term planning and decision making

### Web Automation
- **Form Filling**: Detect text fields and input data
- **Navigation**: Click links, scroll pages
- **Data Extraction**: Read and process screen text

### Social Interaction
- **Chat Bots**: Engage in conversations
- **Customer Service**: Answer questions
- **Community Management**: Monitor and respond to messages

### Research
- **Reinforcement Learning**: Test RL algorithms
- **Computer Vision**: Object detection research
- **Human-AI Interaction**: Study autonomous agent behavior

## ⚙️ Configuration

### Tunable Parameters

In `ReinforcementLearner`:
- `LearningRate` (α): 0.1 - How quickly to learn from new experiences
- `DiscountFactor` (γ): 0.95 - How much to value future rewards
- `ExplorationRate` (ε): 0.2 - Probability of random exploration
- `MaxExperienceSize`: 10000 - Max experiences to store

### Vision Settings

In `VisionSystem`:
- Edge detection thresholds: 50, 150
- Object size filters: Min 20×20, Max screen/2
- UI element size: Min 30×15, Max screen/3

### Goal System

In `AutonomousGoalSystem`:
- Goal update interval: 5 minutes
- Progress increment: Random 0-5% per cycle
- Priority range: 0.5 to 1.0

## 🔬 Advanced Features

### Experience Replay
- Stores last 10,000 experiences
- Randomly samples 32 for batch learning
- Triggers on 10% of cycles when buffer > 100
- Uses reduced learning rate (α × 0.5) for stability

### Multi-Goal Planning
- Concurrent goal tracking
- Priority-based action selection
- Progress monitoring
- Automatic goal generation

### Visual Understanding
- 32×32 downsampled images
- 1024-dimensional feature vectors
- Edge-based object detection
- Color-based UI element recognition
- HSV color space for robustness

## 🛡️ Safety & Ethics

### Built-in Safeguards

- **Screen-only**: Only observes and interacts with screen content
- **User Control**: Easy pause/stop with 'Q' key
- **Transparent**: Overlay shows exactly what AI sees
- **Bounded**: Limited action space, controlled input simulation
- **Logged**: All actions and decisions are recorded

### Responsible Use

- Use only on systems you own or have permission to use
- Monitor AI behavior, especially during initial learning
- Don't use for malicious automation or spam
- Respect privacy and security of others

## 🐛 Troubleshooting

### Overlay not showing
- Ensure Windows, not Linux/Mac
- Check if running with admin privileges
- Verify `WindowState.Maximized` is supported

### OCR not working
- Verify `tessdata/eng.traineddata` exists
- Check file permissions
- Ensure Tesseract package is installed

### Input simulation failing
- Run as administrator
- Check if InputSimulator DLL is present
- Verify not blocked by antivirus

### High memory usage
- Reduce `MaxExperienceSize`
- Limit object detection count
- Clear Q-table periodically

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- **Deep Learning**: Replace Q-table with neural network
- **Better Vision**: Integrate YOLO or other object detection models
- **NLP Enhancement**: Add GPT integration for better chat
- **Multi-Agent**: Coordinate multiple AWIS instances
- **Reward Shaping**: Improve reward function
- **Transfer Learning**: Apply learning across games

## 📝 License

MIT License - See [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- **Tesseract**: OCR engine
- **Emgu.CV**: OpenCV wrapper
- **ML.NET**: Machine learning framework
- **InputSimulator**: Input automation library

## 📞 Contact

**Issues**: [GitHub Issues](https://github.com/The404Studios/AWIS/issues)

**Discussions**: [GitHub Discussions](https://github.com/The404Studios/AWIS/discussions)

---

<div align="center">

**⭐ Star this repo if you find it interesting!**

**Made with 🧠 by The404Studios**

*AWIS - Teaching machines to see, learn, and play*

</div>
