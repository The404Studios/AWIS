using System;
using System.Collections.Generic;
using System.Linq;

namespace AWIS.Core
{
    /// <summary>
    /// Represents an emotional state vector
    /// </summary>
    public class EmotionalVector
    {
        public double Joy { get; set; }
        public double Trust { get; set; }
        public double Fear { get; set; }
        public double Surprise { get; set; }
        public double Sadness { get; set; }
        public double Disgust { get; set; }
        public double Anger { get; set; }
        public double Anticipation { get; set; }

        public EmotionalVector()
        {
            // Start neutral
            Joy = Trust = Fear = Surprise = Sadness = Disgust = Anger = Anticipation = 0.0;
        }

        public EmotionalVector(double joy, double trust, double fear, double surprise,
                              double sadness, double disgust, double anger, double anticipation)
        {
            Joy = Clamp(joy);
            Trust = Clamp(trust);
            Fear = Clamp(fear);
            Surprise = Clamp(surprise);
            Sadness = Clamp(sadness);
            Disgust = Clamp(disgust);
            Anger = Clamp(anger);
            Anticipation = Clamp(anticipation);
        }

        private static double Clamp(double value, double min = -1.0, double max = 1.0)
        {
            return Math.Max(min, Math.Min(max, value));
        }

        public void Normalize()
        {
            Joy = Clamp(Joy);
            Trust = Clamp(Trust);
            Fear = Clamp(Fear);
            Surprise = Clamp(Surprise);
            Sadness = Clamp(Sadness);
            Disgust = Clamp(Disgust);
            Anger = Clamp(Anger);
            Anticipation = Clamp(Anticipation);
        }

        public EmotionalVector Add(EmotionalVector other, double weight = 1.0)
        {
            return new EmotionalVector(
                Joy + other.Joy * weight,
                Trust + other.Trust * weight,
                Fear + other.Fear * weight,
                Surprise + other.Surprise * weight,
                Sadness + other.Sadness * weight,
                Disgust + other.Disgust * weight,
                Anger + other.Anger * weight,
                Anticipation + other.Anticipation * weight
            );
        }

        public EmotionalVector Decay(double factor = 0.95)
        {
            return new EmotionalVector(
                Joy * factor,
                Trust * factor,
                Fear * factor,
                Surprise * factor,
                Sadness * factor,
                Disgust * factor,
                Anger * factor,
                Anticipation * factor
            );
        }

        public string GetDominantEmotion()
        {
            var emotions = new Dictionary<string, double>
            {
                ["Joy"] = Joy,
                ["Trust"] = Trust,
                ["Fear"] = Fear,
                ["Surprise"] = Surprise,
                ["Sadness"] = Sadness,
                ["Disgust"] = Disgust,
                ["Anger"] = Anger,
                ["Anticipation"] = Anticipation
            };

            var max = emotions.MaxBy(kvp => Math.Abs(kvp.Value));
            return max.Key;
        }

        public double GetValence()
        {
            // Overall positive/negative sentiment
            var positive = Joy + Trust + Anticipation;
            var negative = Fear + Sadness + Disgust + Anger;
            return (positive - negative) / 7.0;
        }

        public double GetArousal()
        {
            // Overall activation/energy level
            return (Math.Abs(Joy) + Math.Abs(Fear) + Math.Abs(Anger) + Math.Abs(Surprise)) / 4.0;
        }
    }

    /// <summary>
    /// Manages emotional responses based on experiences and context
    /// </summary>
    public class EmotionalSocketManager
    {
        private EmotionalVector currentState;
        private readonly List<(DateTime time, EmotionalVector state)> emotionalHistory = new();
        private readonly double decayRate = 0.98;
        private readonly int historyLimit = 1000;

        public EmotionalVector CurrentState => currentState;

        public EmotionalSocketManager()
        {
            currentState = new EmotionalVector();
        }

        public void ProcessExperience(Experience experience)
        {
            var emotionalResponse = ComputeEmotionalResponse(experience);
            UpdateState(emotionalResponse);
        }

        private EmotionalVector ComputeEmotionalResponse(Experience experience)
        {
            var emotion = new EmotionalVector();

            if (experience.Result.Success)
            {
                emotion.Joy = 0.5 * experience.Reward;
                emotion.Trust = 0.3;
                emotion.Anticipation = 0.2;
            }
            else
            {
                emotion.Sadness = 0.4;
                emotion.Fear = 0.2;
                emotion.Anger = -0.3 * experience.Reward;
            }

            // Surprise based on expectation
            if (Math.Abs(experience.Reward) > 0.7)
            {
                emotion.Surprise = 0.5;
            }

            return emotion;
        }

        public void UpdateState(EmotionalVector stimulus)
        {
            // Decay current state
            currentState = currentState.Decay(decayRate);

            // Add new stimulus
            currentState = currentState.Add(stimulus);
            currentState.Normalize();

            // Record history
            emotionalHistory.Add((DateTime.UtcNow, new EmotionalVector(
                currentState.Joy, currentState.Trust, currentState.Fear, currentState.Surprise,
                currentState.Sadness, currentState.Disgust, currentState.Anger, currentState.Anticipation
            )));

            // Prune history
            if (emotionalHistory.Count > historyLimit)
            {
                emotionalHistory.RemoveRange(0, emotionalHistory.Count - historyLimit);
            }
        }

        public EmotionalVector GetAverageEmotion(TimeSpan period)
        {
            var cutoff = DateTime.UtcNow - period;
            var recentStates = emotionalHistory.Where(h => h.time >= cutoff).ToList();

            if (!recentStates.Any())
            {
                return new EmotionalVector();
            }

            return new EmotionalVector(
                recentStates.Average(s => s.state.Joy),
                recentStates.Average(s => s.state.Trust),
                recentStates.Average(s => s.state.Fear),
                recentStates.Average(s => s.state.Surprise),
                recentStates.Average(s => s.state.Sadness),
                recentStates.Average(s => s.state.Disgust),
                recentStates.Average(s => s.state.Anger),
                recentStates.Average(s => s.state.Anticipation)
            );
        }

        public string GetMoodReport()
        {
            var dominant = currentState.GetDominantEmotion();
            var valence = currentState.GetValence();
            var arousal = currentState.GetArousal();

            var moodDescriptor = valence > 0.3 ? "Positive" :
                                 valence < -0.3 ? "Negative" : "Neutral";

            var energyDescriptor = arousal > 0.6 ? "High Energy" :
                                  arousal < 0.3 ? "Low Energy" : "Moderate Energy";

            return $"{moodDescriptor}, {energyDescriptor} (Dominant: {dominant})";
        }
    }
}
