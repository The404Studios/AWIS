#!/usr/bin/env python3
"""
Script to split the monolithic Program.cs into organized files
"""

import re
import os

def extract_region(content, region_name, start_line, end_line=None):
    """Extract a region from the content"""
    lines = content.split('\n')

    # Find the region
    in_region = False
    region_content = []
    brace_count = 0

    for i, line in enumerate(lines[start_line-1:], start=start_line):
        if f'#region {region_name}' in line:
            in_region = True
            region_content.append(line)
            continue

        if in_region:
            region_content.append(line)

            # Count braces to know when region ends
            brace_count += line.count('{') - line.count('}')

            if '#endregion' in line and brace_count <= 0:
                break

            if end_line and i >= end_line:
                break

    return '\n'.join(region_content)

def create_file(filename, namespace, usings, content):
    """Create a C# file with proper structure"""
    file_content = f"""using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
{usings}

namespace AWIS{namespace}
{{
{content}
}}
"""
    return file_content

# Define file mappings
file_mappings = [
    # (output_file, namespace, start_region, additional_usings)
    ("Utilities/Optimizers.cs", ".Utilities", "Advanced Optimization Algorithms", "using System.Threading.Tasks;"),
    ("Utilities/LearningRateSchedulers.cs", ".Utilities", "Learning Rate Schedulers", ""),
    ("Utilities/Regularization.cs", ".Utilities", "Advanced Regularization Techniques", ""),
    ("Utilities/TrainingInfrastructure.cs", ".Utilities", "Utility Methods and Training Infrastructure", "using System.IO;\\nusing System.Diagnostics;"),
    ("NeuralNetworks/Normalization.cs", ".NeuralNetworks", "Normalization Layers", ""),
    ("NeuralNetworks/RNN.cs", ".NeuralNetworks", "Recurrent Neural Networks", ""),
    ("NeuralNetworks/CNN.cs", ".NeuralNetworks", "Convolutional Neural Networks", ""),
    ("NeuralNetworks/Transformers.cs", ".NeuralNetworks", "Transformer Architecture", ""),
    ("NeuralNetworks/GraphNeuralNetworks.cs", ".NeuralNetworks", "Graph Neural Networks", ""),
    ("GenerativeModels/VAE.cs", ".GenerativeModels", "Variational Autoencoder", ""),
    ("GenerativeModels/GAN.cs", ".GenerativeModels", "Generative Adversarial Network", ""),
    ("GenerativeModels/DiffusionModels.cs", ".GenerativeModels", "Diffusion Models", ""),
    ("NLP/SequenceToSequence.cs", ".NLP", "Sequence-to-Sequence Models", ""),
    ("NLP/Embeddings.cs", ".NLP", "Advanced NLP Features", ""),
    ("ComputerVision/ObjectDetection.cs", ".ComputerVision", "Object Detection", ""),
    ("ComputerVision/ImageProcessing.cs", ".ComputerVision", "Advanced Image Processing", ""),
    ("MachineLearning/DecisionTrees.cs", ".MachineLearning", "Decision Trees and Random Forests", ""),
    ("MachineLearning/SVM.cs", ".MachineLearning", "Support Vector Machines", ""),
    ("MachineLearning/Clustering.cs", ".MachineLearning", "Clustering Algorithms", ""),
    ("MachineLearning/DimensionalityReduction.cs", ".MachineLearning", "Dimensionality Reduction", ""),
    ("MachineLearning/Evaluation.cs", ".MachineLearning", "Evaluation Metrics and Cross-Validation", ""),
    ("MachineLearning/TimeSeries.cs", ".MachineLearning", "Time Series Analysis", ""),
    ("MachineLearning/Recommendation.cs", ".MachineLearning", "Recommendation Systems", ""),
    ("Probabilistic/BayesianMethods.cs", ".Probabilistic", "Bayesian Methods", ""),
    ("Probabilistic/ProbabilisticProgramming.cs", ".Probabilistic", "Probabilistic Programming", ""),
    ("Probabilistic/Interpretability.cs", ".Probabilistic", "Interpretability and Explanation", ""),
    ("Audio/AudioProcessing.cs", ".Audio", "Audio Processing", ""),
    ("Graph/GraphAlgorithms.cs", ".Graph", "Graph Algorithms", ""),
    ("ReinforcementLearning/AdvancedRL.cs", ".ReinforcementLearning", "Advanced Reinforcement Learning", ""),
]

print("File mapping created. This script would extract regions from Program.cs")
print(f"Total files to create: {len(file_mappings)}")

for output_file, namespace, region, usings in file_mappings:
    print(f"  - {output_file} (from region: {region})")
