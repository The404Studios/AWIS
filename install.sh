#!/bin/bash

echo "================================================================================"
echo "AWIS v8.0 - Installing Required NuGet Packages"
echo "================================================================================"
echo ""

echo "Installing Core ML Libraries..."
dotnet add package Microsoft.ML --version 3.0.1
dotnet add package Microsoft.ML.Vision --version 3.0.1
dotnet add package Microsoft.ML.ImageAnalytics --version 3.0.1
dotnet add package Microsoft.ML.TensorFlow --version 3.0.1

echo ""
echo "Installing Computer Vision Libraries..."
dotnet add package Emgu.CV --version 4.8.1.5350
dotnet add package Accord.Vision --version 3.8.0
dotnet add package Accord.Imaging --version 3.8.0

echo ""
echo "Installing Web Automation Libraries..."
dotnet add package Selenium.WebDriver --version 4.15.0
dotnet add package Selenium.Support --version 4.15.0

echo ""
echo "Installing Machine Learning Libraries..."
dotnet add package Accord.MachineLearning --version 3.8.0
dotnet add package Accord.Statistics --version 3.8.0
dotnet add package Accord.Math --version 3.8.0
dotnet add package Accord.Neuro --version 3.8.0

echo ""
echo "Installing Input Simulation Libraries..."
dotnet add package InputSimulator --version 1.0.4.0

echo ""
echo "Installing Logging and Utilities..."
dotnet add package Serilog --version 3.1.1
dotnet add package Serilog.Sinks.Console --version 5.0.1
dotnet add package Serilog.Sinks.File --version 5.0.0
dotnet add package Newtonsoft.Json --version 13.0.3

echo ""
echo "Installing Parallel Processing Libraries..."
dotnet add package System.Threading.Tasks.Dataflow --version 8.0.0
dotnet add package System.Collections.Concurrent --version 4.3.0

echo ""
echo "Installing Math and Scientific Computing..."
dotnet add package MathNet.Numerics --version 5.0.0
dotnet add package ILGPU --version 1.5.1
dotnet add package Rationals --version 1.3.1

echo ""
echo "Installing Data Processing Libraries..."
dotnet add package CsvHelper --version 30.0.1
dotnet add package EPPlus --version 7.0.3
dotnet add package Parquet.Net --version 4.12.0

echo ""
echo "Installing Deep Learning Frameworks..."
dotnet add package TensorFlow.NET --version 0.100.0
dotnet add package SciSharp.TensorFlow.Redist --version 2.11.0
dotnet add package Keras.NET --version 3.8.4.1

echo ""
echo "Installing NLP Libraries..."
dotnet add package OpenNLP --version 1.6.0
dotnet add package SharpNLP --version 1.0.2592

echo ""
echo "Installing Graph Processing..."
dotnet add package QuikGraph --version 2.5.0
dotnet add package QuikGraph.Graphviz --version 2.5.0

echo ""
echo "Installing Image Processing Extensions..."
dotnet add package SixLabors.ImageSharp --version 3.1.0
dotnet add package SixLabors.ImageSharp.Drawing --version 2.1.0

echo ""
echo "Installing Audio Processing..."
dotnet add package NAudio --version 2.2.1
dotnet add package Concentus --version 2.0.0
dotnet add package Concentus.Oggfile --version 1.0.4

echo ""
echo "Installing Compression Libraries..."
dotnet add package SharpCompress --version 0.35.0
dotnet add package K4os.Compression.LZ4 --version 1.3.6

echo ""
echo "Installing HTTP and API Libraries..."
dotnet add package RestSharp --version 110.2.0
dotnet add package Flurl.Http --version 4.0.0

echo ""
echo "Installing Database Connectors (Optional)..."
dotnet add package System.Data.SQLite --version 1.0.118
dotnet add package Dapper --version 2.1.24

echo ""
echo "Installing Async and Reactive Extensions..."
dotnet add package System.Reactive --version 6.0.0
dotnet add package Microsoft.Bcl.AsyncInterfaces --version 8.0.0

echo ""
echo "Installing Testing Libraries (Optional)..."
dotnet add package xunit --version 2.6.2
dotnet add package xunit.runner.visualstudio --version 2.5.4
dotnet add package NUnit --version 4.0.1

echo ""
echo "Installing Performance and Profiling..."
dotnet add package BenchmarkDotNet --version 0.13.11

echo ""
echo "Installing Dependency Injection..."
dotnet add package Microsoft.Extensions.DependencyInjection --version 8.0.0
dotnet add package Autofac --version 7.1.0

echo ""
echo "Installing Configuration Management..."
dotnet add package Microsoft.Extensions.Configuration --version 8.0.0
dotnet add package Microsoft.Extensions.Configuration.Json --version 8.0.0

echo ""
echo "================================================================================"
echo "Installation Complete!"
echo "================================================================================"
echo ""
echo "All required NuGet packages have been installed."
echo ""
echo "Next steps:"
echo "1. Build the project: dotnet build"
echo "2. Run the project: dotnet run"
echo ""
echo "For GPU acceleration (CUDA), you may need to install:"
echo "- NVIDIA CUDA Toolkit"
echo "- cuDNN library"
echo ""
