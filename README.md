# Luna Quantum SDK: Traveling Salesman Problem Demo

Welcome to the **Luna Quantum SDK** demonstration! This project showcases Luna Quantum's powerful optimization capabilities through a comprehensive Traveling Salesman Problem (TSP) implementation.

## 🚀 What This Demo Shows

This project demonstrates Luna Quantum's key features:

- **Unified Programming Model**: Write once, run on classical CPUs, quantum simulators, or real quantum hardware
- **Smart Abstractions**: Focus on problem modeling while Luna handles algorithm-specific transformations
- **Plug-and-Play Architecture**: Seamlessly switch between algorithms and backends without code changes
- **Production-Ready**: Enterprise-grade quantum cloud platform with secure token management

## 📍 The Problem

We solve the TSP for 4 German cities: **Berlin**, **Hamburg**, **Munich**, and **Cologne**. The goal is to find the shortest route that visits all cities exactly once and returns to the starting point.

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd /path/to/LunaDemo
   ```

2. **Install dependencies using uv**:
   ```bash
   uv sync
   ```

   This will install all required packages from `pyproject.toml`:
   - `folium>=0.19.7` - Interactive maps
   - `ipywidgets>=8.1.7` - Jupyter widgets
   - `luna-quantum>=1.0.0` - Luna Quantum SDK
   - `matplotlib>=3.10.3` - Plotting
   - `networkx>=3.5` - Graph processing
   - `notebook>=7.4.3` - Jupyter notebook
   - `pandas>=2.3.0` - Data manipulation

### Environment Variables

Set up your quantum computing tokens:

1. **D-Wave Token** (for quantum annealing):
   ```bash
   export DWAVE_TOKEN="your_dwave_token_here"
   ```

2. **Luna Quantum API Key** (if required):
   ```bash
   export LUNA_API_KEY="your_luna_api_key_here"
   ```


## 🚀 Running the Demo

### Jupyter Notebook (Recommended)

1. **Start Jupyter**:
   ```bash
   uv run jupyter notebook
   ```

2. **Open the notebooks**:
   - **Main demo**: Navigate to `notebooks/TSP_with_LunaSolve.ipynb` for the full demonstration
   - **Lite version**: Open `notebooks/TSP_with_LunaSolve_lite.ipynb` for a streamlined version
   - Run cells sequentially to see the demonstrations


## 📁 Project Structure

```
LunaDemo/
├── README.md                           # This file
├── pyproject.toml                      # Project dependencies
├── uv.lock                             # Locked dependencies
├── notebooks/                          # Jupyter notebooks
│   ├── TSP_with_LunaSolve.ipynb        # Main demonstration notebook
│   └── TSP_with_LunaSolve_lite.ipynb   # Streamlined demonstration
├── plots/                              # Directory for managing example pictures
└── utils/                              # Utility functions
```

## 🔧 Key Features Demonstrated

### 1. Predefined Use Cases
- Rapid prototyping with `TravellingSalesmanProblem`
- Automatic model generation from graph data
- Zero-setup optimization patterns

### 2. Custom Modeling with AqModels
- Position-based TSP formulation
- Advanced constraint handling
- Algorithm-agnostic model development

### 3. Multi-Algorithm Execution
- Classical algorithms (Simulated Annealing, Tabu Search)
- Quantum algorithms (Quantum Annealing)
- Hybrid approaches
- 

## 🎯 Expected Results

The demo will show you:

1. **Problem Setup**: Interactive map showing the 4 German cities
2. **Model Creation**: Two different modeling approaches (predefined vs custom)
3. **Algorithm Comparison**: Results from classical and quantum algorithms
4. **Solution Visualization**: Optimal tour plotted on an interactive map

## 🔍 Troubleshooting

### Common Issues

1. **Missing tokens**: Ensure environment variables are set correctly
2. **Package conflicts**: Use `uv sync --force` to resolve dependencies
3. **Jupyter not starting**: Try `uv run jupyter lab` instead of `jupyter notebook`
4. **D-Wave connection issues**: Verify your token has sufficient credits and access


## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all environment variables are set
3. Ensure you have sufficient credits on quantum platforms
4. Review the notebook outputs for error messages
5. Reach out to the team at [support@aqarios.com](mailto:support@aqarios.com)

## 📚 Next Steps

After running this demo:

1. **Experiment** with different algorithms and parameters
2. **Modify** the problem by adding more cities or constraints
3. **Explore** other Luna Quantum use cases
4. **Scale** to larger, real-world optimization problems

## 🔗 Additional Resources

- [Luna Quantum Documentation](https://docs.aqarios.com)

---

**Ready to explore quantum-enhanced optimization? Start with the main notebook and experience the future of problem-solving!**