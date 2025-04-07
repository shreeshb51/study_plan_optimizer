# Study Plan Optimizer

## Project Description
Study Plan Optimizer is a web application that helps students optimize their study time allocation for maximum value as a 0-1 Knapsack Problem. By utilizing different optimizing algorithms: **Dynamic Programming, Branch and Bound, Greedy Algorithm**, the application recommends the most efficient study plan based on the available time and the relative value of different study topics.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Methodology](#methodology)
- [Examples](#examples)
- [References](#references)
- [Dependencies](#dependencies)
- [Algorithms/Mathematical Concepts Used](#algorithmsmathematical-concepts-used)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Data Source](#data-source)

## Installation

1. Install the required dependencies:
```bash
pip install streamlit pandas numpy plotly
```

2. Run the application:
```bash
streamlit run study_plan_optimizer.py
```

## Usage

1. Launch the application using the command above
2. In the sidebar:
   - Adjust the "Available Study Time" slider to set your total study hours
   - Select an optimization algorithm: Dynamic Programming, Branch and Bound, or Greedy Approach
   - Choose to upload your own CSV file or use the provided example data
3. If uploading your own data, ensure your CSV has the following columns:
   - `Module`: Name of the study topic
   - `Cost`: Time required (in hours)
   - `Value`: Importance/value of the topic
4. Click "Optimize My Study Plan" to run the analysis
5. Review the optimized study plan, metrics, and visualizations

Example CSV format:
```csv
Module,Cost,Value
Linear Algebra,2.5,8.3
Calculus,3.0,9.1
Probability,1.5,5.7
```

## Features

- **Multiple Optimization Algorithms**: Choose between Dynamic Programming, Branch and Bound, or Greedy approaches
- **Interactive UI**: Adjust study time and instantly see updated recommendations
- **Data Visualization**: View your optimized study plan with interactive charts
- **Performance Metrics**: See total value gained, time used, number of topics selected, and algorithm execution time
- **Flexible Data Input**: Upload your own data or use example data
- **Precise Decimal Arithmetic**: Uses Python's Decimal for precise calculations

## Methodology

The Study Plan Optimizer follows a systematic process flow:

1. **Data Input**: The application accepts study topics through CSV upload or example data
2. **Topic Validation**: Each study topic is validated to ensure positive time and value
3. **Problem Formulation**: The study planning problem is formulated as a 0-1 Knapsack Problem:
   - Items (study topics) each have a weight (time required) and value (importance)
   - The goal is to maximize total value while staying within the weight constraint (available study time)
4. **Algorithm Selection & Execution**:
   - **Dynamic Programming**: Creates a 2D table to build up the optimal solution bottom-up
   - **Branch and Bound**: Uses a priority queue and upper bounds to efficiently search the solution space
   - **Greedy Approach**: Sorts topics by value density (value/time) and selects in descending order
5. **Result Presentation**: Results are displayed through interactive visualizations and formatted lists

## Examples

When running with the example dataset and 6 hours of available study time, the optimizer produces a study plan that includes:

- Probability (Time: 1.5h, Value: 5.7)
- AI Ethics and Society (Time: 2.5h, Value: 7.5)
- Statistics (Time: 2.0h, Value: 6.8)

Total Value: 20.0
Time Used: 6.0 hours
Topics Selected: 3

*Note: Actual results may vary depending on the algorithm selected.*

## References

1. Martello, S., & Toth, P. (1990). *Knapsack Problems: Algorithms and Computer Implementations*. John Wiley & Sons.
2. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.
3. Kellerer, H., Pferschy, U., & Pisinger, D. (2004). *Knapsack Problems*. Springer.

## Dependencies

- **streamlit**: Web application framework
- **pandas**: Data manipulation library
- **numpy**: Numerical computing library
- **plotly.express**: Interactive visualization
- **Python Standard Library**: decimal, dataclasses, enum, heapq, time, csv, io, functools

## Algorithms/Mathematical Concepts Used

### 0-1 Knapsack Problem
The core optimization problem is formulated as a 0-1 Knapsack Problem, a combinatorial optimization problem where:
- Each item (study topic) has a weight (time required) and value (importance)
- Items can either be included (1) or excluded (0)
- The goal is to maximize total value while staying within weight capacity (available time)

### Implementation Methods

1. **Dynamic Programming Solution**:
   - Time Complexity: O(n * W) where n is the number of topics and W is the time capacity
   - Space Complexity: O(n * W)
   - Uses a tabulation approach with a 2D array

2. **Branch and Bound Solution**:
   - Uses a priority queue to explore the most promising branches first
   - Implements upper bound estimation based on value density
   - Prunes branches that cannot lead to better solutions than the current best

3. **Greedy Approach**:
   - Time Complexity: O(n log n) for sorting
   - Approximation algorithm that may not always find the optimal solution
   - Selects topics in descending order of value density (value/time)

### Special Considerations
- Scaled time units for more precise dynamic programming solution
- Decimal arithmetic to avoid floating-point precision issues

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all the researchers who developed and refined Knapsack algorithms
- Special thanks to the Streamlit team for creating such a powerful framework for data applications

## Data Source

The example data provided in the application is fictional and created for demonstration purposes. For real-world applications, you should create your own CSV file with your actual study topics, their required times, and assigned values. **Ensure that** the CSV file you use has the following columns: 
  - `Module`: Name of the study topic
  - `Cost`: Time required (in hours)
  - `Value`: Importance/value of the topic
