"""
Application of Optimizing Algorithms: Dynamic Programming, Branch and Bound, Greedy.
Solves 0-1 Knapsack problem to optimize study time allocation for maximum value.
"""

# Import libraries
import streamlit as st  # Web app framework for creating UI
import pandas as pd  # Data manipulation library
import numpy as np  # Numerical computing library
import heapq  # Priority queue implementation
import time  # Time measurement
from dataclasses import dataclass  # For creating data classes
from typing import List, Tuple, Dict, Set, Optional  # Type hints
from decimal import Decimal, getcontext  # Precise decimal arithmetic
import csv  # CSV file handling
from io import StringIO  # In-memory file-like object
import functools  # Higher-order functions
import plotly.express as px  # Interactive visualization
from enum import Enum, auto  # Enumerations

# ================== CORE DATA STRUCTURES ==================
@dataclass(frozen=True)
class StudyTopic:
    """Immutable study topic with validation"""
    name: str  # Name of the study topic
    time: Decimal  # Time required to study this topic
    value: Decimal  # Value/importance of studying this topic

    def __post_init__(self):
        """Validation after initialization"""
        if self.time <= 0 or self.value <= 0:
            raise ValueError("Time and value must be positive")

    @property
    def value_density(self) -> Decimal:
        """Calculate value per unit time"""
        return self.value / self.time


class AlgorithmType(Enum):
    """Enumeration of available optimization algorithms"""
    DYNAMIC_PROGRAMMING = auto()  # Dynamic programming approach
    BRANCH_AND_BOUND = auto()  # Branch and bound approach
    GREEDY = auto()  # Greedy heuristic approach


# ================== ALGORITHM IMPLEMENTATIONS ==================
class StudyPlanner:
    """Main class that implements study planning algorithms"""
    def __init__(self, max_study_time: Decimal = Decimal('6.0')):
        """Initialize with maximum available study time"""
        getcontext().prec = 6  # Set decimal precision
        self.max_time = max_study_time  # Maximum study time available
        self._topics: List[StudyTopic] = []  # List to store study topics

    @property
    def topics(self) -> List[StudyTopic]:
        """Getter for topics that returns a copy"""
        return self._topics.copy()

    def load_from_csv(self, file_contents: str) -> None:
        """Streaming CSV loader with schema validation"""
        try:
            reader = csv.DictReader(StringIO(file_contents))  # Create CSV reader
            required = {'Module', 'Cost', 'Value'}  # Required columns

            # Validate CSV has required columns
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError(f"CSV must contain these columns: {required}")

            self._topics = []  # Reset topics list
            for i, row in enumerate(reader, 1):  # Process each row
                try:
                    self._topics.append(
                        StudyTopic(
                            name=str(row['Module']),  # Get module name
                            time=Decimal(str(row['Cost'])),  # Get time cost
                            value=Decimal(str(row['Value']))  # Get value
                        ))
                except (ValueError, KeyError) as e:
                    st.warning(f"Row {i} skipped: {str(e)}")  # Show warning for invalid rows

            if not self._topics:  # Check if any valid topics were loaded
                raise ValueError("No valid topics found in file")

        except Exception as e:
            raise ValueError(f"Error processing CSV: {str(e)}")

    def solve_dynamic_programming(self) -> Tuple[List[StudyTopic], Decimal, Decimal]:
        """Solves the 0-1 knapsack problem using dynamic programming.
        Returns: (selected_topics, total_value, time_used)
        """
        n = len(self._topics)
        if n == 0:  # Handle empty topics case
            return [], Decimal('0'), Decimal('0')

        # Convert to granular time units for DP table
        scale_factor = Decimal('10')
        scaled_max_time = int(self.max_time * scale_factor)

        # Initialize DP table with zeros
        dp = [[Decimal('0') for _ in range(scaled_max_time + 1)] for _ in range(n + 1)]

        # Fill DP table
        for i in range(1, n + 1):  # For each topic
            topic = self._topics[i - 1]
            scaled_time = int(topic.time * scale_factor)  # Scale time for DP

            for j in range(scaled_max_time + 1):  # For each possible time capacity
                if scaled_time <= j:  # If current topic fits in remaining capacity
                    dp[i][j] = max(
                        dp[i-1][j],  # Value without including current topic
                        dp[i-1][j-scaled_time] + topic.value  # Value with current topic
                    )
                else:
                    dp[i][j] = dp[i-1][j]  # Carry forward previous value

        # Backtrack to find selected topics
        selected_topics = []
        time_used = Decimal('0')
        total_value = dp[n][scaled_max_time]  # Optimal value from DP table

        j = scaled_max_time  # Start from maximum capacity
        for i in range(n, 0, -1):  # Backtrack through topics
            if dp[i][j] != dp[i-1][j]:  # If topic was included
                topic = self._topics[i-1]
                selected_topics.append(topic)
                scaled_time = int(topic.time * scale_factor)
                time_used += topic.time
                j -= scaled_time  # Reduce remaining capacity

        return selected_topics, total_value, time_used

    def solve_branch_and_bound(self) -> Tuple[List[StudyTopic], Decimal, Decimal]:
        """Solves the 0-1 knapsack problem using branch and bound.
        Returns: (selected_topics, total_value, time_used)
        """
        n = len(self._topics)
        if n == 0:  # Handle empty topics case
            return [], Decimal('0'), Decimal('0')

        # Initialize as Decimal
        best_value = Decimal('0')  # Track best solution value found
        best_solution = []  # Track best solution indices

        # Sort topics by value density (value/time) for better bounds
        sorted_topics = sorted(
            [(i, topic) for i, topic in enumerate(self._topics)],
            key=lambda x: x[1].value_density,
            reverse=True
        )

        # Node for branch and bound tree
        @dataclass
        class Node:
            level: int  # Level in the tree
            value: Decimal  # Value so far
            time: Decimal  # Time used so far
            bound: Decimal  # Upper bound
            included: List[int]  # Indices of included topics

        def compute_bound(node: Node) -> Decimal:
            """Compute upper bound for node."""
            if node.time >= self.max_time:  # If no capacity left
                return Decimal('0')

            bound = node.value  # Start with current value
            j = node.level
            remaining_time = self.max_time - node.time  # Calculate remaining time

            # Add items by value density until knapsack is full
            while j < n and sorted_topics[j][1].time <= remaining_time:
                idx, topic = sorted_topics[j]
                bound += topic.value
                remaining_time -= topic.time
                j += 1

            # Add fractional part of next item if possible
            if j < n:
                idx, topic = sorted_topics[j]
                bound += topic.value_density * remaining_time

            return bound

        # Initialize priority queue with root node
        # Use negative bound for max-heap behavior with heapq
        queue = []
        root = Node(
            level=0,
            value=Decimal('0'),
            time=Decimal('0'),
            bound=Decimal('0'),
            included=[]
        )
        root.bound = compute_bound(root)  # Compute initial bound
        heapq.heappush(queue, (-root.bound, id(root), root))  # Push to priority queue

        while queue:  # While there are nodes to explore
            _, _, node = heapq.heappop(queue)  # Get node with best bound

            # Skip if bound is worse than current best solution
            if node.bound < best_value:
                continue

            # Check if we've reached the end of the tree
            if node.level == n:
                if node.value > best_value:  # Update best solution if better
                    best_value = node.value
                    best_solution = node.included.copy()
                continue

            idx, topic = sorted_topics[node.level]  # Get current topic

            # Include current topic
            if node.time + topic.time <= self.max_time:  # If it fits
                include_node = Node(
                    level=node.level + 1,
                    value=node.value + topic.value,
                    time=node.time + topic.time,
                    bound=Decimal('0'),
                    included=node.included + [idx]
                )
                include_node.bound = compute_bound(include_node)  # Compute new bound

                # Update best solution if we found a better one
                if include_node.value > best_value and include_node.level == n:
                    best_value = include_node.value
                    best_solution = include_node.included.copy()

                # Only add to queue if bound is promising
                if include_node.bound > best_value:
                    heapq.heappush(queue, (-include_node.bound, id(include_node), include_node))

            # Exclude current topic
            exclude_node = Node(
                level=node.level + 1,
                value=node.value,
                time=node.time,
                bound=Decimal('0'),
                included=node.included.copy()
            )
            exclude_node.bound = compute_bound(exclude_node)  # Compute bound

            if exclude_node.bound > best_value:  # Only add if promising
                heapq.heappush(queue, (-exclude_node.bound, id(exclude_node), exclude_node))

        # Reconstruct solution
        selected_topics = [self._topics[i] for i in best_solution]  # Get topic objects
        time_used = sum(topic.time for topic in selected_topics)  # Calculate total time

        return selected_topics, best_value, time_used

    def solve_greedy(self) -> Tuple[List[StudyTopic], Decimal, Decimal]:
        """Solves the 0-1 knapsack problem using greedy approach.
        Returns: (selected_topics, total_value, time_used)
        """
        if not self._topics:  # Handle empty topics case
            return [], Decimal('0'), Decimal('0')

        # Sort topics by value density
        sorted_topics = sorted(
            self._topics,
            key=lambda topic: topic.value_density,
            reverse=True
        )

        selected_topics = []
        total_value = Decimal('0')
        time_used = Decimal('0')

        # Greedily select topics
        for topic in sorted_topics:
            if time_used + topic.time <= self.max_time:  # If topic fits
                selected_topics.append(topic)
                total_value += topic.value
                time_used += topic.time

        return selected_topics, total_value, time_used

    def solve(self, algorithm_type: AlgorithmType) -> Dict:
        """Solve the study planning problem using the specified algorithm.
        Returns a results dictionary with selected_topics, total_value, and time_used.
        """
        start_time = time.time()  # Start timer

        # Select algorithm based on input
        if algorithm_type == AlgorithmType.DYNAMIC_PROGRAMMING:
            selected_topics, total_value, time_used = self.solve_dynamic_programming()
        elif algorithm_type == AlgorithmType.BRANCH_AND_BOUND:
            selected_topics, total_value, time_used = self.solve_branch_and_bound()
        elif algorithm_type == AlgorithmType.GREEDY:
            selected_topics, total_value, time_used = self.solve_greedy()
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")

        execution_time = time.time() - start_time  # Calculate execution time

        return {
            "selected_topics": selected_topics,
            "total_value": total_value,
            "time_used": time_used,
            "execution_time": execution_time
        }


# ================== STREAMLIT UI ==================
def main():
    """Main function to run the Streamlit application"""
    # Configure page
    st.set_page_config(
        page_title="Study Plan Optimizer",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better visuals
    st.markdown("""
    <style>
        .metric-card {
            background: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }
        .plot-container {
            background: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.title("⚙️ Settings")

        # Study time input
        study_time = st.slider(
            "Available Study Time (hours)",
            1.0, 12.0, 6.0, 0.5,
            help="Total time you can dedicate to studying"
        )

        # Algorithm selection
        algorithm_map = {
            "Dynamic Programming (Recommended)": AlgorithmType.DYNAMIC_PROGRAMMING,
            "Branch and Bound": AlgorithmType.BRANCH_AND_BOUND,
            "Greedy Approach": AlgorithmType.GREEDY
        }

        algorithm_option = st.radio(
            "Optimization Method",
            list(algorithm_map.keys()),
            index=0,
            help="Dynamic Programming: Best balance | Branch and Bound: Most accurate | Greedy: Fastest"
        )

        selected_algorithm = algorithm_map[algorithm_option]  # Get selected algorithm

        # Data input options
        st.subheader("📁 Study Topics Data")
        data_option = st.radio(
            "Data Source",
            ["Upload CSV", "Use Example Data"],
            index=0
        )

        uploaded_file = None
        if data_option == "Upload CSV":  # Handle file upload
            uploaded_file = st.file_uploader(
                "Choose CSV file",
                type=["csv"],
                help="Required columns: Module, Cost, Value"
            )
        else:  # Use example data
            example_csv = """Module,Cost,Value
Linear Algebra,2.5,8.3
Calculus,3.0,9.1
Probability,1.5,5.7
Algorithms,4.0,10.0
Statistics,2.0,6.8
Natural Language Processing,6.0,9.4
Computer Vision,5.5,9.2
AI Ethics and Society,2.5,7.5
Blockchain Fundamentals,4.0,8.5
Robotics and Automation,5.0,8.9
Edge Computing Concepts,3.5,8.0
Digital Signal Processing,4.5,8.7
Augmented Reality Basics,4.0,8.2
Game Development Essentials,4.5,8.6
3D Modeling and Animation,4.0,8.0"""
            uploaded_file = StringIO(example_csv)
            st.info("Using example dataset. Switch to 'Upload CSV' for your own data.")

    # Main content area
    st.title("📚 Study Plan Optimizer")
    st.caption("Get the most value from your limited study time")

    # Only proceed if we have data
    if uploaded_file is not None:
        try:
            # Initialize planner with user's study time
            planner = StudyPlanner(Decimal(str(study_time)))

            # Load data
            with st.spinner("📊 Loading your data..."):
                file_content = (
                    uploaded_file.getvalue()  # Get content from StringIO
                    if isinstance(uploaded_file, StringIO)
                    else uploaded_file.getvalue().decode("utf-8")  # Decode uploaded file
                )
                planner.load_from_csv(file_content)  # Load data into planner

                # Show data preview
                with st.expander("🔍 Preview your study topics"):
                    df = pd.DataFrame([{
                        'Topic': t.name,
                        'Time Needed (hrs)': float(t.time),
                        'Value': float(t.value),
                        'Value per Hour': float(t.value_density)
                    } for t in planner.topics])
                    st.dataframe(
                        df.sort_values('Value per Hour', ascending=False),
                        use_container_width=True
                    )

            # Run analysis when button is clicked
            if st.button("🚀 Optimize My Study Plan"):
                with st.spinner(f"🧠 Finding the best plan using {algorithm_option}..."):
                    results = planner.solve(selected_algorithm)  # Run optimization

                    # Display results
                    st.success("Optimization complete!")

                    # Metrics cards
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h3>Total Value</h3>
                                <h1>{float(results['total_value']):.1f}</h1>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col2:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h3>Time Used</h3>
                                <h1>{float(results['time_used']):.1f} hrs</h1>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col3:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h3>Topics Selected</h3>
                                <h1>{len(results['selected_topics'])}</h1>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    with col4:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <h3>Compute Time</h3>
                                <h1>{results['execution_time']:.3f} s</h1>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Topic list
                    st.subheader("📝 Recommended Study Plan")
                    for topic in results['selected_topics']:  # Display each selected topic
                        st.markdown(
                            f"""
                            <div style="padding: 10px; margin: 5px 0; background: #f0f2f6; border-radius: 5px;">
                                <b>{topic.name}</b><br>
                                ⏱️ {float(topic.time):.1f} hours | 💎 Value: {float(topic.value):.1f}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # Visualization
                    st.subheader("📈 Plan Visualization")
                    fig = px.bar(
                        pd.DataFrame([{
                            'Topic': t.name,
                            'Time (hours)': float(t.time),
                            'Value': float(t.value)
                        } for t in results['selected_topics']]),
                        x='Topic',
                        y=['Time (hours)', 'Value'],
                        barmode='group',
                        color_discrete_sequence=['#636EFA', '#EF553B'],
                        labels={'value': 'Score'},
                        height=400
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        yaxis_title="Score/Time",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)  # Display interactive chart

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")  # Show error message
            st.info("Please check your data format and try again.")
    else:
        st.info("👈 Please upload your study topics data or use the example dataset")


if __name__ == "__main__":
    main()  # Run the application
