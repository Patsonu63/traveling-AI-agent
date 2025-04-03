#!/usr/bin/env python3
"""
Traveling AI Agent - User Manual

This document provides a comprehensive guide to using the Traveling AI Agent.
"""

# Traveling AI Agent - User Manual

## Introduction

The Traveling AI Agent is a sophisticated software framework that simulates an autonomous agent capable of navigating virtual environments, collecting and processing data, learning from experiences, and making intelligent decisions. This user manual will guide you through the setup, configuration, and operation of the agent.

## Installation

### System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 1GB free disk space
- Operating System: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+ recommended)

### Installation Steps

1. Clone or download the Traveling AI Agent repository:
   ```
   git clone https://github.com/yourusername/traveling-ai-agent.git
   cd traveling-ai-agent
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Quick Start

The fastest way to get started is to run the example script:

```
python example.py
```

This will:
- Initialize the agent with all components
- Set up a sample environment with multiple locations
- Configure exploration goals and tasks
- Run the agent for a simulated exploration mission
- Generate visualizations and export collected data

## Components Overview

### Agent Core

The central component that manages the lifecycle of all other components and facilitates communication between them through an event bus.

### Resource Manager

Monitors system resources like CPU, memory, and simulated energy levels. Configure thresholds in the constructor:

```python
resource_manager = ResourceManager(
    energy_warning=30,  # Percentage
    energy_critical=10,  # Percentage
    cpu_warning=80,     # Percentage
    cpu_critical=95     # Percentage
)
```

### Knowledge Base

Stores and retrieves information about locations, routes, points of interest, and facts. The database file can be specified in the constructor:

```python
kb = KnowledgeBase("my_knowledge.db")
```

### Decision Engine

Manages goals and tasks, making decisions based on priorities. Goals and tasks can be added programmatically:

```python
# Add a goal
goal_id = "explore_area"
decision_engine.add_goal(
    goal_id=goal_id,
    description="Explore Target Area",
    priority=GoalPriority.HIGH
)

# Add a task
decision_engine.add_task(
    task_id="navigate_to_target",
    goal_id=goal_id,
    description="Navigate to Target Location",
    metadata={"strategy": "navigate", "target_id": location_id}
)
```

### Navigation System

Manages location tracking, path planning, and exploration. Key methods include:

```python
# Set current location
nav_system.set_current_location(location_id)

# Navigate to a destination
await nav_system.navigate_to(destination_id)

# Explore an area
await nav_system.explore_area(location_id, radius=1.0)
```

### Perception Module

Collects data from various sensors in the environment. Data collection can be controlled with:

```python
# Start data collection
perception.start_data_collection(interval=1.0)  # seconds

# Stop data collection
perception.stop_data_collection()

# Process collected data
results = await perception.process_data()

# Export data
export_file = perception.export_data()
```

### Learning Module

Analyzes collected data and adapts agent behavior. Learning can be controlled with:

```python
# Start learning
learning.start_learning(interval=5.0)  # seconds

# Stop learning
learning.stop_learning()

# Train models manually
results = await learning.train_models()

# Get predictions
env_prediction = learning.predict_environment(location_id)
recommended_location = learning.recommend_location()
```

### Visualization Module

Creates visual representations of agent state and data. Key methods include:

```python
# Create individual visualizations
agent_status_path = await visualization.create_agent_status_visualization()
map_path = await visualization.create_map_visualization()
resource_path = await visualization.create_resource_usage_visualization()
env_data_path = await visualization.create_environmental_data_visualization()

# Create a comprehensive dashboard
dashboard_path = await visualization.create_dashboard()
```

## Configuration

### Component Configuration

Most components can be configured through their constructor parameters. For example:

```python
# Configure the Perception Module
perception = PerceptionModule(
    data_dir="custom_data_directory",
    enable_camera=True,
    enable_environmental=True,
    enable_gps=True,
    enable_compass=True
)

# Configure the Visualization Module
visualization = VisualizationModule(
    output_dir="custom_visualizations_directory",
    map_size=(1024, 768),
    chart_size=(12, 8),
    dpi=150
)
```

### Environment Configuration

The agent operates in a virtual environment defined by locations and routes. You can configure this environment by adding locations and points of interest to the Knowledge Base:

```python
# Add a location
location_id = kb.add_location(
    name="Mountain Peak",
    latitude=40.7484,
    longitude=-73.9857,
    description="High elevation point with panoramic views"
)

# Add a point of interest
poi_id = kb.add_point_of_interest(
    name="Observation Tower",
    location_id=location_id,
    poi_type="landmark",
    description="Tall tower providing views of the surrounding area",
    importance=8
)

# Add a fact about the point of interest
kb.add_fact(
    subject="Observation Tower",
    predicate="height",
    obj="50 meters",
    confidence=0.95,
    source="geographical_survey"
)
```

## Running the Agent

To run the agent, you need to:

1. Create and register all components
2. Initialize the agent
3. Configure the environment
4. Set up goals and tasks
5. Start the agent's run loop

Here's a minimal example:

```python
import asyncio
from agent_core import TravelingAgent

async def main():
    # Create the agent
    agent = TravelingAgent("MyAgent")
    
    # Create and register components
    # ... (create and register all required components)
    
    # Initialize the agent
    await agent.initialize()
    
    # Configure the environment
    # ... (add locations, points of interest, etc.)
    
    # Set up goals and tasks
    # ... (add goals and tasks)
    
    # Run the agent
    await agent.run()
    
    # Shutdown when done
    await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## Extending the Agent

### Creating Custom Components

You can extend the agent's functionality by creating custom components:

```python
from agent_core import Component

class MyCustomComponent(Component):
    def __init__(self):
        super().__init__("MyCustomComponent")
        # Initialize your component
        
    async def initialize(self) -> bool:
        # Initialize resources, subscribe to events
        self.agent.event_bus.subscribe("data_collected", self.on_data_collected)
        self.active = True
        return True
        
    async def shutdown(self) -> bool:
        # Clean up resources, unsubscribe from events
        self.agent.event_bus.unsubscribe("data_collected", self.on_data_collected)
        self.active = False
        return True
        
    async def update(self) -> None:
        # Update component state
        # This is called regularly by the agent
        pass
        
    async def on_data_collected(self, data):
        # Handle data collected events
        self.logger.info(f"Received data: {data}")
```

### Adding New Sensors

You can add new sensor types to the Perception Module by extending the `SensorType` enum and implementing the corresponding sensor class:

```python
from perception_module import SensorType, Sensor

# Add a new sensor type
SensorType.RADIATION = "radiation"

# Implement the sensor
class RadiationSensor(Sensor):
    def __init__(self):
        super().__init__(SensorType.RADIATION)
        
    async def read(self):
        # Implement sensor reading logic
        return random.uniform(0.1, 10.0)  # Simulated radiation level
```

## Troubleshooting

### Common Issues

1. **Component initialization failed**
   - Check that all required components are registered
   - Ensure that component dependencies are properly configured

2. **Navigation errors**
   - Verify that locations exist in the Knowledge Base
   - Check that routes between locations are defined or can be automatically created

3. **Data collection issues**
   - Ensure that the Perception Module is properly initialized
   - Check that sensors are enabled and configured correctly

4. **Visualization errors**
   - Verify that required directories exist and are writable
   - Check that matplotlib and Pillow are properly installed

### Logging

The agent uses Python's logging module to log information, warnings, and errors. By default, logs are written to both the console and a log file. You can configure logging in your main script:

```python
import logging

logging.basicConfig(
    level=logging.INFO,  # Set to logging.DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent.log')
    ]
)
```

## Advanced Topics

### Event System

Components communicate through an event-based system. You can publish and subscribe to events:

```python
# Subscribe to an event
self.agent.event_bus.subscribe("data_collected", self.on_data_collected)

# Publish an event
await self.agent.event_bus.publish("custom_event", {"data": "value"})
```

### Custom Decision Strategies

You can implement custom decision strategies by extending the Decision Engine:

```python
class CustomDecisionEngine(DecisionEngine):
    async def _make_decisions(self):
        # Implement custom decision logic
        pass
```

### Performance Optimization

For better performance:
- Adjust the update interval of components
- Optimize data collection frequency
- Use efficient data structures for large datasets
- Consider using numpy for numerical operations

## Conclusion

The Traveling AI Agent provides a flexible framework for simulating autonomous agents in virtual environments. By understanding its components and configuration options, you can create sophisticated agents for various applications including virtual exploration, data collection, and autonomous decision-making.

For more information, refer to the README.md file and the example.py script included with the agent.
