# Traveling AI Agent

A modular, extensible AI agent framework designed for autonomous exploration, data collection, and learning in virtual environments.

## Overview

The Traveling AI Agent is a sophisticated software framework that simulates an autonomous agent capable of navigating virtual environments, collecting and processing data, learning from experiences, and making intelligent decisions. Built with a modular component-based architecture, it provides a flexible foundation for developing AI agents for various applications including virtual exploration, data collection, and autonomous decision-making.

## Features

- **Modular Component Architecture**: Easily extensible with plug-and-play components
- **Event-Based Communication**: Components communicate through a centralized event bus
- **Navigation and Path Planning**: A* algorithm for efficient path planning and exploration
- **Knowledge Management**: Persistent storage of locations, facts, and experiences
- **Data Collection**: Multi-sensor perception system with configurable data collection
- **Machine Learning**: Pattern recognition and predictive capabilities
- **Decision Making**: Goal-oriented planning and task execution
- **Visualization**: Rich visualizations of agent status, map, and collected data
- **Resource Management**: Monitoring and management of system resources

## Components

The agent consists of the following core components:

1. **Agent Core**: Central management of component lifecycle and event communication
2. **Resource Manager**: Monitors system resources like CPU, memory, and energy
3. **Knowledge Base**: Stores and retrieves information about locations, routes, and facts
4. **Decision Engine**: Manages goals and tasks, making decisions based on priorities
5. **User Interface**: Handles user commands and provides status information
6. **Navigation System**: Manages location tracking, path planning, and exploration
7. **Perception Module**: Collects data from various sensors in the environment
8. **Learning Module**: Analyzes collected data and adapts agent behavior
9. **Visualization Module**: Creates visual representations of agent state and data

## Installation

### Prerequisites

- Python 3.8 or higher
- Required Python packages:
  - numpy
  - matplotlib
  - pillow
  - sqlite3 (usually included with Python)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/traveling-ai-agent.git
   cd traveling-ai-agent
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

To create and run a basic agent:

```python
import asyncio
from agent_core import TravelingAgent
from resource_manager import ResourceManager
from knowledge_base import KnowledgeBase
from decision_engine import DecisionEngine
from user_interface import UserInterface
from navigation_system import NavigationSystem
from perception_module import PerceptionModule
from learning_module import LearningModule
from visualization_module import VisualizationModule

async def main():
    # Create the agent
    agent = TravelingAgent("MyAgent")
    
    # Create and register components
    resource_manager = ResourceManager()
    kb = KnowledgeBase("knowledge.db")
    decision_engine = DecisionEngine()
    ui = UserInterface()
    nav_system = NavigationSystem()
    perception = PerceptionModule()
    learning = LearningModule()
    visualization = VisualizationModule()
    
    # Register all components
    agent.register_component(resource_manager)
    agent.register_component(kb)
    agent.register_component(decision_engine)
    agent.register_component(ui)
    agent.register_component(nav_system)
    agent.register_component(perception)
    agent.register_component(learning)
    agent.register_component(visualization)
    
    # Initialize the agent
    await agent.initialize()
    
    # Add locations to the knowledge base
    home = kb.add_location(
        name="Home Base",
        latitude=40.7128,
        longitude=-74.0060,
        description="Starting location"
    )
    
    target = kb.add_location(
        name="Target Location",
        latitude=40.7484,
        longitude=-73.9857,
        description="Target destination"
    )
    
    # Import map from knowledge base
    nav_system.import_map_from_knowledge_base()
    
    # Set current location
    nav_system.set_current_location(home)
    
    # Start data collection
    perception.start_data_collection(interval=1.0)
    
    # Start learning
    learning.start_learning(interval=5.0)
    
    # Add exploration goal
    goal_id = "explore_target"
    decision_engine.add_goal(
        goal_id=goal_id,
        description=f"Explore Target Location",
        priority=GoalPriority.MEDIUM
    )
    
    # Add tasks for the goal
    decision_engine.add_task(
        task_id=f"{goal_id}_task1",
        goal_id=goal_id,
        description=f"Navigate to Target Location",
        metadata={"strategy": "explore", "target_id": target}
    )
    
    # Run the agent
    await agent.run()
    
    # Shutdown when done
    await agent.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

### Advanced Usage

#### Creating Custom Components

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

#### Working with the Knowledge Base

Adding and retrieving information:

```python
# Add a location
location_id = kb.add_location(
    name="New York",
    latitude=40.7128,
    longitude=-74.0060,
    description="The Big Apple"
)

# Add a point of interest
poi_id = kb.add_point_of_interest(
    name="Empire State Building",
    location_id=location_id,
    poi_type="landmark",
    description="Famous skyscraper",
    importance=8
)

# Add a fact
kb.add_fact(
    subject="Empire State Building",
    predicate="has_height",
    obj="381 meters",
    confidence=0.99,
    source="Wikipedia"
)

# Get facts about a subject
facts = kb.get_facts(subject="Empire State Building")
```

#### Navigation and Exploration

```python
# Navigate to a location
await nav_system.navigate_to(location_id)

# Explore an area
await nav_system.explore_area(location_id, radius=1.0)

# Find a path between locations
path = nav_system.find_path(start_location_id, end_location_id)
```

#### Data Collection and Learning

```python
# Start data collection
perception.start_data_collection(interval=1.0)

# Process collected data
results = await perception.process_data()

# Export collected data
export_file = perception.export_data()

# Start learning
learning.start_learning(interval=5.0)

# Train models manually
results = await learning.train_models()

# Get predictions
env_prediction = learning.predict_environment(location_id)
recommended_location = learning.recommend_location()
```

#### Visualization

```python
# Create visualizations
agent_status_path = await visualization.create_agent_status_visualization()
map_path = await visualization.create_map_visualization()
resource_path = await visualization.create_resource_usage_visualization()
env_data_path = await visualization.create_environmental_data_visualization()

# Create a comprehensive dashboard
dashboard_path = await visualization.create_dashboard()
```

## Testing

The project includes both unit tests and integration tests:

```bash
# Run unit tests
python test_unit.py

# Run integration tests
python test_integration.py
```

## Architecture

### Component Interaction

Components interact through the event bus, which allows for loose coupling and easy extensibility:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Component A    │     │    Event Bus    │     │  Component B    │
│                 │     │                 │     │                 │
│  publish event  │────▶│  route event    │────▶│  handle event   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Data Flow

Data flows through the system as follows:

1. Perception Module collects data from sensors
2. Data is processed and stored in the Knowledge Base
3. Learning Module analyzes data to identify patterns
4. Decision Engine uses knowledge and patterns to make decisions
5. Navigation System executes movement based on decisions
6. Visualization Module creates visual representations of the agent's state and data

## Extending the Agent

The agent can be extended in several ways:

1. **Add new components**: Create new components that implement the Component interface
2. **Add new sensors**: Extend the Perception Module with new sensor types
3. **Improve learning algorithms**: Enhance the Learning Module with more sophisticated algorithms
4. **Add new visualization types**: Extend the Visualization Module with new visualization capabilities
5. **Implement new decision strategies**: Add new strategies to the Decision Engine

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The A* algorithm implementation is based on [A* Search Algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)
- Visualization components use [Matplotlib](https://matplotlib.org/) and [Pillow](https://python-pillow.org/)
