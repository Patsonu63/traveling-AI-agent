#!/usr/bin/env python3
"""
Traveling AI Agent - Integration Test

This script tests the integration of all components of the Traveling AI Agent.
"""

import asyncio
import logging
import os
import time
from datetime import datetime

from agent_core import TravelingAgent, AgentState
from resource_manager import ResourceManager
from knowledge_base import KnowledgeBase
from decision_engine import DecisionEngine, GoalPriority
from user_interface import UserInterface
from navigation_system import NavigationSystem
from perception_module import PerceptionModule
from learning_module import LearningModule
from visualization_module import VisualizationModule


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_integration.log')
    ]
)

logger = logging.getLogger("test_integration")


async def test_initialization():
    """Test the initialization of all components."""
    logger.info("Testing component initialization")
    
    # Create the agent
    agent = TravelingAgent("TestAgent")
    
    # Create all components
    resource_manager = ResourceManager()
    kb = KnowledgeBase("test_knowledge.db")
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
    success = await agent.initialize()
    assert success, "Agent initialization failed"
    
    # Check that all components are active
    for name, component in agent.components.items():
        assert component.active, f"Component {name} is not active after initialization"
        
    logger.info("All components initialized successfully")
    
    # Shutdown the agent
    await agent.shutdown()
    
    return True


async def test_knowledge_base():
    """Test the Knowledge Base component."""
    logger.info("Testing Knowledge Base")
    
    # Create the agent
    agent = TravelingAgent("KBTest")
    
    # Create and register the knowledge base
    kb = KnowledgeBase("test_knowledge.db")
    agent.register_component(kb)
    
    # Initialize the agent
    await agent.initialize()
    
    # Add a location
    location_id = kb.add_location(
        name="Test Location",
        latitude=40.7128,
        longitude=-74.0060,
        description="A test location"
    )
    assert location_id > 0, "Failed to add location"
    
    # Get the location
    location = kb.get_location(location_id)
    assert location is not None, "Failed to retrieve location"
    assert location["name"] == "Test Location", "Location name mismatch"
    
    # Add a point of interest
    poi_id = kb.add_point_of_interest(
        name="Test POI",
        location_id=location_id,
        poi_type="landmark",
        description="A test point of interest",
        importance=5
    )
    assert poi_id > 0, "Failed to add point of interest"
    
    # Get points of interest
    pois = kb.get_points_of_interest(location_id)
    assert len(pois) > 0, "Failed to retrieve points of interest"
    assert pois[0]["name"] == "Test POI", "Point of interest name mismatch"
    
    # Add a fact
    fact_id = kb.add_fact(
        subject="Test Location",
        predicate="has_feature",
        obj="Test POI",
        confidence=0.9,
        source="Test"
    )
    assert fact_id > 0, "Failed to add fact"
    
    # Get facts
    facts = kb.get_facts(subject="Test Location")
    assert len(facts) > 0, "Failed to retrieve facts"
    assert facts[0]["predicate"] == "has_feature", "Fact predicate mismatch"
    
    # Mark location as visited
    success = kb.mark_location_visited(location_id)
    assert success, "Failed to mark location as visited"
    
    # Check if location is marked as visited
    location = kb.get_location(location_id)
    assert location["visited"] == 1, "Location not marked as visited"
    
    # Test memory functions
    kb.remember("test_key", "test_value")
    value = kb.recall("test_key")
    assert value == "test_value", "Memory recall failed"
    
    kb.forget("test_key")
    value = kb.recall("test_key")
    assert value is None, "Memory forget failed"
    
    # Export knowledge
    export_data = kb.export_knowledge()
    assert export_data, "Knowledge export failed"
    
    logger.info("Knowledge Base tests passed")
    
    # Shutdown the agent
    await agent.shutdown()
    
    return True


async def test_navigation_system():
    """Test the Navigation System component."""
    logger.info("Testing Navigation System")
    
    # Create the agent
    agent = TravelingAgent("NavTest")
    
    # Create and register components
    kb = KnowledgeBase("test_knowledge.db")
    nav_system = NavigationSystem()
    
    agent.register_component(kb)
    agent.register_component(nav_system)
    
    # Initialize the agent
    await agent.initialize()
    
    # Add locations
    loc1 = kb.add_location(
        name="Location 1",
        latitude=40.7128,
        longitude=-74.0060,
        description="Starting location"
    )
    
    loc2 = kb.add_location(
        name="Location 2",
        latitude=40.7484,
        longitude=-73.9857,
        description="Destination location"
    )
    
    # Import map from knowledge base
    success = nav_system.import_map_from_knowledge_base()
    assert success, "Failed to import map from knowledge base"
    
    # Set current location
    success = nav_system.set_current_location(loc1)
    assert success, "Failed to set current location"
    assert nav_system.current_location_id == loc1, "Current location not set correctly"
    
    # Navigate to destination
    success = await nav_system.navigate_to(loc2)
    assert success, "Failed to start navigation"
    assert nav_system.destination_id == loc2, "Destination not set correctly"
    assert nav_system.navigation_active, "Navigation not active"
    
    # Let navigation run for a bit
    for _ in range(5):
        await nav_system.update()
        await asyncio.sleep(0.1)
        
    # Check that progress is being made
    assert nav_system.route_progress > 0, "No progress made in navigation"
    
    # Test path finding
    path = nav_system.find_path(loc1, loc2)
    assert len(path) > 0, "Failed to find path"
    assert path[0] == loc1, "Path does not start at the correct location"
    assert path[-1] == loc2, "Path does not end at the correct location"
    
    # Test exploration
    success = await nav_system.explore_area(loc1)
    assert success, "Failed to start exploration"
    
    # Let exploration run for a bit
    await asyncio.sleep(1)
    
    # Export map to knowledge base
    success = nav_system.export_map_to_knowledge_base()
    assert success, "Failed to export map to knowledge base"
    
    logger.info("Navigation System tests passed")
    
    # Shutdown the agent
    await agent.shutdown()
    
    return True


async def test_perception_module():
    """Test the Perception Module component."""
    logger.info("Testing Perception Module")
    
    # Create the agent
    agent = TravelingAgent("PerceptionTest")
    
    # Create and register the perception module
    perception = PerceptionModule()
    agent.register_component(perception)
    
    # Initialize the agent
    await agent.initialize()
    
    # Start data collection
    success = perception.start_data_collection(interval=0.5)
    assert success, "Failed to start data collection"
    assert perception.data_collection_active, "Data collection not active"
    
    # Wait for data to be collected
    await asyncio.sleep(2)
    
    # Check that data was collected
    assert len(perception.collected_data) > 0, "No data collected"
    
    # Process the collected data
    results = await perception.process_data()
    assert results, "Data processing failed"
    
    # Stop data collection
    success = perception.stop_data_collection()
    assert success, "Failed to stop data collection"
    assert not perception.data_collection_active, "Data collection still active"
    
    # Export data
    export_file = perception.export_data()
    assert export_file, "Data export failed"
    assert os.path.exists(export_file), "Export file does not exist"
    
    logger.info("Perception Module tests passed")
    
    # Shutdown the agent
    await agent.shutdown()
    
    return True


async def test_learning_module():
    """Test the Learning Module component."""
    logger.info("Testing Learning Module")
    
    # Create the agent
    agent = TravelingAgent("LearningTest")
    
    # Create and register components
    perception = PerceptionModule()
    kb = KnowledgeBase("test_knowledge.db")
    learning = LearningModule()
    
    agent.register_component(perception)
    agent.register_component(kb)
    agent.register_component(learning)
    
    # Initialize the agent
    await agent.initialize()
    
    # Add test locations
    loc1 = kb.add_location(
        name="Location 1",
        latitude=40.7128,
        longitude=-74.0060,
        description="Test location 1"
    )
    
    loc2 = kb.add_location(
        name="Location 2",
        latitude=40.7484,
        longitude=-73.9857,
        description="Test location 2"
    )
    
    # Mark locations as visited
    kb.mark_location_visited(loc1)
    kb.mark_location_visited(loc2)
    
    # Start data collection
    perception.start_data_collection(interval=0.5)
    
    # Start learning
    success = learning.start_learning(interval=1.0)
    assert success, "Failed to start learning"
    assert learning.learning_active, "Learning not active"
    
    # Wait for data to be collected and processed
    await asyncio.sleep(3)
    
    # Train models
    results = await learning.train_models()
    
    # Stop learning and data collection
    learning.stop_learning()
    perception.stop_data_collection()
    
    # Export models
    export_file = learning.export_models()
    assert export_file, "Model export failed"
    assert os.path.exists(export_file), "Export file does not exist"
    
    logger.info("Learning Module tests passed")
    
    # Shutdown the agent
    await agent.shutdown()
    
    return True


async def test_visualization_module():
    """Test the Visualization Module component."""
    logger.info("Testing Visualization Module")
    
    # Create the agent
    agent = TravelingAgent("VisualizationTest")
    
    # Create and register components
    perception = PerceptionModule()
    resource_manager = ResourceManager()
    kb = KnowledgeBase("test_knowledge.db")
    nav_system = NavigationSystem()
    visualization = VisualizationModule()
    
    agent.register_component(perception)
    agent.register_component(resource_manager)
    agent.register_component(kb)
    agent.register_component(nav_system)
    agent.register_component(visualization)
    
    # Initialize the agent
    await agent.initialize()
    
    # Add test locations
    loc1 = kb.add_location(
        name="Location 1",
        latitude=40.7128,
        longitude=-74.0060,
        description="Test location 1"
    )
    
    loc2 = kb.add_location(
        name="Location 2",
        latitude=40.7484,
        longitude=-73.9857,
        description="Test location 2"
    )
    
    # Mark locations as visited
    kb.mark_location_visited(loc1)
    
    # Add a route
    kb.add_route(
        start_location_id=loc1,
        end_location_id=loc2,
        distance=5.0,
        estimated_time=60.0,
        difficulty=3
    )
    
    # Import map from knowledge base
    nav_system.import_map_from_knowledge_base()
    
    # Set current location
    nav_system.set_current_location(loc1)
    
    # Start data collection
    perception.start_data_collection(interval=0.5)
    
    # Wait for data to be collected
    await asyncio.sleep(2)
    
    # Create visualizations
    agent_status_path = await visualization.create_agent_status_visualization()
    assert agent_status_path, "Agent status visualization failed"
    assert os.path.exists(agent_status_path), "Agent status visualization file does not exist"
    
    map_path = await visualization.create_map_visualization()
    assert map_path, "Map visualization failed"
    assert os.path.exists(map_path), "Map visualization file does not exist"
    
    resource_path = await visualization.create_resource_usage_visualization()
    assert resource_path, "Resource usage visualization failed"
    assert os.path.exists(resource_path), "Resource usage visualization file does not exist"
    
    env_data_path = await visualization.create_environmental_data_visualization()
    assert env_data_path, "Environmental data visualization failed"
    assert os.path.exists(env_data_path), "Environmental data visualization file does not exist"
    
    dashboard_path = await visualization.create_dashboard()
    assert dashboard_path, "Dashboard creation failed"
    assert os.path.exists(dashboard_path), "Dashboard file does not exist"
    
    # Stop data collection
    perception.stop_data_collection()
    
    logger.info("Visualization Module tests passed")
    
    # Shutdown the agent
    await agent.shutdown()
    
    return True


async def test_decision_engine():
    """Test the Decision Engine component."""
    logger.info("Testing Decision Engine")
    
    # Create the agent
    agent = TravelingAgent("DecisionTest")
    
    # Create and register components
    resource_manager = ResourceManager()
    kb = KnowledgeBase("test_knowledge.db")
    decision_engine = DecisionEngine()
    
    agent.register_component(resource_manager)
    agent.register_component(kb)
    agent.register_component(decision_engine)
    
    # Initialize the agent
    await agent.initialize()
    
    # Add goals and tasks
    goal_id = "test_goal"
    goal = decision_engine.add_goal(
        goal_id=goal_id,
        description="Test Goal",
        priority=GoalPriority.MEDIUM
    )
    assert goal, "Failed to add goal"
    
    task1_id = "test_task1"
    task1 = decision_engine.add_task(
        task_id=task1_id,
        goal_id=goal_id,
        description="Test Task 1",
        metadata={"strategy": "explore"}
    )
    assert task1, "Failed to add task 1"
    
    task2_id = "test_task2"
    task2 = decision_engine.add_task(
        task_id=task2_id,
        goal_id=goal_id,
        description="Test Task 2",
        dependencies=[task1_id],
        metadata={"strategy": "collect_data", "location": "Test Area"}
    )
    assert task2, "Failed to add task 2"
    
    # Run the decision engine for a bit
    for _ in range(5):
        await decision_engine.update()
        await asyncio.sleep(0.1)
        
    # Complete a task
    success = decision_engine.complete_task(task1_id)
    assert success, "Failed to complete task"
    assert decision_engine.tasks[task1_id].completed, "Task not marked as completed"
    
    # Run the decision engine again
    for _ in range(5):
        await decision_engine.update()
        await asyncio.sleep(0.1)
        
    # Complete the goal
    success = decision_engine.complete_goal(goal_id)
    assert success, "Failed to complete goal"
    assert decision_engine.goals[goal_id].completed, "Goal not marked as completed"
    
    logger.info("Decision Engine tests passed")
    
    # Shutdown the agent
    await agent.shutdown()
    
    return True


async def test_full_integration():
    """Test full integration of all components."""
    logger.info("Testing full integration")
    
    # Create the agent
    agent = TravelingAgent("IntegrationTest")
    
    # Create all components
    resource_manager = ResourceManager()
    kb = KnowledgeBase("integration_test.db")
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
    
    # Add locations
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
    
    decision_engine.add_task(
        task_id=f"{goal_id}_task2",
        goal_id=goal_id,
        description=f"Collect data at Target Location",
        dependencies=[f"{goal_id}_task1"],
        metadata={"strategy": "collect_data", "location": "Target Location"}
    )
    
    decision_engine.add_task(
        task_id=f"{goal_id}_task3",
        goal_id=goal_id,
        description=f"Process collected data",
        dependencies=[f"{goal_id}_task2"],
        metadata={"strategy": "process_data"}
    )
    
    # Run the agent for a short time
    agent_task = asyncio.create_task(agent.run())
    
    # Wait for a bit to let the agent run
    await asyncio.sleep(10)
    
    # Navigate to the target location
    await nav_system.navigate_to(target)
    
    # Wait for navigation to complete
    await asyncio.sleep(5)
    
    # Create visualizations
    dashboard_path = await visualization.create_dashboard()
    assert dashboard_path, "Dashboard creation failed"
    
    # Stop data collection and learning
    perception.stop_data_collection()
    learning.stop_learning()
    
    # Shutdown the agent
    await agent.shutdown()
    await agent_task
    
    logger.info("Full integration test passed")
    
    return True


async def run_all_tests():
    """Run all tests."""
    logger.info("Starting all tests")
    
    tests = [
        test_initialization,
        test_knowledge_base,
        test_navigation_system,
        test_perception_module,
        test_learning_module,
        test_visualization_module,
        test_decision_engine,
        test_full_integration
    ]
    
    results = []
    
    for test in tests:
        try:
            logger.info(f"Running test: {test.__name__}")
            result = await test()
            results.append((test.__name__, result))
            logger.info(f"Test {test.__name__} {'passed' if result else 'failed'}")
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with exception: {e}")
            results.append((test.__name__, False))
            
    # Print summary
    logger.info("Test Summary:")
    passed = 0
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{name}: {status}")
        if result:
            passed += 1
            
    logger.info(f"Passed {passed}/{len(tests)} tests")
    
    return passed == len(tests)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
