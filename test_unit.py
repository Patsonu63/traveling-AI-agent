#!/usr/bin/env python3
"""
Traveling AI Agent - Unit Tests

This script contains unit tests for individual components of the Traveling AI Agent.
"""

import unittest
import asyncio
import os
import tempfile
from datetime import datetime

from agent_core import TravelingAgent, Component, AgentState
from resource_manager import ResourceManager
from knowledge_base import KnowledgeBase
from decision_engine import DecisionEngine, GoalPriority
from navigation_system import NavigationSystem, Coordinates, Location
from perception_module import PerceptionModule, SensorType
from learning_module import LearningModule


class TestAgentCore(unittest.TestCase):
    """Test cases for the Agent Core component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.agent = TravelingAgent("TestAgent")
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.loop.run_until_complete(self.agent.shutdown())
        self.loop.close()
        
    def test_agent_initialization(self):
        """Test agent initialization."""
        # Run initialization
        success = self.loop.run_until_complete(self.agent.initialize())
        
        # Check results
        self.assertTrue(success)
        self.assertEqual(self.agent.state, AgentState.IDLE)
        
    def test_component_registration(self):
        """Test component registration."""
        # Create a test component
        class TestComponent(Component):
            def __init__(self):
                super().__init__("TestComponent")
                
            async def initialize(self):
                self.active = True
                return True
                
            async def shutdown(self):
                self.active = False
                return True
                
        # Register the component
        component = TestComponent()
        self.agent.register_component(component)
        
        # Check registration
        self.assertIn("TestComponent", self.agent.components)
        self.assertEqual(component.agent, self.agent)
        
        # Initialize the agent
        success = self.loop.run_until_complete(self.agent.initialize())
        self.assertTrue(success)
        
        # Check component state
        self.assertTrue(component.active)
        
    def test_event_bus(self):
        """Test the event bus."""
        # Create a test event handler
        event_received = False
        event_data = None
        
        async def test_handler(data):
            nonlocal event_received, event_data
            event_received = True
            event_data = data
            
        # Subscribe to an event
        self.agent.event_bus.subscribe("test_event", test_handler)
        
        # Publish an event
        test_data = {"test": "data"}
        self.loop.run_until_complete(self.agent.event_bus.publish("test_event", test_data))
        
        # Check that the event was received
        self.assertTrue(event_received)
        self.assertEqual(event_data, test_data)
        
        # Unsubscribe from the event
        self.agent.event_bus.unsubscribe("test_event", test_handler)
        
        # Reset flags
        event_received = False
        event_data = None
        
        # Publish another event
        self.loop.run_until_complete(self.agent.event_bus.publish("test_event", {"test": "more data"}))
        
        # Check that the event was not received
        self.assertFalse(event_received)
        self.assertIsNone(event_data)


class TestKnowledgeBase(unittest.TestCase):
    """Test cases for the Knowledge Base component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.agent = TravelingAgent("TestAgent")
        
        # Create a temporary database file
        fd, self.db_path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        # Create and register the knowledge base
        self.kb = KnowledgeBase(self.db_path)
        self.agent.register_component(self.kb)
        
        # Initialize the agent
        self.loop.run_until_complete(self.agent.initialize())
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.loop.run_until_complete(self.agent.shutdown())
        self.loop.close()
        
        # Remove the temporary database file
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
            
    def test_location_operations(self):
        """Test location operations."""
        # Add a location
        location_id = self.kb.add_location(
            name="Test Location",
            latitude=40.7128,
            longitude=-74.0060,
            description="A test location"
        )
        self.assertGreater(location_id, 0)
        
        # Get the location
        location = self.kb.get_location(location_id)
        self.assertIsNotNone(location)
        self.assertEqual(location["name"], "Test Location")
        self.assertEqual(location["latitude"], 40.7128)
        self.assertEqual(location["longitude"], -74.0060)
        self.assertEqual(location["description"], "A test location")
        self.assertEqual(location["visited"], 0)
        
        # Mark the location as visited
        success = self.kb.mark_location_visited(location_id)
        self.assertTrue(success)
        
        # Get the location again
        location = self.kb.get_location(location_id)
        self.assertEqual(location["visited"], 1)
        self.assertEqual(location["visit_count"], 1)
        self.assertIsNotNone(location["last_visited"])
        
        # Get all locations
        locations = self.kb.get_locations()
        self.assertEqual(len(locations), 1)
        
        # Get visited locations
        visited_locations = self.kb.get_locations(visited=True)
        self.assertEqual(len(visited_locations), 1)
        
        # Get unvisited locations
        unvisited_locations = self.kb.get_locations(visited=False)
        self.assertEqual(len(unvisited_locations), 0)
        
    def test_point_of_interest_operations(self):
        """Test point of interest operations."""
        # Add a location
        location_id = self.kb.add_location(
            name="Test Location",
            latitude=40.7128,
            longitude=-74.0060,
            description="A test location"
        )
        
        # Add a point of interest
        poi_id = self.kb.add_point_of_interest(
            name="Test POI",
            location_id=location_id,
            poi_type="landmark",
            description="A test point of interest",
            importance=5
        )
        self.assertGreater(poi_id, 0)
        
        # Get points of interest
        pois = self.kb.get_points_of_interest(location_id)
        self.assertEqual(len(pois), 1)
        self.assertEqual(pois[0]["name"], "Test POI")
        self.assertEqual(pois[0]["type"], "landmark")
        self.assertEqual(pois[0]["description"], "A test point of interest")
        self.assertEqual(pois[0]["importance"], 5)
        
    def test_fact_operations(self):
        """Test fact operations."""
        # Add a fact
        fact_id = self.kb.add_fact(
            subject="Test Subject",
            predicate="test_predicate",
            obj="Test Object",
            confidence=0.9,
            source="Test Source"
        )
        self.assertGreater(fact_id, 0)
        
        # Get facts by subject
        facts = self.kb.get_facts(subject="Test Subject")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["subject"], "Test Subject")
        self.assertEqual(facts[0]["predicate"], "test_predicate")
        self.assertEqual(facts[0]["object"], "Test Object")
        self.assertEqual(facts[0]["confidence"], 0.9)
        self.assertEqual(facts[0]["source"], "Test Source")
        
        # Get facts by predicate
        facts = self.kb.get_facts(predicate="test_predicate")
        self.assertEqual(len(facts), 1)
        
        # Get facts by object
        facts = self.kb.get_facts(obj="Test Object")
        self.assertEqual(len(facts), 1)
        
        # Get facts by multiple criteria
        facts = self.kb.get_facts(subject="Test Subject", predicate="test_predicate")
        self.assertEqual(len(facts), 1)
        
        # Get facts with non-matching criteria
        facts = self.kb.get_facts(subject="Nonexistent")
        self.assertEqual(len(facts), 0)
        
    def test_memory_operations(self):
        """Test memory operations."""
        # Store a simple value
        self.kb.remember("test_key", "test_value")
        value = self.kb.recall("test_key")
        self.assertEqual(value, "test_value")
        
        # Store a complex value
        complex_value = {
            "name": "Test",
            "values": [1, 2, 3],
            "nested": {"a": 1, "b": 2}
        }
        self.kb.remember("complex_key", complex_value)
        recalled_value = self.kb.recall("complex_key")
        self.assertEqual(recalled_value, complex_value)
        
        # Forget a value
        self.kb.forget("test_key")
        value = self.kb.recall("test_key")
        self.assertIsNone(value)
        
        # Recall a non-existent key
        value = self.kb.recall("nonexistent_key")
        self.assertIsNone(value)


class TestNavigationSystem(unittest.TestCase):
    """Test cases for the Navigation System component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.agent = TravelingAgent("TestAgent")
        
        # Create and register components
        self.nav_system = NavigationSystem()
        self.agent.register_component(self.nav_system)
        
        # Initialize the agent
        self.loop.run_until_complete(self.agent.initialize())
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.loop.run_until_complete(self.agent.shutdown())
        self.loop.close()
        
    def test_coordinates(self):
        """Test coordinate operations."""
        # Create coordinates
        coord1 = Coordinates(40.7128, -74.0060)  # New York
        coord2 = Coordinates(34.0522, -118.2437)  # Los Angeles
        
        # Calculate distance
        distance = coord1.distance_to(coord2)
        
        # The distance should be approximately 3935 km
        self.assertAlmostEqual(distance, 3935, delta=100)
        
    def test_location_operations(self):
        """Test location operations."""
        # Create locations
        loc1 = Location(
            id=1,
            name="Location 1",
            coordinates=Coordinates(40.7128, -74.0060),
            description="Test location 1"
        )
        
        loc2 = Location(
            id=2,
            name="Location 2",
            coordinates=Coordinates(34.0522, -118.2437),
            description="Test location 2"
        )
        
        # Add locations to map
        self.nav_system.map.add_location(loc1)
        self.nav_system.map.add_location(loc2)
        
        # Get locations
        retrieved_loc1 = self.nav_system.map.get_location(1)
        self.assertEqual(retrieved_loc1, loc1)
        
        retrieved_loc2 = self.nav_system.map.get_location(2)
        self.assertEqual(retrieved_loc2, loc2)
        
        # Calculate distance between locations
        distance = loc1.distance_to(loc2)
        self.assertAlmostEqual(distance, 3935, delta=100)
        
        # Find nearest location
        nearest = self.nav_system.map.find_nearest_location(Coordinates(40.0, -74.0))
        self.assertEqual(nearest, loc1)
        
    def test_route_planning(self):
        """Test route planning."""
        # Create locations
        loc1 = Location(
            id=1,
            name="Location 1",
            coordinates=Coordinates(40.7128, -74.0060),
            description="Test location 1"
        )
        
        loc2 = Location(
            id=2,
            name="Location 2",
            coordinates=Coordinates(40.7484, -73.9857),
            description="Test location 2"
        )
        
        # Add locations to map
        self.nav_system.map.add_location(loc1)
        self.nav_system.map.add_location(loc2)
        
        # Plan a route
        route = self.nav_system._plan_route(loc1, loc2)
        
        # Check route properties
        self.assertEqual(route.start_location, loc1)
        self.assertEqual(route.end_location, loc2)
        self.assertGreater(route.distance, 0)
        self.assertGreater(route.estimated_time, 0)
        self.assertGreaterEqual(route.difficulty, 0)
        self.assertLessEqual(route.difficulty, 10)
        
        # Add route to map
        self.nav_system.map.add_route(route)
        
        # Find route
        found_route = self.nav_system.map.find_route(1, 2)
        self.assertEqual(found_route, route)
        
    def test_navigation(self):
        """Test navigation."""
        # Create locations
        loc1 = Location(
            id=1,
            name="Location 1",
            coordinates=Coordinates(40.7128, -74.0060),
            description="Test location 1"
        )
        
        loc2 = Location(
            id=2,
            name="Location 2",
            coordinates=Coordinates(40.7484, -73.9857),
            description="Test location 2"
        )
        
        # Add locations to map
        self.nav_system.map.add_location(loc1)
        self.nav_system.map.add_location(loc2)
        
        # Set current location
        success = self.nav_system.set_current_location(1)
        self.assertTrue(success)
        self.assertEqual(self.nav_system.current_location_id, 1)
        
        # Start navigation
        success = self.loop.run_until_complete(self.nav_system.navigate_to(2))
        self.assertTrue(success)
        self.assertEqual(self.nav_system.destination_id, 2)
        self.assertTrue(self.nav_system.navigation_active)
        
        # Update navigation a few times
        for _ in range(5):
            self.loop.run_until_complete(self.nav_system.update())
            
        # Check that progress is being made
        self.assertGreater(self.nav_system.route_progress, 0)
        
    def test_path_finding(self):
        """Test path finding."""
        # Create locations
        loc1 = Location(
            id=1,
            name="Location 1",
            coordinates=Coordinates(40.7128, -74.0060),
            description="Test location 1"
        )
        
        loc2 = Location(
            id=2,
            name="Location 2",
            coordinates=Coordinates(40.7484, -73.9857),
            description="Test location 2"
        )
        
        loc3 = Location(
            id=3,
            name="Location 3",
            coordinates=Coordinates(40.7300, -73.9950),
            description="Test location 3"
        )
        
        # Add locations to map
        self.nav_system.map.add_location(loc1)
        self.nav_system.map.add_location(loc2)
        self.nav_system.map.add_location(loc3)
        
        # Create routes
        route1_2 = self.nav_system._plan_route(loc1, loc2)
        route1_3 = self.nav_system._plan_route(loc1, loc3)
        route3_2 = self.nav_system._plan_route(loc3, loc2)
        
        # Add routes to map
        self.nav_system.map.add_route(route1_2)
        self.nav_system.map.add_route(route1_3)
        self.nav_system.map.add_route(route3_2)
        
        # Find direct path
        path1_2 = self.nav_system.find_path(1, 2)
        self.assertEqual(path1_2, [1, 2])
        
        # Find path through intermediate location
        # Note: A* might find the direct path if it's shorter
        path1_3_2 = self.nav_system.find_path(1, 2)
        self.assertTrue(path1_3_2 == [1, 2] or path1_3_2 == [1, 3, 2])


class TestDecisionEngine(unittest.TestCase):
    """Test cases for the Decision Engine component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.agent = TravelingAgent("TestAgent")
        
        # Create and register the decision engine
        self.decision_engine = DecisionEngine()
        self.agent.register_component(self.decision_engine)
        
        # Initialize the agent
        self.loop.run_until_complete(self.agent.initialize())
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.loop.run_until_complete(self.agent.shutdown())
        self.loop.close()
        
    def test_goal_management(self):
        """Test goal management."""
        # Add a goal
        goal_id = "test_goal"
        goal = self.decision_engine.add_goal(
            goal_id=goal_id,
            description="Test Goal",
            priority=GoalPriority.MEDIUM
        )
        
        # Check goal properties
        self.assertEqual(goal.id, goal_id)
        self.assertEqual(goal.description, "Test Goal")
        self.assertEqual(goal.priority, GoalPriority.MEDIUM)
        self.assertFalse(goal.completed)
        
        # Complete the goal
        success = self.decision_engine.complete_goal(goal_id)
        self.assertTrue(success)
        
        # Check that the goal is completed
        self.assertTrue(self.decision_engine.goals[goal_id].completed)
        self.assertIsNotNone(self.decision_engine.goals[goal_id].completed_at)
        
    def test_task_management(self):
        """Test task management."""
        # Add a goal
        goal_id = "test_goal"
        self.decision_engine.add_goal(
            goal_id=goal_id,
            description="Test Goal",
            priority=GoalPriority.MEDIUM
        )
        
        # Add tasks
        task1_id = "test_task1"
        task1 = self.decision_engine.add_task(
            task_id=task1_id,
            goal_id=goal_id,
            description="Test Task 1",
            metadata={"strategy": "explore"}
        )
        
        task2_id = "test_task2"
        task2 = self.decision_engine.add_task(
            task_id=task2_id,
            goal_id=goal_id,
            description="Test Task 2",
            dependencies=[task1_id],
            metadata={"strategy": "collect_data"}
        )
        
        # Check task properties
        self.assertEqual(task1.id, task1_id)
        self.assertEqual(task1.goal_id, goal_id)
        self.assertEqual(task1.description, "Test Task 1")
        self.assertFalse(task1.completed)
        
        self.assertEqual(task2.id, task2_id)
        self.assertEqual(task2.goal_id, goal_id)
        self.assertEqual(task2.description, "Test Task 2")
        self.assertFalse(task2.completed)
        self.assertEqual(task2.dependencies, [task1_id])
        
        # Complete task 1
        success = self.decision_engine.complete_task(task1_id)
        self.assertTrue(success)
        
        # Check that task 1 is completed
        self.assertTrue(self.decision_engine.tasks[task1_id].completed)
        self.assertIsNotNone(self.decision_engine.tasks[task1_id].completed_at)
        
        # Complete task 2
        success = self.decision_engine.complete_task(task2_id)
        self.assertTrue(success)
        
        # Check that task 2 is completed
        self.assertTrue(self.decision_engine.tasks[task2_id].completed)
        
        # Check that the goal is automatically completed when all tasks are completed
        self.assertTrue(self.decision_engine.goals[goal_id].completed)
        
    def test_decision_making(self):
        """Test decision making."""
        # Add a goal
        goal_id = "test_goal"
        self.decision_engine.add_goal(
            goal_id=goal_id,
            description="Test Goal",
            priority=GoalPriority.MEDIUM
        )
        
        # Add a task
        task_id = "test_task"
        self.decision_engine.add_task(
            task_id=task_id,
            goal_id=goal_id,
            description="Test Task",
            metadata={"strategy": "explore"}
        )
        
        # Run the decision engine
        self.loop.run_until_complete(self.decision_engine._make_decisions())
        
        # Check that the goal and task were selected
        self.assertEqual(self.decision_engine.current_goal_id, goal_id)
        self.assertEqual(self.decision_engine.current_task_id, task_id)


class TestPerceptionModule(unittest.TestCase):
    """Test cases for the Perception Module component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.agent = TravelingAgent("TestAgent")
        
        # Create and register the perception module
        self.perception = PerceptionModule()
        self.agent.register_component(self.perception)
        
        # Initialize the agent
        self.loop.run_until_complete(self.agent.initialize())
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.loop.run_until_complete(self.agent.shutdown())
        self.loop.close()
        
        # Clean up data directory
        if os.path.exists("data"):
            for root, dirs, files in os.walk("data", topdown=False):
                for file in files:
                    os.unlink(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir("data")
            
    def test_sensor_creation(self):
        """Test sensor creation."""
        # Check that sensors were created
        self.assertIn("camera", self.perception.sensors)
        self.assertIn("temperature", self.perception.sensors)
        self.assertIn("humidity", self.perception.sensors)
        self.assertIn("pressure", self.perception.sensors)
        self.assertIn("light", self.perception.sensors)
        self.assertIn("gps", self.perception.sensors)
        self.assertIn("compass", self.perception.sensors)
        
        # Check sensor types
        self.assertEqual(self.perception.sensors["camera"].sensor_type, SensorType.CAMERA)
        self.assertEqual(self.perception.sensors["temperature"].sensor_type, SensorType.TEMPERATURE)
        self.assertEqual(self.perception.sensors["humidity"].sensor_type, SensorType.HUMIDITY)
        self.assertEqual(self.perception.sensors["pressure"].sensor_type, SensorType.PRESSURE)
        self.assertEqual(self.perception.sensors["light"].sensor_type, SensorType.LIGHT)
        self.assertEqual(self.perception.sensors["gps"].sensor_type, SensorType.GPS)
        self.assertEqual(self.perception.sensors["compass"].sensor_type, SensorType.COMPASS)
        
    def test_data_collection(self):
        """Test data collection."""
        # Collect data
        env_data = self.loop.run_until_complete(self.perception.collect_data())
        
        # Check that data was collected
        self.assertIsNotNone(env_data)
        self.assertIsInstance(env_data.timestamp, datetime)
        self.assertGreater(len(env_data.readings), 0)
        
        # Check that the data was stored
        self.assertEqual(len(self.perception.collected_data), 1)
        self.assertEqual(self.perception.collected_data[0], env_data)
        
    def test_data_processing(self):
        """Test data processing."""
        # Collect data
        self.loop.run_until_complete(self.perception.collect_data())
        
        # Process data
        results = self.loop.run_until_complete(self.perception.process_data())
        
        # Check that results were generated
        self.assertIsNotNone(results)
        
        # Check that the data was marked as processed
        self.assertTrue(self.perception.collected_data[0].processed)
        
    def test_data_collection_control(self):
        """Test data collection control."""
        # Start data collection
        success = self.perception.start_data_collection(interval=0.5)
        self.assertTrue(success)
        self.assertTrue(self.perception.data_collection_active)
        self.assertEqual(self.perception.collection_interval, 0.5)
        
        # Wait for data to be collected
        self.loop.run_until_complete(asyncio.sleep(1.5))
        
        # Check that data was collected
        self.assertGreater(len(self.perception.collected_data), 0)
        
        # Stop data collection
        success = self.perception.stop_data_collection()
        self.assertTrue(success)
        self.assertFalse(self.perception.data_collection_active)


class TestLearningModule(unittest.TestCase):
    """Test cases for the Learning Module component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.agent = TravelingAgent("TestAgent")
        
        # Create and register components
        self.perception = PerceptionModule()
        self.kb = KnowledgeBase(":memory:")  # In-memory database for testing
        self.learning = LearningModule()
        
        self.agent.register_component(self.perception)
        self.agent.register_component(self.kb)
        self.agent.register_component(self.learning)
        
        # Initialize the agent
        self.loop.run_until_complete(self.agent.initialize())
        
        # Create model directory if it doesn't exist
        if not os.path.exists("models"):
            os.makedirs("models")
            
    def tearDown(self):
        """Tear down test fixtures."""
        self.loop.run_until_complete(self.agent.shutdown())
        self.loop.close()
        
        # Clean up model files
        if os.path.exists("models"):
            for file in os.listdir("models"):
                if file.endswith(".pkl") or file.endswith(".json"):
                    os.unlink(os.path.join("models", file))
                    
    def test_learning_control(self):
        """Test learning control."""
        # Start learning
        success = self.learning.start_learning(interval=1.0)
        self.assertTrue(success)
        self.assertTrue(self.learning.learning_active)
        self.assertEqual(self.learning.training_interval, 1.0)
        
        # Stop learning
        success = self.learning.stop_learning()
        self.assertTrue(success)
        self.assertFalse(self.learning.learning_active)
        
    def test_model_export(self):
        """Test model export."""
        # Add a simple model
        self.learning.models["test_model"] = {"test": "data"}
        
        # Export models
        export_file = self.learning.export_models()
        
        # Check that the export file was created
        self.assertTrue(os.path.exists(export_file))
        
        # Check the content of the export file
        with open(export_file, 'r') as f:
            export_data = json.loads(f.read())
            
        self.assertIn("test_model", export_data)
        self.assertEqual(export_data["test_model"], {"test": "data"})


if __name__ == "__main__":
    unittest.main()
