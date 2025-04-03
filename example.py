#!/usr/bin/env python3
"""
Traveling AI Agent - Main Example

This script demonstrates how to set up and run a complete Traveling AI Agent
with all components configured for a basic exploration scenario.
"""

import asyncio
import logging
import os
import time
from datetime import datetime

from agent_core import TravelingAgent
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
        logging.FileHandler('agent.log')
    ]
)

logger = logging.getLogger("main")


async def main():
    """Run the Traveling AI Agent with a basic exploration scenario."""
    logger.info("Starting Traveling AI Agent")
    
    # Create the agent
    agent = TravelingAgent("ExplorerAgent")
    
    # Create all components
    resource_manager = ResourceManager()
    kb = KnowledgeBase("explorer_knowledge.db")
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
    
    # Create directories for outputs
    os.makedirs("data", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Add locations to the knowledge base
    logger.info("Adding locations to knowledge base")
    
    home = kb.add_location(
        name="Home Base",
        latitude=40.7128,
        longitude=-74.0060,
        description="Starting location with basic facilities"
    )
    
    location1 = kb.add_location(
        name="Mountain Peak",
        latitude=40.7484,
        longitude=-73.9857,
        description="High elevation point with panoramic views"
    )
    
    location2 = kb.add_location(
        name="Forest Clearing",
        latitude=40.7300,
        longitude=-73.9950,
        description="Open area surrounded by dense forest"
    )
    
    location3 = kb.add_location(
        name="Lakeside",
        latitude=40.7400,
        longitude=-74.0100,
        description="Shore of a large freshwater lake"
    )
    
    # Add points of interest
    logger.info("Adding points of interest")
    
    kb.add_point_of_interest(
        name="Weather Station",
        location_id=home,
        poi_type="facility",
        description="Automated weather monitoring station",
        importance=7
    )
    
    kb.add_point_of_interest(
        name="Observation Tower",
        location_id=location1,
        poi_type="landmark",
        description="Tall tower providing views of the surrounding area",
        importance=8
    )
    
    kb.add_point_of_interest(
        name="Ancient Tree",
        location_id=location2,
        poi_type="natural",
        description="Centuries-old tree with historical significance",
        importance=6
    )
    
    kb.add_point_of_interest(
        name="Research Station",
        location_id=location3,
        poi_type="facility",
        description="Small research outpost for environmental studies",
        importance=9
    )
    
    # Add facts
    logger.info("Adding facts to knowledge base")
    
    kb.add_fact(
        subject="Weather Station",
        predicate="measures",
        obj="temperature, humidity, pressure",
        confidence=1.0,
        source="facility_records"
    )
    
    kb.add_fact(
        subject="Observation Tower",
        predicate="height",
        obj="50 meters",
        confidence=0.95,
        source="geographical_survey"
    )
    
    kb.add_fact(
        subject="Ancient Tree",
        predicate="age",
        obj="approximately 500 years",
        confidence=0.8,
        source="dendrochronology_study"
    )
    
    kb.add_fact(
        subject="Research Station",
        predicate="purpose",
        obj="environmental monitoring and biodiversity studies",
        confidence=0.9,
        source="research_records"
    )
    
    # Import map from knowledge base
    logger.info("Importing map from knowledge base")
    nav_system.import_map_from_knowledge_base()
    
    # Set current location
    logger.info("Setting current location to Home Base")
    nav_system.set_current_location(home)
    
    # Mark home as visited
    kb.mark_location_visited(home)
    
    # Start data collection
    logger.info("Starting data collection")
    perception.start_data_collection(interval=2.0)
    
    # Start learning
    logger.info("Starting learning processes")
    learning.start_learning(interval=10.0)
    
    # Add exploration goals
    logger.info("Setting up exploration goals")
    
    # Goal 1: Explore Mountain Peak
    goal1_id = "explore_mountain"
    decision_engine.add_goal(
        goal_id=goal1_id,
        description="Explore Mountain Peak",
        priority=GoalPriority.HIGH
    )
    
    # Tasks for Goal 1
    decision_engine.add_task(
        task_id=f"{goal1_id}_task1",
        goal_id=goal1_id,
        description="Navigate to Mountain Peak",
        metadata={"strategy": "navigate", "target_id": location1}
    )
    
    decision_engine.add_task(
        task_id=f"{goal1_id}_task2",
        goal_id=goal1_id,
        description="Collect environmental data at Mountain Peak",
        dependencies=[f"{goal1_id}_task1"],
        metadata={"strategy": "collect_data", "location": "Mountain Peak", "duration": 30}
    )
    
    decision_engine.add_task(
        task_id=f"{goal1_id}_task3",
        goal_id=goal1_id,
        description="Document Observation Tower",
        dependencies=[f"{goal1_id}_task2"],
        metadata={"strategy": "document", "poi": "Observation Tower"}
    )
    
    # Goal 2: Explore Forest Clearing
    goal2_id = "explore_forest"
    decision_engine.add_goal(
        goal_id=goal2_id,
        description="Explore Forest Clearing",
        priority=GoalPriority.MEDIUM
    )
    
    # Tasks for Goal 2
    decision_engine.add_task(
        task_id=f"{goal2_id}_task1",
        goal_id=goal2_id,
        description="Navigate to Forest Clearing",
        metadata={"strategy": "navigate", "target_id": location2}
    )
    
    decision_engine.add_task(
        task_id=f"{goal2_id}_task2",
        goal_id=goal2_id,
        description="Collect environmental data at Forest Clearing",
        dependencies=[f"{goal2_id}_task1"],
        metadata={"strategy": "collect_data", "location": "Forest Clearing", "duration": 30}
    )
    
    decision_engine.add_task(
        task_id=f"{goal2_id}_task3",
        goal_id=goal2_id,
        description="Study Ancient Tree",
        dependencies=[f"{goal2_id}_task2"],
        metadata={"strategy": "document", "poi": "Ancient Tree"}
    )
    
    # Goal 3: Explore Lakeside
    goal3_id = "explore_lakeside"
    decision_engine.add_goal(
        goal_id=goal3_id,
        description="Explore Lakeside",
        priority=GoalPriority.LOW
    )
    
    # Tasks for Goal 3
    decision_engine.add_task(
        task_id=f"{goal3_id}_task1",
        goal_id=goal3_id,
        description="Navigate to Lakeside",
        metadata={"strategy": "navigate", "target_id": location3}
    )
    
    decision_engine.add_task(
        task_id=f"{goal3_id}_task2",
        goal_id=goal3_id,
        description="Collect environmental data at Lakeside",
        dependencies=[f"{goal3_id}_task1"],
        metadata={"strategy": "collect_data", "location": "Lakeside", "duration": 30}
    )
    
    decision_engine.add_task(
        task_id=f"{goal3_id}_task3",
        goal_id=goal3_id,
        description="Visit Research Station",
        dependencies=[f"{goal3_id}_task2"],
        metadata={"strategy": "document", "poi": "Research Station"}
    )
    
    # Create initial visualizations
    logger.info("Creating initial visualizations")
    await visualization.create_agent_status_visualization()
    await visualization.create_map_visualization()
    
    # Run the agent
    logger.info("Starting agent run loop")
    agent_task = asyncio.create_task(agent.run())
    
    try:
        # Simulate exploration for a period of time
        total_runtime = 120  # seconds
        update_interval = 10  # seconds
        
        for i in range(total_runtime // update_interval):
            # Wait for the update interval
            await asyncio.sleep(update_interval)
            
            # Log progress
            logger.info(f"Agent running for {(i+1) * update_interval} seconds")
            
            # Create visualizations periodically
            if i % 3 == 0:  # Every 30 seconds
                logger.info("Creating visualizations")
                await visualization.create_dashboard()
                
            # Simulate completing tasks based on time
            if i == 3:  # After 30 seconds
                logger.info("Simulating completion of first navigation task")
                decision_engine.complete_task(f"{goal1_id}_task1")
                nav_system.set_current_location(location1)
                kb.mark_location_visited(location1)
                
            if i == 6:  # After 60 seconds
                logger.info("Simulating completion of first data collection task")
                decision_engine.complete_task(f"{goal1_id}_task2")
                
            if i == 9:  # After 90 seconds
                logger.info("Simulating completion of first documentation task")
                decision_engine.complete_task(f"{goal1_id}_task3")
                decision_engine.complete_goal(goal1_id)
                
                # Start navigation to next location
                logger.info("Starting navigation to Forest Clearing")
                await nav_system.navigate_to(location2)
                
        # Create final visualizations
        logger.info("Creating final visualizations")
        dashboard_path = await visualization.create_dashboard()
        logger.info(f"Final dashboard saved to: {dashboard_path}")
        
        # Export collected data
        logger.info("Exporting collected data")
        data_export = perception.export_data()
        logger.info(f"Data exported to: {data_export}")
        
        # Export models
        logger.info("Exporting learned models")
        model_export = learning.export_models()
        logger.info(f"Models exported to: {model_export}")
        
        # Export knowledge
        logger.info("Exporting knowledge base")
        kb_export = kb.export_knowledge()
        logger.info(f"Knowledge exported to: {kb_export}")
        
    except Exception as e:
        logger.error(f"Error during agent execution: {e}")
    finally:
        # Stop data collection and learning
        logger.info("Stopping data collection and learning")
        perception.stop_data_collection()
        learning.stop_learning()
        
        # Shutdown the agent
        logger.info("Shutting down agent")
        await agent.shutdown()
        
        # Cancel the agent task if it's still running
        if not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
    
    logger.info("Agent execution completed")


if __name__ == "__main__":
    asyncio.run(main())
