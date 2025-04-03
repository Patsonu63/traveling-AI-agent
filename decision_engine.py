#!/usr/bin/env python3
"""
Traveling AI Agent - Decision Engine Module

This module implements the Decision Engine component responsible for
goal management, task planning, and decision-making for the agent.
"""

import logging
import asyncio
import random
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from agent_core import Component, AgentState


class GoalPriority(Enum):
    """Enumeration of goal priorities."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Goal:
    """Class representing a goal for the agent."""
    id: str
    description: str
    priority: GoalPriority
    created_at: datetime
    completed: bool = False
    completed_at: Optional[datetime] = None
    parent_goal_id: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class Task:
    """Class representing a task for achieving a goal."""
    id: str
    goal_id: str
    description: str
    created_at: datetime
    completed: bool = False
    completed_at: Optional[datetime] = None
    dependencies: List[str] = None  # List of task IDs that must be completed first
    estimated_duration: float = 0.0  # Estimated duration in seconds
    metadata: Dict[str, Any] = None


class DecisionEngine(Component):
    """Decision Engine component for goal management and decision-making."""
    
    def __init__(self):
        """Initialize the Decision Engine component."""
        super().__init__("DecisionEngine")
        self.goals: Dict[str, Goal] = {}
        self.tasks: Dict[str, Task] = {}
        self.current_goal_id: Optional[str] = None
        self.current_task_id: Optional[str] = None
        self.decision_strategies: Dict[str, Callable] = {}
        self.last_decision_time = 0
        self.decision_interval = 5.0  # Seconds between decision cycles
        
    async def initialize(self) -> bool:
        """Initialize the Decision Engine.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.logger.info("Initializing Decision Engine")
        
        # Register default decision strategies
        self._register_default_strategies()
        
        # Subscribe to relevant events
        if self.agent:
            self.agent.event_bus.subscribe("resource_warning", self.on_resource_warning)
            self.agent.event_bus.subscribe("resource_critical", self.on_resource_critical)
            
        self.active = True
        self.last_decision_time = datetime.now().timestamp()
        
        return True
        
    async def shutdown(self) -> bool:
        """Shutdown the Decision Engine.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        self.logger.info("Shutting down Decision Engine")
        
        # Unsubscribe from events
        if self.agent:
            self.agent.event_bus.unsubscribe("resource_warning", self.on_resource_warning)
            self.agent.event_bus.unsubscribe("resource_critical", self.on_resource_critical)
            
        self.active = False
        
        return True
        
    def _register_default_strategies(self) -> None:
        """Register default decision strategies."""
        self.decision_strategies["explore"] = self._explore_strategy
        self.decision_strategies["conserve_energy"] = self._conserve_energy_strategy
        self.decision_strategies["collect_data"] = self._collect_data_strategy
        self.decision_strategies["process_data"] = self._process_data_strategy
        
    async def update(self) -> None:
        """Update the Decision Engine state."""
        current_time = datetime.now().timestamp()
        
        # Make decisions at regular intervals
        if current_time - self.last_decision_time >= self.decision_interval:
            await self._make_decisions()
            self.last_decision_time = current_time
            
        # Update state
        self.state = {
            "current_goal": self.goals.get(self.current_goal_id) if self.current_goal_id else None,
            "current_task": self.tasks.get(self.current_task_id) if self.current_task_id else None,
            "goal_count": len(self.goals),
            "task_count": len(self.tasks),
            "completed_goals": sum(1 for goal in self.goals.values() if goal.completed),
            "completed_tasks": sum(1 for task in self.tasks.values() if task.completed),
        }
        
    async def _make_decisions(self) -> None:
        """Make decisions based on current state and goals."""
        if not self.agent:
            return
            
        # If no current goal, select one
        if not self.current_goal_id or self.goals.get(self.current_goal_id).completed:
            await self._select_next_goal()
            
        # If no current task, select one for the current goal
        if not self.current_task_id or self.tasks.get(self.current_task_id).completed:
            await self._select_next_task()
            
        # Execute the appropriate strategy based on current goal and task
        if self.current_goal_id and self.current_task_id:
            goal = self.goals[self.current_goal_id]
            task = self.tasks[self.current_task_id]
            
            # Choose a strategy based on the task
            strategy_name = task.metadata.get("strategy", "explore")
            if strategy_name in self.decision_strategies:
                strategy = self.decision_strategies[strategy_name]
                await strategy(goal, task)
            else:
                self.logger.warning(f"Unknown strategy: {strategy_name}")
                
    async def _select_next_goal(self) -> None:
        """Select the next goal to pursue."""
        # Find incomplete goals
        incomplete_goals = [g for g in self.goals.values() if not g.completed]
        
        if not incomplete_goals:
            self.logger.info("No incomplete goals available")
            self.current_goal_id = None
            return
            
        # Sort by priority (highest first)
        incomplete_goals.sort(key=lambda g: g.priority.value, reverse=True)
        
        # Select the highest priority goal
        selected_goal = incomplete_goals[0]
        self.current_goal_id = selected_goal.id
        
        self.logger.info(f"Selected goal: {selected_goal.description} (Priority: {selected_goal.priority.name})")
        
        # Publish event
        if self.agent:
            await self.agent.event_bus.publish("goal_selected", {
                "goal_id": selected_goal.id,
                "description": selected_goal.description,
                "priority": selected_goal.priority.name,
            })
            
    async def _select_next_task(self) -> None:
        """Select the next task to pursue for the current goal."""
        if not self.current_goal_id:
            self.current_task_id = None
            return
            
        # Find incomplete tasks for the current goal
        incomplete_tasks = [
            t for t in self.tasks.values() 
            if not t.completed and t.goal_id == self.current_goal_id
        ]
        
        if not incomplete_tasks:
            self.logger.info(f"No incomplete tasks for goal {self.current_goal_id}")
            self.current_task_id = None
            return
            
        # Filter tasks whose dependencies are satisfied
        available_tasks = []
        for task in incomplete_tasks:
            if not task.dependencies:
                available_tasks.append(task)
            else:
                dependencies_met = all(
                    self.tasks.get(dep_id).completed 
                    for dep_id in task.dependencies 
                    if dep_id in self.tasks
                )
                if dependencies_met:
                    available_tasks.append(task)
                    
        if not available_tasks:
            self.logger.info("No available tasks (dependencies not met)")
            self.current_task_id = None
            return
            
        # Select the first available task (could implement more sophisticated selection)
        selected_task = available_tasks[0]
        self.current_task_id = selected_task.id
        
        self.logger.info(f"Selected task: {selected_task.description}")
        
        # Publish event
        if self.agent:
            await self.agent.event_bus.publish("task_selected", {
                "task_id": selected_task.id,
                "description": selected_task.description,
                "goal_id": selected_task.goal_id,
            })
            
    def add_goal(self, goal_id: str, description: str, priority: GoalPriority = GoalPriority.MEDIUM,
                parent_goal_id: str = None, metadata: Dict[str, Any] = None) -> Goal:
        """Add a new goal.
        
        Args:
            goal_id: Unique identifier for the goal
            description: Description of the goal
            priority: Priority level of the goal
            parent_goal_id: ID of the parent goal, if any
            metadata: Additional metadata for the goal
            
        Returns:
            Goal: The newly created goal
        """
        goal = Goal(
            id=goal_id,
            description=description,
            priority=priority,
            created_at=datetime.now(),
            parent_goal_id=parent_goal_id,
            metadata=metadata or {}
        )
        
        self.goals[goal_id] = goal
        self.logger.info(f"Added goal: {description} (Priority: {priority.name})")
        
        return goal
        
    def add_task(self, task_id: str, goal_id: str, description: str,
                dependencies: List[str] = None, estimated_duration: float = 0.0,
                metadata: Dict[str, Any] = None) -> Task:
        """Add a new task for a goal.
        
        Args:
            task_id: Unique identifier for the task
            goal_id: ID of the goal this task belongs to
            description: Description of the task
            dependencies: List of task IDs that must be completed first
            estimated_duration: Estimated duration in seconds
            metadata: Additional metadata for the task
            
        Returns:
            Task: The newly created task
        """
        if goal_id not in self.goals:
            self.logger.error(f"Cannot add task: Goal {goal_id} does not exist")
            return None
            
        task = Task(
            id=task_id,
            goal_id=goal_id,
            description=description,
            created_at=datetime.now(),
            dependencies=dependencies or [],
            estimated_duration=estimated_duration,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        self.logger.info(f"Added task: {description} for goal {goal_id}")
        
        return task
        
    def complete_task(self, task_id: str) -> bool:
        """Mark a task as completed.
        
        Args:
            task_id: ID of the task to complete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if task_id not in self.tasks:
            self.logger.error(f"Cannot complete task: Task {task_id} does not exist")
            return False
            
        task = self.tasks[task_id]
        task.completed = True
        task.completed_at = datetime.now()
        
        self.logger.info(f"Completed task: {task.description}")
        
        # Check if all tasks for the goal are completed
        goal_id = task.goal_id
        goal_tasks = [t for t in self.tasks.values() if t.goal_id == goal_id]
        
        if all(t.completed for t in goal_tasks):
            self.complete_goal(goal_id)
            
        return True
        
    def complete_goal(self, goal_id: str) -> bool:
        """Mark a goal as completed.
        
        Args:
            goal_id: ID of the goal to complete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if goal_id not in self.goals:
            self.logger.error(f"Cannot complete goal: Goal {goal_id} does not exist")
            return False
            
        goal = self.goals[goal_id]
        goal.completed = True
        goal.completed_at = datetime.now()
        
        self.logger.info(f"Completed goal: {goal.description}")
        
        return True
        
    def register_strategy(self, name: str, strategy: Callable) -> None:
        """Register a decision strategy.
        
        Args:
            name: Name of the strategy
            strategy: Strategy function
        """
        self.decision_strategies[name] = strategy
        self.logger.info(f"Registered strategy: {name}")
        
    # Event handlers
    
    async def on_resource_warning(self, data: Dict[str, Any]) -> None:
        """Handle resource warning events.
        
        Args:
            data: Event data
        """
        resource = data["resource"]
        value = data["value"]
        threshold = data["threshold"]
        
        self.logger.warning(f"Resource warning: {resource} at {value}% (threshold: {threshold}%)")
        
        # If energy is low, prioritize energy conservation
        if resource == "energy" and self.agent:
            # Add a high priority goal to find energy
            goal_id = f"conserve_energy_{datetime.now().timestamp()}"
            goal = self.add_goal(
                goal_id=goal_id,
                description="Conserve energy and find charging station",
                priority=GoalPriority.HIGH
            )
            
            # Add tasks for the goal
            self.add_task(
                task_id=f"{goal_id}_task1",
                goal_id=goal_id,
                description="Reduce energy consumption",
                metadata={"strategy": "conserve_energy"}
            )
            
            self.add_task(
                task_id=f"{goal_id}_task2",
                goal_id=goal_id,
                description="Locate nearest charging station",
                dependencies=[f"{goal_id}_task1"],
                metadata={"strategy": "explore"}
            )
            
    async def on_resource_critical(self, data: Dict[str, Any]) -> None:
        """Handle resource critical events.
        
        Args:
            data: Event data
        """
        resource = data["resource"]
        value = data["value"]
        threshold = data["threshold"]
        
        self.logger.critical(f"Resource critical: {resource} at {value}% (threshold: {threshold}%)")
        
        # If energy is critically low, immediately switch to energy conservation
        if resource == "energy" and self.agent:
            # Add a critical priority goal to find energy
            goal_id = f"critical_energy_{datetime.now().timestamp()}"
            goal = self.add_goal(
                goal_id=goal_id,
                description="CRITICAL: Find charging station immediately",
                priority=GoalPriority.CRITICAL
            )
            
            # Add task for the goal
            self.add_task(
                task_id=f"{goal_id}_task1",
                goal_id=goal_id,
                description="Enter emergency power saving mode and find charging",
                metadata={"strategy": "conserve_energy"}
            )
            
            # Force selection of this goal and task
            self.current_goal_id = goal_id
            self.current_task_id = f"{goal_id}_task1"
            
    # Decision strategies
    
    async def _explore_strategy(self, goal: Goal, task: Task) -> None:
        """Strategy for exploration.
        
        Args:
            goal: The current goal
            task: The current task
        """
        if not self.agent:
            return
            
        self.logger.info(f"Executing explore strategy for task: {task.description}")
        
        # Change agent state to exploring
        await self.agent.change_state(AgentState.EXPLORING)
        
        # Simulate exploration
        # In a real implementation, this would interact with the Navigation System
        # to explore the environment
        
        # For demonstration, we'll just simulate progress
        progress = task.metadata.get("progress", 0)
        progress += random.uniform(5, 15)  # Random progress between 5-15%
        
        if progress >= 100:
            self.complete_task(task.id)
        else:
            # Update progress in task metadata
            task.metadata["progress"] = progress
            self.logger.info(f"Exploration progress: {progress:.1f}%")
            
    async def _conserve_energy_strategy(self, goal: Goal, task: Task) -> None:
        """Strategy for conserving energy.
        
        Args:
            goal: The current goal
            task: The current task
        """
        if not self.agent:
            return
            
        self.logger.info(f"Executing conserve energy strategy for task: {task.description}")
        
        # Change agent state to idle (lowest energy consumption)
        await self.agent.change_state(AgentState.IDLE)
        
        # Get the resource manager component
        resource_manager = self.agent.get_component("ResourceManager")
        if not resource_manager:
            self.logger.error("Resource Manager not found")
            return
            
        # Simulate finding a charging station
        # In a real implementation, this would interact with the Navigation System
        # to find and navigate to a charging station
        
        # For demonstration, we'll just simulate charging
        energy_level = resource_manager.get_resource_level("energy")
        
        if energy_level < 50:
            # Simulate charging
            resource_manager.recharge_energy(10.0)
            self.logger.info(f"Found charging station. Energy level: {resource_manager.get_resource_level('energy'):.1f}%")
            
        if resource_manager.get_resource_level("energy") >= 80:
            self.complete_task(task.id)
            
    async def _collect_data_strategy(self, goal: Goal, task: Task) -> None:
        """Strategy for collecting data.
        
        Args:
            goal: The current goal
            task: The current task
        """
        if not self.agent:
            return
            
        self.logger.info(f"Executing collect data strategy for task: {task.description}")
        
        # Change agent state to collecting data
        await self.agent.change_state(AgentState.COLLECTING_DATA)
        
        # Get the knowledge base component
        kb = self.agent.get_component("KnowledgeBase")
        if not kb:
            self.logger.error("Knowledge Base not found")
            return
            
        # Simulate data collection
        # In a real implementation, this would interact with the Perception Module
        # to collect data from the environment
        
        # For demonstration, we'll just simulate collecting data about a location
        location_name = task.metadata.get("location", "Unknown Location")
        
        # Add a new location to the knowledge base
        location_id = kb.add_location(
            name=location_name,
            latitude=random.uniform(-90, 90),
            longitude=random.uniform(-180, 180),
            description=f"Discovered during exploration on {datetime.now().isoformat()}"
        )
        
        # Add some points of interest
        kb.add_point_of_interest(
            name=f"Landmark in {location_name}",
            location_id=location_id,
            poi_type="landmark",
            description="Discovered landmark",
            importance=random.randint(1, 10)
        )
        
        # Add some facts
        kb.add_fact(
            subject=location_name,
            predicate="temperature",
            obj=f"{random.uniform(0, 30):.1f}°C",
            confidence=0.9,
            source="Environmental sensors"
        )
        
        # Mark the location as visited
        kb.mark_location_visited(location_id)
        
        # Complete the task
        self.complete_task(task.id)
        
    async def _process_data_strategy(self, goal: Goal, task: Task) -> None:
        """Strategy for processing collected data.
        
        Args:
            goal: The current goal
            task: The current task
        """
        if not self.agent:
            return
            
        self.logger.info(f"Executing process data strategy for task: {task.description}")
        
        # Change agent state to processing
        await self.agent.change_state(AgentState.PROCESSING)
        
        # Get the knowledge base component
        kb = self.agent.get_component("KnowledgeBase")
        if not kb:
            self.logger.error("Knowledge Base not found")
            return
            
        # Simulate data processing
        # In a real implementation, this would perform analysis on collected data
        
        # For demonstration, we'll just simulate processing by retrieving and analyzing data
        locations = kb.get_locations(visited=True)
        
        if locations:
            # Simulate analyzing the data
            self.logger.info(f"Processing data for {len(locations)} visited locations")
            
            # Add some derived facts based on "analysis"
            for location in locations:
                kb.add_fact(
                    subject=location["name"],
                    predicate="safety_rating",
                    obj=f"{random.uniform(1, 10):.1f}/10",
                    confidence=0.8,
                    source="Data analysis"
                )
                
        # Complete the task
        self.complete_task(task.id)


# Example usage
if __name__ == "__main__":
    import asyncio
    from agent_core import TravelingAgent
    from resource_manager import ResourceManager
    from knowledge_base import KnowledgeBase
    
    async def main():
        # Create the agent
        agent = TravelingAgent("DecisionTest")
        
        # Create and register components
        resource_manager = ResourceManager()
        kb = KnowledgeBase(":memory:")  # In-memory database for testing
        decision_engine = DecisionEngine()
        
        agent.register_component(resource_manager)
        agent.register_component(kb)
        agent.register_component(decision_engine)
        
        # Initialize the agent
        await agent.initialize()
        
        # Add some goals and tasks
        decision_engine.add_goal(
            goal_id="explore_area",
            description="Explore the surrounding area",
            priority=GoalPriority.MEDIUM
        )
        
        decision_engine.add_task(
            task_id="explore_area_task1",
            goal_id="explore_area",
            description="Scout the perimeter",
            metadata={"strategy": "explore"}
        )
        
        decision_engine.add_task(
            task_id="explore_area_task2",
            goal_id="explore_area",
            description="Collect data about the area",
            dependencies=["explore_area_task1"],
            metadata={"strategy": "collect_data", "location": "Test Area"}
        )
        
        decision_engine.add_task(
            task_id="explore_area_task3",
            goal_id="explore_area",
            description="Process collected data",
            dependencies=["explore_area_task2"],
            metadata={"strategy": "process_data"}
        )
        
        # Run the agent for a short time
        try:
            agent_task = asyncio.create_task(agent.run())
            
            # Let it run for a while
            await asyncio.sleep(30)
            
            # Trigger a resource warning
            await agent.event_bus.publish("resource_warning", {
                "resource": "energy",
                "value": 25.0,
                "threshold": 30.0,
            })
            
            # Let it run a bit more
            await asyncio.sleep(10)
            
            # Trigger a resource critical event
            await agent.event_bus.publish("resource_critical", {
                "resource": "energy",
                "value": 5.0,
                "threshold": 10.0,
            })
            
            # Let it handle the critical situation
            await asyncio.sleep(10)
            
            # Shutdown
            await agent.shutdown()
            await agent_task
            
        except Exception as e:
            print(f"Error: {e}")
            
    asyncio.run(main())
