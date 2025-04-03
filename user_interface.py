#!/usr/bin/env python3
"""
Traveling AI Agent - User Interface Module

This module implements the User Interface component responsible for
handling user interactions and displaying information to the user.
"""

import logging
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from agent_core import Component, AgentState


class UserInterface(Component):
    """User Interface component for handling user interactions."""
    
    def __init__(self):
        """Initialize the User Interface component."""
        super().__init__("UserInterface")
        self.command_handlers: Dict[str, Callable] = {}
        self.command_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.display_mode = "text"  # Can be "text" or "gui"
        
    async def initialize(self) -> bool:
        """Initialize the User Interface.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.logger.info("Initializing User Interface")
        
        # Register command handlers
        self._register_command_handlers()
        
        # Subscribe to relevant events
        if self.agent:
            self.agent.event_bus.subscribe("state_changed", self.on_state_changed)
            self.agent.event_bus.subscribe("goal_selected", self.on_goal_selected)
            self.agent.event_bus.subscribe("task_selected", self.on_task_selected)
            self.agent.event_bus.subscribe("resource_warning", self.on_resource_warning)
            self.agent.event_bus.subscribe("resource_critical", self.on_resource_critical)
            
        self.active = True
        
        # Start the command processor
        asyncio.create_task(self._process_commands())
        
        # Display welcome message
        await self.display(f"Traveling AI Agent '{self.agent.name if self.agent else 'Unknown'}' initialized.")
        await self.display("Type 'help' for a list of available commands.")
        
        return True
        
    async def shutdown(self) -> bool:
        """Shutdown the User Interface.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        self.logger.info("Shutting down User Interface")
        
        # Unsubscribe from events
        if self.agent:
            self.agent.event_bus.unsubscribe("state_changed", self.on_state_changed)
            self.agent.event_bus.unsubscribe("goal_selected", self.on_goal_selected)
            self.agent.event_bus.unsubscribe("task_selected", self.on_task_selected)
            self.agent.event_bus.unsubscribe("resource_warning", self.on_resource_warning)
            self.agent.event_bus.unsubscribe("resource_critical", self.on_resource_critical)
            
        self.active = False
        
        # Display goodbye message
        await self.display("Shutting down Traveling AI Agent. Goodbye!")
        
        return True
        
    def _register_command_handlers(self) -> None:
        """Register command handlers."""
        self.command_handlers["help"] = self._handle_help
        self.command_handlers["status"] = self._handle_status
        self.command_handlers["goals"] = self._handle_goals
        self.command_handlers["tasks"] = self._handle_tasks
        self.command_handlers["resources"] = self._handle_resources
        self.command_handlers["locations"] = self._handle_locations
        self.command_handlers["explore"] = self._handle_explore
        self.command_handlers["navigate"] = self._handle_navigate
        self.command_handlers["collect"] = self._handle_collect
        self.command_handlers["process"] = self._handle_process
        self.command_handlers["export"] = self._handle_export
        
    async def update(self) -> None:
        """Update the User Interface state."""
        # Nothing to do here for now
        pass
        
    async def handle_command(self, command: str) -> None:
        """Handle a user command.
        
        Args:
            command: The command string to handle
        """
        await self.command_queue.put(command)
        
    async def _process_commands(self) -> None:
        """Process commands from the command queue."""
        while self.active:
            try:
                # Get a command from the queue
                command = await self.command_queue.get()
                
                # Parse the command
                parts = command.strip().split()
                if not parts:
                    continue
                    
                cmd = parts[0].lower()
                args = parts[1:]
                
                # Handle the command
                if cmd in self.command_handlers:
                    handler = self.command_handlers[cmd]
                    try:
                        await handler(args)
                    except Exception as e:
                        await self.display(f"Error handling command '{cmd}': {e}")
                else:
                    await self.display(f"Unknown command: {cmd}")
                    await self.display("Type 'help' for a list of available commands.")
                    
                # Mark the command as done
                self.command_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error processing commands: {e}")
                
    async def display(self, message: str) -> None:
        """Display a message to the user.
        
        Args:
            message: The message to display
        """
        # Add timestamp to message
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Put the message in the output queue
        await self.output_queue.put(formatted_message)
        
        # In a real implementation, this would display the message in a GUI or terminal
        # For now, we'll just print it
        print(formatted_message)
        
    # Event handlers
    
    async def on_state_changed(self, data: Dict[str, Any]) -> None:
        """Handle state changed events.
        
        Args:
            data: Event data
        """
        old_state = data["old_state"]
        new_state = data["new_state"]
        
        await self.display(f"Agent state changed: {old_state.value} -> {new_state.value}")
        
    async def on_goal_selected(self, data: Dict[str, Any]) -> None:
        """Handle goal selected events.
        
        Args:
            data: Event data
        """
        goal_id = data["goal_id"]
        description = data["description"]
        priority = data["priority"]
        
        await self.display(f"New goal selected: {description} (Priority: {priority})")
        
    async def on_task_selected(self, data: Dict[str, Any]) -> None:
        """Handle task selected events.
        
        Args:
            data: Event data
        """
        task_id = data["task_id"]
        description = data["description"]
        
        await self.display(f"New task selected: {description}")
        
    async def on_resource_warning(self, data: Dict[str, Any]) -> None:
        """Handle resource warning events.
        
        Args:
            data: Event data
        """
        resource = data["resource"]
        value = data["value"]
        threshold = data["threshold"]
        
        await self.display(f"WARNING: {resource} level at {value:.1f}% (threshold: {threshold:.1f}%)")
        
    async def on_resource_critical(self, data: Dict[str, Any]) -> None:
        """Handle resource critical events.
        
        Args:
            data: Event data
        """
        resource = data["resource"]
        value = data["value"]
        threshold = data["threshold"]
        
        await self.display(f"CRITICAL: {resource} level at {value:.1f}% (threshold: {threshold:.1f}%)")
        
    # Command handlers
    
    async def _handle_help(self, args: List[str]) -> None:
        """Handle the help command.
        
        Args:
            args: Command arguments
        """
        await self.display("Available commands:")
        await self.display("  help - Display this help message")
        await self.display("  status - Display agent status")
        await self.display("  goals - Display current goals")
        await self.display("  tasks - Display current tasks")
        await self.display("  resources - Display resource levels")
        await self.display("  locations - Display known locations")
        await self.display("  explore <area> - Explore an area")
        await self.display("  navigate <location> - Navigate to a location")
        await self.display("  collect <location> - Collect data at a location")
        await self.display("  process - Process collected data")
        await self.display("  export <filename> - Export knowledge to a file")
        
    async def _handle_status(self, args: List[str]) -> None:
        """Handle the status command.
        
        Args:
            args: Command arguments
        """
        if not self.agent:
            await self.display("Agent not available")
            return
            
        state = self.agent.get_state()
        
        await self.display(f"Agent: {self.agent.name}")
        await self.display(f"State: {state['agent']['state']}")
        await self.display(f"Running: {state['agent']['running']}")
        
        # Display component status
        await self.display("Components:")
        for name, component_state in state.get("components", {}).items():
            await self.display(f"  {name}: {'Active' if component_state.get('active', False) else 'Inactive'}")
            
    async def _handle_goals(self, args: List[str]) -> None:
        """Handle the goals command.
        
        Args:
            args: Command arguments
        """
        if not self.agent:
            await self.display("Agent not available")
            return
            
        decision_engine = self.agent.get_component("DecisionEngine")
        if not decision_engine:
            await self.display("Decision Engine not available")
            return
            
        goals = decision_engine.goals
        
        if not goals:
            await self.display("No goals defined")
            return
            
        await self.display("Current goals:")
        for goal_id, goal in goals.items():
            status = "Completed" if goal.completed else "Active"
            current = " (Current)" if goal_id == decision_engine.current_goal_id else ""
            await self.display(f"  [{goal.priority.name}] {goal.description} - {status}{current}")
            
    async def _handle_tasks(self, args: List[str]) -> None:
        """Handle the tasks command.
        
        Args:
            args: Command arguments
        """
        if not self.agent:
            await self.display("Agent not available")
            return
            
        decision_engine = self.agent.get_component("DecisionEngine")
        if not decision_engine:
            await self.display("Decision Engine not available")
            return
            
        tasks = decision_engine.tasks
        
        if not tasks:
            await self.display("No tasks defined")
            return
            
        # If a goal ID is provided, filter tasks for that goal
        goal_id = args[0] if args else None
        
        if goal_id:
            filtered_tasks = {task_id: task for task_id, task in tasks.items() if task.goal_id == goal_id}
            if not filtered_tasks:
                await self.display(f"No tasks found for goal {goal_id}")
                return
            tasks_to_display = filtered_tasks
        else:
            tasks_to_display = tasks
            
        await self.display("Current tasks:")
        for task_id, task in tasks_to_display.items():
            status = "Completed" if task.completed else "Active"
            current = " (Current)" if task_id == decision_engine.current_task_id else ""
            goal_info = f"Goal: {task.goal_id}"
            await self.display(f"  {task.description} - {status}{current} ({goal_info})")
            
    async def _handle_resources(self, args: List[str]) -> None:
        """Handle the resources command.
        
        Args:
            args: Command arguments
        """
        if not self.agent:
            await self.display("Agent not available")
            return
            
        resource_manager = self.agent.get_component("ResourceManager")
        if not resource_manager:
            await self.display("Resource Manager not available")
            return
            
        resources = resource_manager.resources
        
        await self.display("Current resource levels:")
        for resource, value in resources.items():
            if resource != "energy_consumption_rate":
                await self.display(f"  {resource}: {value:.1f}%")
                
        await self.display(f"Energy consumption rate: {resources.get('energy_consumption_rate', 0):.3f} units/tick")
        
    async def _handle_locations(self, args: List[str]) -> None:
        """Handle the locations command.
        
        Args:
            args: Command arguments
        """
        if not self.agent:
            await self.display("Agent not available")
            return
            
        kb = self.agent.get_component("KnowledgeBase")
        if not kb:
            await self.display("Knowledge Base not available")
            return
            
        # Get locations
        locations = kb.get_locations()
        
        if not locations:
            await self.display("No locations known")
            return
            
        await self.display("Known locations:")
        for location in locations:
            visited = "Visited" if location.get("visited") else "Not visited"
            coords = f"({location.get('latitude', 'N/A')}, {location.get('longitude', 'N/A')})"
            await self.display(f"  {location.get('name')} {coords} - {visited}")
            
            # If detailed flag is provided, show points of interest
            if args and args[0] == "detailed":
                pois = kb.get_points_of_interest(location.get("id"))
                if pois:
                    await self.display(f"    Points of interest:")
                    for poi in pois:
                        await self.display(f"      {poi.get('name')} - {poi.get('type', 'N/A')}")
                        
    async def _handle_explore(self, args: List[str]) -> None:
        """Handle the explore command.
        
        Args:
            args: Command arguments
        """
        if not args:
            await self.display("Usage: explore <area>")
            return
            
        area = " ".join(args)
        
        if not self.agent:
            await self.display("Agent not available")
            return
            
        decision_engine = self.agent.get_component("DecisionEngine")
        if not decision_engine:
            await self.display("Decision Engine not available")
            return
            
        # Create a new goal to explore the area
        goal_id = f"explore_{area.replace(' ', '_').lower()}"
        goal = decision_engine.add_goal(
            goal_id=goal_id,
            description=f"Explore {area}",
            priority=decision_engine.GoalPriority.MEDIUM
        )
        
        # Add tasks for the goal
        decision_engine.add_task(
            task_id=f"{goal_id}_task1",
            goal_id=goal_id,
            description=f"Scout {area}",
            metadata={"strategy": "explore", "area": area}
        )
        
        decision_engine.add_task(
            task_id=f"{goal_id}_task2",
            goal_id=goal_id,
            description=f"Collect data about {area}",
            dependencies=[f"{goal_id}_task1"],
            metadata={"strategy": "collect_data", "location": area}
        )
        
        decision_engine.add_task(
            task_id=f"{goal_id}_task3",
            goal_id=goal_id,
            description=f"Process data collected from {area}",
            dependencies=[f"{goal_id}_task2"],
            metadata={"strategy": "process_data"}
        )
        
        await self.display(f"Created new goal to explore {area}")
        
    async def _handle_navigate(self, args: List[str]) -> None:
        """Handle the navigate command.
        
        Args:
            args: Command arguments
        """
        if not args:
            await self.display("Usage: navigate <location>")
            return
            
        location = " ".join(args)
        
        if not self.agent:
            await self.display("Agent not available")
            return
            
        # Change agent state to navigating
        await self.agent.change_state(AgentState.NAVIGATING)
        
        await self.display(f"Navigating to {location}...")
        
        # In a real implementation, this would interact with the Navigation System
        # For now, we'll just simulate navigation
        await asyncio.sleep(2)
        
        await self.display(f"Arrived at {location}")
        
        # Change agent state back to active
        await self.agent.change_state(AgentState.ACTIVE)
        
    async def _handle_collect(self, args: List[str]) -> None:
        """Handle the collect command.
        
        Args:
            args: Command arguments
        """
        if not args:
            await self.display("Usage: collect <location>")
            return
            
        location = " ".join(args)
        
        if not self.agent:
            await self.display("Agent not available")
            return
            
        kb = self.agent.get_component("KnowledgeBase")
        if not kb:
            await self.display("Knowledge Base not available")
            return
            
        # Change agent state to collecting data
        await self.agent.change_state(AgentState.COLLECTING_DATA)
        
        await self.display(f"Collecting data at {location}...")
        
        # Add a new location to the knowledge base
        location_id = kb.add_location(
            name=location,
            latitude=0.0,  # Placeholder
            longitude=0.0,  # Placeholder
            description=f"Location visited on {datetime.now().isoformat()}"
        )
        
        # Add some points of interest
        kb.add_point_of_interest(
            name=f"Main Square in {location}",
            location_id=location_id,
            poi_type="landmark",
            description="Central square",
            importance=8
        )
        
        # Add some facts
        kb.add_fact(
            subject=location,
            predicate="visited",
            obj=datetime.now().isoformat(),
            confidence=1.0,
            source="User command"
        )
        
        # Mark the location as visited
        kb.mark_location_visited(location_id)
        
        await self.display(f"Data collection at {location} complete")
        
        # Change agent state back to active
        await self.agent.change_state(AgentState.ACTIVE)
        
    async def _handle_process(self, args: List[str]) -> None:
        """Handle the process command.
        
        Args:
            args: Command arguments
        """
        if not self.agent:
            await self.display("Agent not available")
            return
            
        kb = self.agent.get_component("KnowledgeBase")
        if not kb:
            await self.display("Knowledge Base not available")
            return
            
        # Change agent state to processing
        await self.agent.change_state(AgentState.PROCESSING)
        
        await self.display("Processing collected data...")
        
        # Get locations
        locations = kb.get_locations(visited=True)
        
        if not locations:
            await self.display("No visited locations to process")
            await self.agent.change_state(AgentState.ACTIVE)
            return
            
        await self.display(f"Processing data for {len(locations)} visited locations")
        
        # Simulate processing
        await asyncio.sleep(2)
        
        # Add some derived facts
        for location in locations:
            kb.add_fact(
                subject=location["name"],
                predicate="processed",
                obj="true",
                confidence=1.0,
                source="Data processing"
            )
            
        await self.display("Data processing complete")
        
        # Change agent state back to active
        await self.agent.change_state(AgentState.ACTIVE)
        
    async def _handle_export(self, args: List[str]) -> None:
        """Handle the export command.
        
        Args:
            args: Command arguments
        """
        if not args:
            await self.display("Usage: export <filename>")
            return
            
        filename = args[0]
        
        if not self.agent:
            await self.display("Agent not available")
            return
            
        kb = self.agent.get_component("KnowledgeBase")
        if not kb:
            await self.display("Knowledge Base not available")
            return
            
        # Export knowledge
        export_data = kb.export_knowledge()
        
        if not export_data:
            await self.display("No data to export")
            return
            
        # In a real implementation, this would write to a file
        # For now, we'll just display the data
        await self.display(f"Exported knowledge to {filename}")
        await self.display(f"Export size: {len(export_data)} bytes")


# Example usage
if __name__ == "__main__":
    import asyncio
    from agent_core import TravelingAgent
    from resource_manager import ResourceManager
    from knowledge_base import KnowledgeBase
    from decision_engine import DecisionEngine
    
    async def main():
        # Create the agent
        agent = TravelingAgent("UITest")
        
        # Create and register components
        resource_manager = ResourceManager()
        kb = KnowledgeBase(":memory:")  # In-memory database for testing
        decision_engine = DecisionEngine()
        ui = UserInterface()
        
        agent.register_component(resource_manager)
        agent.register_component(kb)
        agent.register_component(decision_engine)
        agent.register_component(ui)
        
        # Initialize the agent
        await agent.initialize()
        
        # Run the agent
        agent_task = asyncio.create_task(agent.run())
        
        # Simulate user commands
        await asyncio.sleep(1)
        await ui.handle_command("help")
        
        await asyncio.sleep(1)
        await ui.handle_command("status")
        
        await asyncio.sleep(1)
        await ui.handle_command("explore New York City")
        
        await asyncio.sleep(5)
        await ui.handle_command("goals")
        
        await asyncio.sleep(1)
        await ui.handle_command("tasks")
        
        await asyncio.sleep(1)
        await ui.handle_command("collect Central Park")
        
        await asyncio.sleep(2)
        await ui.handle_command("locations")
        
        await asyncio.sleep(1)
        await ui.handle_command("process")
        
        await asyncio.sleep(2)
        await ui.handle_command("resources")
        
        # Let it run for a bit
        await asyncio.sleep(5)
        
        # Shutdown
        await agent.shutdown()
        await agent_task
        
    asyncio.run(main())
