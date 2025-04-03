#!/usr/bin/env python3
"""
Traveling AI Agent - Core Module

This module implements the core functionality of the Traveling AI Agent,
serving as the central controller that coordinates all other components.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Dict, List, Optional, Any, Callable


class AgentState(Enum):
    """Enumeration of possible agent states."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    ACTIVE = "active"
    NAVIGATING = "navigating"
    EXPLORING = "exploring"
    COLLECTING_DATA = "collecting_data"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


class Component:
    """Base class for all agent components."""
    
    def __init__(self, name: str):
        """Initialize a component with a name.
        
        Args:
            name: The name of the component
        """
        self.name = name
        self.agent = None  # Reference to the agent will be set during registration
        self.logger = logging.getLogger(f"agent.{name}")
        self.active = False
        self.state = {}
        
    async def initialize(self) -> bool:
        """Initialize the component.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.logger.info(f"Initializing component: {self.name}")
        self.active = True
        return True
        
    async def shutdown(self) -> bool:
        """Shutdown the component.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        self.logger.info(f"Shutting down component: {self.name}")
        self.active = False
        return True
        
    async def update(self) -> None:
        """Update the component state. Called on each agent tick."""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the component.
        
        Returns:
            Dict[str, Any]: The current state
        """
        return self.state


class EventBus:
    """Event bus for communication between components."""
    
    def __init__(self):
        """Initialize the event bus."""
        self.subscribers = {}
        self.logger = logging.getLogger("agent.eventbus")
        
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to an event type.
        
        Args:
            event_type: The type of event to subscribe to
            callback: The function to call when the event occurs
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
        self.logger.debug(f"Subscribed to event: {event_type}")
        
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from an event type.
        
        Args:
            event_type: The type of event to unsubscribe from
            callback: The function to unsubscribe
        """
        if event_type in self.subscribers and callback in self.subscribers[event_type]:
            self.subscribers[event_type].remove(callback)
            self.logger.debug(f"Unsubscribed from event: {event_type}")
            
    async def publish(self, event_type: str, data: Any = None) -> None:
        """Publish an event.
        
        Args:
            event_type: The type of event to publish
            data: The data associated with the event
        """
        self.logger.debug(f"Publishing event: {event_type}")
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Error in event handler: {e}")


class TravelingAgent:
    """Main agent class that coordinates all components."""
    
    def __init__(self, name: str = "TravelingAI"):
        """Initialize the agent.
        
        Args:
            name: The name of the agent
        """
        self.name = name
        self.components: Dict[str, Component] = {}
        self.state = AgentState.INITIALIZING
        self.event_bus = EventBus()
        self.logger = self._setup_logger()
        self.running = False
        self.tick_rate = 10  # Updates per second
        self.config = {
            "tick_rate": 10,
            "log_level": logging.INFO,
        }
        
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger for the agent.
        
        Returns:
            logging.Logger: The configured logger
        """
        logger = logging.getLogger("agent")
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def register_component(self, component: Component) -> None:
        """Register a component with the agent.
        
        Args:
            component: The component to register
        """
        self.components[component.name] = component
        component.agent = self
        self.logger.info(f"Registered component: {component.name}")
        
    def get_component(self, name: str) -> Optional[Component]:
        """Get a component by name.
        
        Args:
            name: The name of the component to get
            
        Returns:
            Optional[Component]: The component if found, None otherwise
        """
        return self.components.get(name)
        
    async def initialize(self) -> bool:
        """Initialize the agent and all components.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.logger.info(f"Initializing agent: {self.name}")
        
        # Initialize all components
        for name, component in self.components.items():
            try:
                success = await component.initialize()
                if not success:
                    self.logger.error(f"Failed to initialize component: {name}")
                    self.state = AgentState.ERROR
                    return False
            except Exception as e:
                self.logger.error(f"Error initializing component {name}: {e}")
                self.state = AgentState.ERROR
                return False
                
        self.state = AgentState.IDLE
        return True
        
    async def shutdown(self) -> bool:
        """Shutdown the agent and all components.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        self.logger.info(f"Shutting down agent: {self.name}")
        self.state = AgentState.SHUTTING_DOWN
        
        # Shutdown all components
        for name, component in self.components.items():
            try:
                success = await component.shutdown()
                if not success:
                    self.logger.error(f"Failed to shutdown component: {name}")
            except Exception as e:
                self.logger.error(f"Error shutting down component {name}: {e}")
                
        self.running = False
        return True
        
    async def update(self) -> None:
        """Update the agent and all components."""
        # Update all components
        for name, component in self.components.items():
            if component.active:
                try:
                    await component.update()
                except Exception as e:
                    self.logger.error(f"Error updating component {name}: {e}")
                    
    async def run(self) -> None:
        """Run the agent's main loop."""
        self.running = True
        self.logger.info(f"Starting agent: {self.name}")
        
        # Initialize the agent
        success = await self.initialize()
        if not success:
            self.logger.error("Failed to initialize agent")
            return
            
        self.state = AgentState.ACTIVE
        
        # Main loop
        try:
            while self.running:
                start_time = time.time()
                
                # Update the agent
                await self.update()
                
                # Calculate sleep time to maintain tick rate
                elapsed = time.time() - start_time
                sleep_time = max(0, (1.0 / self.tick_rate) - elapsed)
                await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.info("Agent interrupted")
        except Exception as e:
            self.logger.error(f"Error in agent main loop: {e}")
            self.state = AgentState.ERROR
        finally:
            await self.shutdown()
            
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the agent and all components.
        
        Returns:
            Dict[str, Any]: The current state
        """
        state = {
            "agent": {
                "name": self.name,
                "state": self.state.value,
                "running": self.running,
            },
            "components": {}
        }
        
        for name, component in self.components.items():
            state["components"][name] = component.get_state()
            
        return state
        
    async def change_state(self, new_state: AgentState) -> None:
        """Change the agent's state.
        
        Args:
            new_state: The new state for the agent
        """
        old_state = self.state
        self.state = new_state
        self.logger.info(f"Agent state changed: {old_state.value} -> {new_state.value}")
        await self.event_bus.publish("state_changed", {
            "old_state": old_state,
            "new_state": new_state
        })


# Example usage
if __name__ == "__main__":
    async def main():
        # Create the agent
        agent = TravelingAgent("ExplorerBot")
        
        # Create a simple test component
        class TestComponent(Component):
            def __init__(self):
                super().__init__("TestComponent")
                self.counter = 0
                
            async def update(self):
                self.counter += 1
                self.state["counter"] = self.counter
                if self.counter % 10 == 0:
                    self.logger.info(f"Counter: {self.counter}")
        
        # Register the component
        test_component = TestComponent()
        agent.register_component(test_component)
        
        # Subscribe to state changes
        async def on_state_change(data):
            print(f"State changed: {data['old_state'].value} -> {data['new_state'].value}")
            
        agent.event_bus.subscribe("state_changed", on_state_change)
        
        # Run the agent for a short time
        try:
            agent_task = asyncio.create_task(agent.run())
            await asyncio.sleep(5)  # Run for 5 seconds
            await agent.shutdown()
            await agent_task
        except Exception as e:
            print(f"Error: {e}")
            
    asyncio.run(main())
