#!/usr/bin/env python3
"""
Traveling AI Agent - Resource Manager Module

This module implements the Resource Manager component responsible for
monitoring and managing the agent's resources such as energy, memory,
and computational resources.
"""

import logging
import time
import psutil
from typing import Dict, Any, Optional

from agent_core import Component


class ResourceManager(Component):
    """Resource Manager component for monitoring and managing agent resources."""
    
    def __init__(self):
        """Initialize the Resource Manager component."""
        super().__init__("ResourceManager")
        self.resources = {
            "cpu": 0.0,
            "memory": 0.0,
            "disk": 0.0,
            "energy": 100.0,  # Simulated energy level (percentage)
            "energy_consumption_rate": 0.01,  # Energy consumed per tick
        }
        self.thresholds = {
            "cpu_warning": 80.0,
            "cpu_critical": 95.0,
            "memory_warning": 80.0,
            "memory_critical": 95.0,
            "disk_warning": 80.0,
            "disk_critical": 95.0,
            "energy_warning": 30.0,
            "energy_critical": 10.0,
        }
        self.last_update_time = 0
        
    async def initialize(self) -> bool:
        """Initialize the Resource Manager.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.logger.info("Initializing Resource Manager")
        self.last_update_time = time.time()
        self.active = True
        
        # Subscribe to relevant events
        if self.agent:
            self.agent.event_bus.subscribe("state_changed", self.on_agent_state_changed)
            
        # Initial resource check
        self._update_resource_metrics()
        
        return True
        
    async def shutdown(self) -> bool:
        """Shutdown the Resource Manager.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        self.logger.info("Shutting down Resource Manager")
        self.active = False
        
        # Unsubscribe from events
        if self.agent:
            self.agent.event_bus.unsubscribe("state_changed", self.on_agent_state_changed)
            
        return True
        
    async def update(self) -> None:
        """Update the Resource Manager state."""
        current_time = time.time()
        
        # Update resource metrics every second
        if current_time - self.last_update_time >= 1.0:
            self._update_resource_metrics()
            self.last_update_time = current_time
            
            # Check for resource warnings
            await self._check_resource_warnings()
            
    def _update_resource_metrics(self) -> None:
        """Update the resource metrics."""
        try:
            # Update CPU usage
            self.resources["cpu"] = psutil.cpu_percent(interval=None)
            
            # Update memory usage
            memory = psutil.virtual_memory()
            self.resources["memory"] = memory.percent
            
            # Update disk usage
            disk = psutil.disk_usage('/')
            self.resources["disk"] = disk.percent
            
            # Update simulated energy level
            self.resources["energy"] -= self.resources["energy_consumption_rate"]
            self.resources["energy"] = max(0.0, self.resources["energy"])
            
            # Update state
            self.state = {
                "resources": self.resources.copy(),
                "thresholds": self.thresholds.copy(),
            }
            
        except Exception as e:
            self.logger.error(f"Error updating resource metrics: {e}")
            
    async def _check_resource_warnings(self) -> None:
        """Check for resource warnings and publish events if necessary."""
        if not self.agent:
            return
            
        # Check CPU usage
        if self.resources["cpu"] >= self.thresholds["cpu_critical"]:
            await self.agent.event_bus.publish("resource_critical", {
                "resource": "cpu",
                "value": self.resources["cpu"],
                "threshold": self.thresholds["cpu_critical"],
            })
        elif self.resources["cpu"] >= self.thresholds["cpu_warning"]:
            await self.agent.event_bus.publish("resource_warning", {
                "resource": "cpu",
                "value": self.resources["cpu"],
                "threshold": self.thresholds["cpu_warning"],
            })
            
        # Check memory usage
        if self.resources["memory"] >= self.thresholds["memory_critical"]:
            await self.agent.event_bus.publish("resource_critical", {
                "resource": "memory",
                "value": self.resources["memory"],
                "threshold": self.thresholds["memory_critical"],
            })
        elif self.resources["memory"] >= self.thresholds["memory_warning"]:
            await self.agent.event_bus.publish("resource_warning", {
                "resource": "memory",
                "value": self.resources["memory"],
                "threshold": self.thresholds["memory_warning"],
            })
            
        # Check disk usage
        if self.resources["disk"] >= self.thresholds["disk_critical"]:
            await self.agent.event_bus.publish("resource_critical", {
                "resource": "disk",
                "value": self.resources["disk"],
                "threshold": self.thresholds["disk_critical"],
            })
        elif self.resources["disk"] >= self.thresholds["disk_warning"]:
            await self.agent.event_bus.publish("resource_warning", {
                "resource": "disk",
                "value": self.resources["disk"],
                "threshold": self.thresholds["disk_warning"],
            })
            
        # Check energy level
        if self.resources["energy"] <= self.thresholds["energy_critical"]:
            await self.agent.event_bus.publish("resource_critical", {
                "resource": "energy",
                "value": self.resources["energy"],
                "threshold": self.thresholds["energy_critical"],
            })
        elif self.resources["energy"] <= self.thresholds["energy_warning"]:
            await self.agent.event_bus.publish("resource_warning", {
                "resource": "energy",
                "value": self.resources["energy"],
                "threshold": self.thresholds["energy_warning"],
            })
            
    async def on_agent_state_changed(self, data: Dict[str, Any]) -> None:
        """Handle agent state changes.
        
        Args:
            data: The state change data
        """
        # Adjust energy consumption rate based on agent state
        old_state = data["old_state"]
        new_state = data["new_state"]
        
        # Different states consume different amounts of energy
        energy_rates = {
            "idle": 0.005,
            "active": 0.01,
            "navigating": 0.02,
            "exploring": 0.03,
            "collecting_data": 0.025,
            "processing": 0.015,
        }
        
        if new_state.value in energy_rates:
            self.resources["energy_consumption_rate"] = energy_rates[new_state.value]
            self.logger.debug(f"Adjusted energy consumption rate to {self.resources['energy_consumption_rate']} for state {new_state.value}")
            
    def get_resource_level(self, resource: str) -> Optional[float]:
        """Get the current level of a specific resource.
        
        Args:
            resource: The resource to get the level for
            
        Returns:
            Optional[float]: The resource level if available, None otherwise
        """
        return self.resources.get(resource)
        
    def set_threshold(self, resource: str, warning: float, critical: float) -> bool:
        """Set the warning and critical thresholds for a resource.
        
        Args:
            resource: The resource to set thresholds for
            warning: The warning threshold
            critical: The critical threshold
            
        Returns:
            bool: True if thresholds were set, False otherwise
        """
        warning_key = f"{resource}_warning"
        critical_key = f"{resource}_critical"
        
        if warning_key in self.thresholds and critical_key in self.thresholds:
            self.thresholds[warning_key] = warning
            self.thresholds[critical_key] = critical
            return True
        return False
        
    def recharge_energy(self, amount: float) -> None:
        """Recharge the agent's energy.
        
        Args:
            amount: The amount of energy to add
        """
        self.resources["energy"] += amount
        self.resources["energy"] = min(100.0, self.resources["energy"])
        self.logger.info(f"Energy recharged by {amount}. New level: {self.resources['energy']}%")


# Example usage
if __name__ == "__main__":
    import asyncio
    from agent_core import TravelingAgent, AgentState
    
    async def main():
        # Create the agent
        agent = TravelingAgent("ResourceTest")
        
        # Create and register the resource manager
        resource_manager = ResourceManager()
        agent.register_component(resource_manager)
        
        # Subscribe to resource warnings
        async def on_resource_warning(data):
            print(f"WARNING: {data['resource']} usage at {data['value']}% (threshold: {data['threshold']}%)")
            
        async def on_resource_critical(data):
            print(f"CRITICAL: {data['resource']} usage at {data['value']}% (threshold: {data['threshold']}%)")
            
        agent.event_bus.subscribe("resource_warning", on_resource_warning)
        agent.event_bus.subscribe("resource_critical", on_resource_critical)
        
        # Run the agent
        try:
            agent_task = asyncio.create_task(agent.run())
            
            # Simulate different agent states
            await asyncio.sleep(2)
            await agent.change_state(AgentState.NAVIGATING)
            await asyncio.sleep(2)
            await agent.change_state(AgentState.EXPLORING)
            await asyncio.sleep(2)
            await agent.change_state(AgentState.IDLE)
            
            # Recharge energy
            await asyncio.sleep(2)
            resource_manager.recharge_energy(50.0)
            
            await asyncio.sleep(2)
            await agent.shutdown()
            await agent_task
            
        except Exception as e:
            print(f"Error: {e}")
            
    asyncio.run(main())
