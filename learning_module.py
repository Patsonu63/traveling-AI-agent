#!/usr/bin/env python3
"""
Traveling AI Agent - Learning Module

This module implements the Learning Module component responsible for
analyzing collected data, identifying patterns, and adapting agent behavior.
"""

import logging
import asyncio
import random
import json
import os
import time
import pickle
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict

from agent_core import Component, AgentState


class LearningModule(Component):
    """Learning Module component for analyzing data and adapting behavior."""
    
    def __init__(self):
        """Initialize the Learning Module component."""
        super().__init__("LearningModule")
        self.models = {}
        self.learning_active = False
        self.training_interval = 60.0  # seconds
        self.last_training_time = 0
        self.min_data_points = 10
        self.model_dir = "models"
        self.learning_rate = 0.1
        
    async def initialize(self) -> bool:
        """Initialize the Learning Module.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.logger.info("Initializing Learning Module")
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Load existing models if available
        await self._load_models()
        
        # Subscribe to relevant events
        if self.agent:
            self.agent.event_bus.subscribe("data_processed", self.on_data_processed)
            
        self.active = True
        self.last_training_time = time.time()
        
        return True
        
    async def shutdown(self) -> bool:
        """Shutdown the Learning Module.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        self.logger.info("Shutting down Learning Module")
        
        # Save models
        await self._save_models()
        
        # Unsubscribe from events
        if self.agent:
            self.agent.event_bus.unsubscribe("data_processed", self.on_data_processed)
            
        self.active = False
        
        return True
        
    async def _load_models(self) -> None:
        """Load existing models from disk."""
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
        
        for model_file in model_files:
            model_name = model_file[:-4]  # Remove .pkl extension
            model_path = os.path.join(self.model_dir, model_file)
            
            try:
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                self.logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                self.logger.error(f"Error loading model {model_name}: {e}")
                
    async def _save_models(self) -> None:
        """Save models to disk."""
        for model_name, model in self.models.items():
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            
            try:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                self.logger.info(f"Saved model: {model_name}")
            except Exception as e:
                self.logger.error(f"Error saving model {model_name}: {e}")
                
    async def update(self) -> None:
        """Update the Learning Module state."""
        current_time = time.time()
        
        # Train models at regular intervals if active
        if self.learning_active and current_time - self.last_training_time >= self.training_interval:
            await self.train_models()
            self.last_training_time = current_time
            
        # Update state
        self.state = {
            "learning_active": self.learning_active,
            "training_interval": self.training_interval,
            "models": list(self.models.keys()),
        }
        
    async def train_models(self) -> Dict[str, Any]:
        """Train all models with available data.
        
        Returns:
            Dict[str, Any]: Training results
        """
        if not self.agent:
            return {}
            
        self.logger.info("Training models")
        
        # Get the perception module
        perception = self.agent.get_component("PerceptionModule")
        if not perception:
            self.logger.error("Perception Module not found")
            return {}
            
        # Get collected data
        collected_data = perception.collected_data
        if len(collected_data) < self.min_data_points:
            self.logger.info(f"Not enough data for training (have {len(collected_data)}, need {self.min_data_points})")
            return {}
            
        # Train environment model
        env_results = await self._train_environment_model(collected_data)
        
        # Train location preference model
        loc_results = await self._train_location_preference_model(collected_data)
        
        # Train resource usage model
        res_results = await self._train_resource_usage_model()
        
        # Combine results
        results = {
            "environment": env_results,
            "location_preference": loc_results,
            "resource_usage": res_results,
        }
        
        # Publish training event
        await self.agent.event_bus.publish("models_trained", {
            "timestamp": datetime.now().isoformat(),
            "models": list(results.keys()),
        })
        
        return results
        
    async def _train_environment_model(self, collected_data) -> Dict[str, Any]:
        """Train the environment model.
        
        Args:
            collected_data: The collected environment data
            
        Returns:
            Dict[str, Any]: Training results
        """
        # Extract environmental readings
        env_data = []
        for data_point in collected_data:
            env_readings = {}
            for reading in data_point.readings:
                if reading.sensor_type.value in ["temperature", "humidity", "pressure", "light"]:
                    env_readings[reading.sensor_type.value] = reading.value
                    
            if env_readings:
                env_readings["timestamp"] = data_point.timestamp.timestamp()
                env_readings["location_id"] = data_point.location_id
                env_data.append(env_readings)
                
        if not env_data:
            return {"status": "no_data"}
            
        # In a real implementation, this would use a proper machine learning model
        # For simulation, we'll create a simple statistical model
        
        # Calculate averages by location
        location_averages = defaultdict(lambda: defaultdict(list))
        for data_point in env_data:
            loc_id = data_point.get("location_id")
            if loc_id is None:
                continue
                
            for key, value in data_point.items():
                if key not in ["timestamp", "location_id"]:
                    location_averages[loc_id][key].append(value)
                    
        # Calculate mean and standard deviation
        model = {}
        for loc_id, readings in location_averages.items():
            model[loc_id] = {}
            for key, values in readings.items():
                if values:
                    model[loc_id][key] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values),
                    }
                    
        # Store the model
        self.models["environment"] = model
        
        return {
            "status": "success",
            "locations": len(model),
            "features": list(next(iter(model.values())).keys()) if model else [],
        }
        
    async def _train_location_preference_model(self, collected_data) -> Dict[str, Any]:
        """Train the location preference model.
        
        Args:
            collected_data: The collected environment data
            
        Returns:
            Dict[str, Any]: Training results
        """
        if not self.agent:
            return {"status": "no_agent"}
            
        # Get the knowledge base
        kb = self.agent.get_component("KnowledgeBase")
        if not kb:
            return {"status": "no_knowledge_base"}
            
        # Get visited locations
        locations = kb.get_locations(visited=True)
        
        if not locations:
            return {"status": "no_locations"}
            
        # In a real implementation, this would use a proper machine learning model
        # For simulation, we'll create a simple preference model based on visit count
        
        # Calculate preference scores
        preferences = {}
        total_visits = sum(loc["visit_count"] for loc in locations)
        
        if total_visits == 0:
            return {"status": "no_visits"}
            
        for location in locations:
            loc_id = location["id"]
            visit_count = location["visit_count"]
            
            # Calculate preference score (0-1)
            preference = visit_count / total_visits
            
            # Store preference
            preferences[loc_id] = {
                "preference": preference,
                "visit_count": visit_count,
                "last_visited": location.get("last_visited"),
            }
            
        # Store the model
        self.models["location_preference"] = preferences
        
        return {
            "status": "success",
            "locations": len(preferences),
        }
        
    async def _train_resource_usage_model(self) -> Dict[str, Any]:
        """Train the resource usage model.
        
        Returns:
            Dict[str, Any]: Training results
        """
        if not self.agent:
            return {"status": "no_agent"}
            
        # Get the resource manager
        resource_manager = self.agent.get_component("ResourceManager")
        if not resource_manager:
            return {"status": "no_resource_manager"}
            
        # In a real implementation, this would analyze resource usage patterns
        # For simulation, we'll create a simple model based on current state
        
        # Get current resource levels
        resources = resource_manager.resources
        
        # Create a simple model
        model = {
            "energy": {
                "current": resources.get("energy", 0),
                "consumption_rate": resources.get("energy_consumption_rate", 0),
                "estimated_remaining_time": (resources.get("energy", 0) / resources.get("energy_consumption_rate", 0.01)) 
                                           if resources.get("energy_consumption_rate", 0) > 0 else float('inf'),
            },
            "cpu": {
                "current": resources.get("cpu", 0),
                "threshold": resource_manager.thresholds.get("cpu_warning", 80),
            },
            "memory": {
                "current": resources.get("memory", 0),
                "threshold": resource_manager.thresholds.get("memory_warning", 80),
            },
        }
        
        # Store the model
        self.models["resource_usage"] = model
        
        return {
            "status": "success",
            "resources": list(model.keys()),
        }
        
    def predict_environment(self, location_id: int) -> Dict[str, Any]:
        """Predict environmental conditions at a location.
        
        Args:
            location_id: The ID of the location
            
        Returns:
            Dict[str, Any]: Predicted environmental conditions
        """
        # Check if we have an environment model
        if "environment" not in self.models:
            return {}
            
        # Check if we have data for this location
        env_model = self.models["environment"]
        if str(location_id) not in env_model:
            return {}
            
        # Get the location data
        location_data = env_model[str(location_id)]
        
        # Return predictions
        predictions = {}
        for key, stats in location_data.items():
            predictions[key] = {
                "value": stats["mean"],
                "confidence": 1.0 - (stats["std"] / (stats["max"] - stats["min"])) if stats["max"] > stats["min"] else 1.0,
                "range": [stats["mean"] - stats["std"], stats["mean"] + stats["std"]],
            }
            
        return predictions
        
    def recommend_location(self) -> Optional[int]:
        """Recommend a location to visit based on preferences.
        
        Returns:
            Optional[int]: The recommended location ID, or None if no recommendation
        """
        # Check if we have a location preference model
        if "location_preference" not in self.models:
            return None
            
        # Get the preference model
        preferences = self.models["location_preference"]
        
        if not preferences:
            return None
            
        # Find the location with the highest preference
        max_preference = -1
        recommended_location = None
        
        for loc_id, data in preferences.items():
            preference = data["preference"]
            if preference > max_preference:
                max_preference = preference
                recommended_location = int(loc_id)
                
        return recommended_location
        
    def predict_resource_depletion(self) -> Dict[str, Any]:
        """Predict when resources will be depleted.
        
        Returns:
            Dict[str, Any]: Predicted depletion times
        """
        # Check if we have a resource usage model
        if "resource_usage" not in self.models:
            return {}
            
        # Get the resource model
        resource_model = self.models["resource_usage"]
        
        # Calculate depletion predictions
        predictions = {}
        
        # Energy depletion
        if "energy" in resource_model:
            energy_data = resource_model["energy"]
            current = energy_data.get("current", 0)
            rate = energy_data.get("consumption_rate", 0)
            
            if rate > 0:
                time_to_depletion = current / rate
                predictions["energy"] = {
                    "time_to_depletion": time_to_depletion,
                    "time_to_critical": time_to_depletion * 0.8,  # 80% of time to depletion
                    "current_level": current,
                }
                
        return predictions
        
    def start_learning(self, interval: float = None) -> bool:
        """Start the learning process.
        
        Args:
            interval: The training interval in seconds
            
        Returns:
            bool: True if learning started successfully
        """
        if interval is not None:
            self.training_interval = max(10.0, interval)
            
        self.learning_active = True
        self.last_training_time = time.time()
        
        self.logger.info(f"Started learning with interval {self.training_interval} seconds")
        return True
        
    def stop_learning(self) -> bool:
        """Stop the learning process.
        
        Returns:
            bool: True if learning stopped successfully
        """
        self.learning_active = False
        self.logger.info("Stopped learning")
        return True
        
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a model by name.
        
        Args:
            model_name: The name of the model
            
        Returns:
            Optional[Any]: The model if found, None otherwise
        """
        return self.models.get(model_name)
        
    def export_models(self) -> str:
        """Export all models to a JSON file.
        
        Returns:
            str: The path to the exported file
        """
        # Convert models to a serializable format
        export_data = {}
        
        for model_name, model in self.models.items():
            # Handle numpy arrays and other non-serializable types
            if model_name == "environment":
                serializable_model = {}
                for loc_id, loc_data in model.items():
                    serializable_model[loc_id] = {}
                    for feature, stats in loc_data.items():
                        serializable_model[loc_id][feature] = {
                            "mean": float(stats["mean"]),
                            "std": float(stats["std"]),
                            "min": float(stats["min"]),
                            "max": float(stats["max"]),
                            "count": int(stats["count"]),
                        }
                export_data[model_name] = serializable_model
            else:
                # For other models, try to convert to serializable format
                try:
                    json.dumps(model)  # Test if serializable
                    export_data[model_name] = model
                except (TypeError, OverflowError):
                    self.logger.warning(f"Model {model_name} is not JSON serializable, skipping")
                    
        # Save to file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = f"{self.model_dir}/models_export_{timestamp_str}.json"
        
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        self.logger.info(f"Exported {len(export_data)} models to {export_file}")
        
        return export_file
        
    # Event handlers
    
    async def on_data_processed(self, data: Dict[str, Any]) -> None:
        """Handle data processed events.
        
        Args:
            data: Event data
        """
        # When new data is processed, we might want to update our models
        if self.learning_active:
            self.logger.debug("New data processed, will update models at next training interval")


# Example usage
if __name__ == "__main__":
    import asyncio
    from agent_core import TravelingAgent
    from perception_module import PerceptionModule
    from resource_manager import ResourceManager
    from knowledge_base import KnowledgeBase
    
    async def main():
        # Create the agent
        agent = TravelingAgent("LearningTest")
        
        # Create and register components
        perception = PerceptionModule()
        resource_manager = ResourceManager()
        kb = KnowledgeBase(":memory:")  # In-memory database for testing
        learning = LearningModule()
        
        agent.register_component(perception)
        agent.register_component(resource_manager)
        agent.register_component(kb)
        agent.register_component(learning)
        
        # Initialize the agent
        await agent.initialize()
        
        # Add some test locations
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
        perception.start_data_collection(interval=1.0)
        
        # Start learning
        learning.start_learning(interval=5.0)
        
        # Run the agent for a short time
        agent_task = asyncio.create_task(agent.run())
        
        # Wait for some data to be collected and processed
        await asyncio.sleep(10)
        
        # Get predictions
        env_prediction = learning.predict_environment(loc1)
        print(f"Environment prediction for Location 1: {env_prediction}")
        
        recommended_location = learning.recommend_location()
        print(f"Recommended location: {recommended_location}")
        
        resource_prediction = learning.predict_resource_depletion()
        print(f"Resource depletion prediction: {resource_prediction}")
        
        # Export models
        export_file = learning.export_models()
        print(f"Models exported to: {export_file}")
        
        # Stop learning and data collection
        learning.stop_learning()
        perception.stop_data_collection()
        
        # Shutdown
        await agent.shutdown()
        await agent_task
        
    asyncio.run(main())
