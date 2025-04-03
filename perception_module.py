#!/usr/bin/env python3
"""
Traveling AI Agent - Perception Module

This module implements the Perception Module component responsible for
sensing the environment, collecting data, and processing sensory information.
"""

import logging
import asyncio
import random
import json
import os
import time
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw

from agent_core import Component, AgentState


class SensorType(Enum):
    """Enumeration of sensor types."""
    CAMERA = "camera"
    MICROPHONE = "microphone"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    LIGHT = "light"
    PROXIMITY = "proximity"
    ACCELEROMETER = "accelerometer"
    GPS = "gps"
    COMPASS = "compass"


@dataclass
class SensorReading:
    """Class representing a sensor reading."""
    sensor_type: SensorType
    timestamp: datetime
    value: Any
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvironmentData:
    """Class representing collected environment data."""
    location_id: Optional[int]
    timestamp: datetime
    readings: List[SensorReading] = field(default_factory=list)
    processed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class Sensor:
    """Base class for all sensors."""
    
    def __init__(self, sensor_type: SensorType, name: str):
        """Initialize a sensor.
        
        Args:
            sensor_type: The type of sensor
            name: The name of the sensor
        """
        self.sensor_type = sensor_type
        self.name = name
        self.active = False
        self.logger = logging.getLogger(f"agent.sensor.{name}")
        
    async def initialize(self) -> bool:
        """Initialize the sensor.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.logger.info(f"Initializing sensor: {self.name}")
        self.active = True
        return True
        
    async def shutdown(self) -> bool:
        """Shutdown the sensor.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        self.logger.info(f"Shutting down sensor: {self.name}")
        self.active = False
        return True
        
    async def read(self) -> SensorReading:
        """Read data from the sensor.
        
        Returns:
            SensorReading: The sensor reading
        """
        raise NotImplementedError("Sensor read method must be implemented by subclasses")


class CameraSensor(Sensor):
    """Camera sensor for capturing images."""
    
    def __init__(self, name: str = "camera"):
        """Initialize a camera sensor.
        
        Args:
            name: The name of the sensor
        """
        super().__init__(SensorType.CAMERA, name)
        self.resolution = (640, 480)
        self.image_format = "JPEG"
        self.image_quality = 85
        self.save_dir = "data/images"
        
    async def initialize(self) -> bool:
        """Initialize the camera sensor.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        await super().initialize()
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        
        return True
        
    async def read(self) -> SensorReading:
        """Read data from the camera sensor.
        
        Returns:
            SensorReading: The sensor reading with image data
        """
        if not self.active:
            self.logger.error("Cannot read from inactive sensor")
            return None
            
        # In a real implementation, this would capture an image from a camera
        # For simulation, we'll generate a simple image
        image = self._generate_simulated_image()
        
        # Save the image to a file
        timestamp = datetime.now()
        filename = f"{self.save_dir}/image_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        image.save(filename, format=self.image_format, quality=self.image_quality)
        
        # Create sensor reading
        reading = SensorReading(
            sensor_type=self.sensor_type,
            timestamp=timestamp,
            value=filename,
            confidence=1.0,
            metadata={
                "resolution": self.resolution,
                "format": self.image_format,
                "quality": self.image_quality,
            }
        )
        
        self.logger.info(f"Captured image: {filename}")
        
        return reading
        
    def _generate_simulated_image(self) -> Image.Image:
        """Generate a simulated image for testing.
        
        Returns:
            Image.Image: The generated image
        """
        # Create a blank image
        image = Image.new('RGB', self.resolution, color=(240, 240, 240))
        draw = ImageDraw.Draw(image)
        
        # Draw some random shapes
        for _ in range(10):
            shape_type = random.choice(['rectangle', 'ellipse'])
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            x1 = random.randint(0, self.resolution[0] - 100)
            y1 = random.randint(0, self.resolution[1] - 100)
            x2 = x1 + random.randint(50, 100)
            y2 = y1 + random.randint(50, 100)
            
            if shape_type == 'rectangle':
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            else:
                draw.ellipse([x1, y1, x2, y2], outline=color, width=2)
                
        # Add a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        draw.text((10, 10), f"Simulated Image - {timestamp}", fill=(0, 0, 0))
        
        return image


class EnvironmentalSensor(Sensor):
    """Sensor for environmental measurements (temperature, humidity, etc.)."""
    
    def __init__(self, sensor_type: SensorType, name: str):
        """Initialize an environmental sensor.
        
        Args:
            sensor_type: The type of sensor
            name: The name of the sensor
        """
        super().__init__(sensor_type, name)
        self.min_value = 0.0
        self.max_value = 100.0
        self.units = ""
        self.precision = 2
        
        # Set ranges and units based on sensor type
        if sensor_type == SensorType.TEMPERATURE:
            self.min_value = -20.0
            self.max_value = 50.0
            self.units = "°C"
        elif sensor_type == SensorType.HUMIDITY:
            self.min_value = 0.0
            self.max_value = 100.0
            self.units = "%"
        elif sensor_type == SensorType.PRESSURE:
            self.min_value = 950.0
            self.max_value = 1050.0
            self.units = "hPa"
        elif sensor_type == SensorType.LIGHT:
            self.min_value = 0.0
            self.max_value = 10000.0
            self.units = "lux"
            
    async def read(self) -> SensorReading:
        """Read data from the environmental sensor.
        
        Returns:
            SensorReading: The sensor reading with environmental data
        """
        if not self.active:
            self.logger.error("Cannot read from inactive sensor")
            return None
            
        # In a real implementation, this would read from an actual sensor
        # For simulation, we'll generate a random value within the range
        value = round(random.uniform(self.min_value, self.max_value), self.precision)
        
        # Add some noise to make it more realistic
        noise = random.uniform(-0.1, 0.1) * (self.max_value - self.min_value)
        value = max(self.min_value, min(self.max_value, value + noise))
        value = round(value, self.precision)
        
        # Create sensor reading
        reading = SensorReading(
            sensor_type=self.sensor_type,
            timestamp=datetime.now(),
            value=value,
            confidence=0.95,
            metadata={
                "units": self.units,
                "min_value": self.min_value,
                "max_value": self.max_value,
            }
        )
        
        self.logger.debug(f"Sensor reading: {value} {self.units}")
        
        return reading


class GPSSensor(Sensor):
    """GPS sensor for location tracking."""
    
    def __init__(self, name: str = "gps"):
        """Initialize a GPS sensor.
        
        Args:
            name: The name of the sensor
        """
        super().__init__(SensorType.GPS, name)
        self.current_latitude = 0.0
        self.current_longitude = 0.0
        self.accuracy = 5.0  # meters
        
    def set_location(self, latitude: float, longitude: float) -> None:
        """Set the current location.
        
        Args:
            latitude: The latitude coordinate
            longitude: The longitude coordinate
        """
        self.current_latitude = latitude
        self.current_longitude = longitude
        
    async def read(self) -> SensorReading:
        """Read data from the GPS sensor.
        
        Returns:
            SensorReading: The sensor reading with GPS data
        """
        if not self.active:
            self.logger.error("Cannot read from inactive sensor")
            return None
            
        # Add some noise to simulate GPS inaccuracy
        lat_noise = random.uniform(-0.0001, 0.0001)
        lon_noise = random.uniform(-0.0001, 0.0001)
        
        latitude = self.current_latitude + lat_noise
        longitude = self.current_longitude + lon_noise
        
        # Create sensor reading
        reading = SensorReading(
            sensor_type=self.sensor_type,
            timestamp=datetime.now(),
            value={"latitude": latitude, "longitude": longitude},
            confidence=0.9,
            metadata={
                "accuracy": self.accuracy,
                "satellites": random.randint(4, 12),
            }
        )
        
        self.logger.debug(f"GPS reading: ({latitude}, {longitude})")
        
        return reading


class CompassSensor(Sensor):
    """Compass sensor for direction tracking."""
    
    def __init__(self, name: str = "compass"):
        """Initialize a compass sensor.
        
        Args:
            name: The name of the sensor
        """
        super().__init__(SensorType.COMPASS, name)
        self.current_heading = 0.0  # degrees (0 = North, 90 = East, etc.)
        
    def set_heading(self, heading: float) -> None:
        """Set the current heading.
        
        Args:
            heading: The heading in degrees
        """
        self.current_heading = heading % 360.0
        
    async def read(self) -> SensorReading:
        """Read data from the compass sensor.
        
        Returns:
            SensorReading: The sensor reading with compass data
        """
        if not self.active:
            self.logger.error("Cannot read from inactive sensor")
            return None
            
        # Add some noise to simulate compass inaccuracy
        heading_noise = random.uniform(-5.0, 5.0)
        heading = (self.current_heading + heading_noise) % 360.0
        
        # Create sensor reading
        reading = SensorReading(
            sensor_type=self.sensor_type,
            timestamp=datetime.now(),
            value=heading,
            confidence=0.85,
            metadata={
                "units": "degrees",
                "reference": "magnetic north",
            }
        )
        
        self.logger.debug(f"Compass reading: {heading} degrees")
        
        return reading


class PerceptionModule(Component):
    """Perception Module component for sensing and processing environmental data."""
    
    def __init__(self):
        """Initialize the Perception Module component."""
        super().__init__("PerceptionModule")
        self.sensors: Dict[str, Sensor] = {}
        self.collected_data: List[EnvironmentData] = []
        self.data_collection_active = False
        self.collection_interval = 10.0  # seconds
        self.last_collection_time = 0
        self.max_stored_data = 1000
        self.data_dir = "data"
        
    async def initialize(self) -> bool:
        """Initialize the Perception Module.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.logger.info("Initializing Perception Module")
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
        
        # Create and register sensors
        await self._create_sensors()
        
        # Subscribe to relevant events
        if self.agent:
            self.agent.event_bus.subscribe("state_changed", self.on_state_changed)
            
        self.active = True
        self.last_collection_time = time.time()
        
        return True
        
    async def shutdown(self) -> bool:
        """Shutdown the Perception Module.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        self.logger.info("Shutting down Perception Module")
        
        # Shutdown all sensors
        for sensor in self.sensors.values():
            await sensor.shutdown()
            
        # Unsubscribe from events
        if self.agent:
            self.agent.event_bus.unsubscribe("state_changed", self.on_state_changed)
            
        self.active = False
        
        return True
        
    async def _create_sensors(self) -> None:
        """Create and initialize sensors."""
        # Camera sensor
        camera = CameraSensor()
        await camera.initialize()
        self.sensors[camera.name] = camera
        
        # Environmental sensors
        temp_sensor = EnvironmentalSensor(SensorType.TEMPERATURE, "temperature")
        await temp_sensor.initialize()
        self.sensors[temp_sensor.name] = temp_sensor
        
        humidity_sensor = EnvironmentalSensor(SensorType.HUMIDITY, "humidity")
        await humidity_sensor.initialize()
        self.sensors[humidity_sensor.name] = humidity_sensor
        
        pressure_sensor = EnvironmentalSensor(SensorType.PRESSURE, "pressure")
        await pressure_sensor.initialize()
        self.sensors[pressure_sensor.name] = pressure_sensor
        
        light_sensor = EnvironmentalSensor(SensorType.LIGHT, "light")
        await light_sensor.initialize()
        self.sensors[light_sensor.name] = light_sensor
        
        # GPS sensor
        gps = GPSSensor()
        await gps.initialize()
        self.sensors[gps.name] = gps
        
        # Compass sensor
        compass = CompassSensor()
        await compass.initialize()
        self.sensors[compass.name] = compass
        
    async def update(self) -> None:
        """Update the Perception Module state."""
        current_time = time.time()
        
        # Collect data at regular intervals if active
        if self.data_collection_active and current_time - self.last_collection_time >= self.collection_interval:
            await self.collect_data()
            self.last_collection_time = current_time
            
        # Update state
        self.state = {
            "data_collection_active": self.data_collection_active,
            "collection_interval": self.collection_interval,
            "collected_data_count": len(self.collected_data),
            "sensors": {name: sensor.active for name, sensor in self.sensors.items()},
        }
        
    async def collect_data(self, location_id: int = None) -> EnvironmentData:
        """Collect data from all active sensors.
        
        Args:
            location_id: The ID of the current location
            
        Returns:
            EnvironmentData: The collected environment data
        """
        self.logger.info("Collecting sensor data")
        
        # Create a new environment data object
        env_data = EnvironmentData(
            location_id=location_id,
            timestamp=datetime.now(),
            readings=[],
            metadata={}
        )
        
        # Update GPS location if navigation system is available
        if self.agent:
            nav_system = self.agent.get_component("NavigationSystem")
            if nav_system:
                current_location = nav_system.get_current_location()
                if current_location:
                    env_data.location_id = current_location.id
                    
                    # Update GPS sensor with current location
                    gps_sensor = self.sensors.get("gps")
                    if gps_sensor:
                        gps_sensor.set_location(
                            current_location.coordinates.latitude,
                            current_location.coordinates.longitude
                        )
                        
        # Collect readings from all active sensors
        for sensor in self.sensors.values():
            if sensor.active:
                try:
                    reading = await sensor.read()
                    if reading:
                        env_data.readings.append(reading)
                except Exception as e:
                    self.logger.error(f"Error reading from sensor {sensor.name}: {e}")
                    
        # Store the collected data
        self.collected_data.append(env_data)
        
        # Trim stored data if it exceeds the maximum
        if len(self.collected_data) > self.max_stored_data:
            self.collected_data = self.collected_data[-self.max_stored_data:]
            
        # Publish data collection event
        if self.agent:
            await self.agent.event_bus.publish("data_collected", {
                "timestamp": env_data.timestamp.isoformat(),
                "location_id": env_data.location_id,
                "reading_count": len(env_data.readings),
            })
            
        self.logger.info(f"Collected {len(env_data.readings)} sensor readings")
        
        return env_data
        
    async def process_data(self, data_index: int = -1) -> Dict[str, Any]:
        """Process collected data to extract information.
        
        Args:
            data_index: Index of the data to process (-1 for latest)
            
        Returns:
            Dict[str, Any]: The processed data results
        """
        if not self.collected_data:
            self.logger.error("No data to process")
            return {}
            
        # Get the data to process
        if data_index < 0 or data_index >= len(self.collected_data):
            data_index = len(self.collected_data) - 1
            
        env_data = self.collected_data[data_index]
        
        self.logger.info(f"Processing data from {env_data.timestamp.isoformat()}")
        
        # Process the data
        results = {}
        
        # Process environmental readings
        env_readings = {}
        for reading in env_data.readings:
            if reading.sensor_type in [SensorType.TEMPERATURE, SensorType.HUMIDITY, 
                                      SensorType.PRESSURE, SensorType.LIGHT]:
                env_readings[reading.sensor_type.value] = {
                    "value": reading.value,
                    "units": reading.metadata.get("units", ""),
                    "confidence": reading.confidence,
                }
                
        if env_readings:
            results["environmental"] = env_readings
            
        # Process GPS data
        gps_readings = [r for r in env_data.readings if r.sensor_type == SensorType.GPS]
        if gps_readings:
            results["location"] = {
                "coordinates": gps_readings[0].value,
                "accuracy": gps_readings[0].metadata.get("accuracy", 0),
                "confidence": gps_readings[0].confidence,
            }
            
        # Process compass data
        compass_readings = [r for r in env_data.readings if r.sensor_type == SensorType.COMPASS]
        if compass_readings:
            heading = compass_readings[0].value
            direction = self._heading_to_direction(heading)
            results["orientation"] = {
                "heading": heading,
                "direction": direction,
                "confidence": compass_readings[0].confidence,
            }
            
        # Process image data
        image_readings = [r for r in env_data.readings if r.sensor_type == SensorType.CAMERA]
        if image_readings:
            image_paths = [r.value for r in image_readings]
            results["images"] = image_paths
            
            # In a real implementation, this would perform image analysis
            # For simulation, we'll just pretend to analyze the images
            image_analysis = self._simulate_image_analysis(image_paths)
            results["image_analysis"] = image_analysis
            
        # Mark the data as processed
        env_data.processed = True
        
        # Save processed results
        timestamp_str = env_data.timestamp.strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.data_dir}/processed/results_{timestamp_str}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Data processing complete, results saved to {results_file}")
        
        # Publish data processing event
        if self.agent:
            await self.agent.event_bus.publish("data_processed", {
                "timestamp": env_data.timestamp.isoformat(),
                "location_id": env_data.location_id,
                "results_file": results_file,
            })
            
        return results
        
    def _heading_to_direction(self, heading: float) -> str:
        """Convert a heading in degrees to a cardinal direction.
        
        Args:
            heading: The heading in degrees
            
        Returns:
            str: The cardinal direction
        """
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        index = round(heading / 45) % 8
        return directions[index]
        
    def _simulate_image_analysis(self, image_paths: List[str]) -> Dict[str, Any]:
        """Simulate image analysis for testing.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dict[str, Any]: Simulated image analysis results
        """
        # In a real implementation, this would use computer vision techniques
        # For simulation, we'll generate random results
        
        # Possible objects that could be "detected"
        possible_objects = [
            "tree", "building", "person", "car", "road", "sign", "bench", 
            "bird", "dog", "bicycle", "mountain", "river", "lake"
        ]
        
        # Generate random detections
        detections = []
        for _ in range(random.randint(1, 5)):
            obj = random.choice(possible_objects)
            confidence = random.uniform(0.7, 0.99)
            detections.append({
                "object": obj,
                "confidence": round(confidence, 2),
                "bounding_box": [
                    random.randint(0, 500),
                    random.randint(0, 400),
                    random.randint(50, 150),
                    random.randint(50, 150)
                ]
            })
            
        # Generate random scene classification
        scene_types = ["urban", "rural", "forest", "mountain", "water", "indoor"]
        scene = random.choice(scene_types)
        scene_confidence = random.uniform(0.8, 0.99)
        
        # Generate random lighting conditions
        lighting = random.choice(["bright", "dim", "dark"])
        
        return {
            "detections": detections,
            "scene": {
                "type": scene,
                "confidence": round(scene_confidence, 2)
            },
            "conditions": {
                "lighting": lighting,
                "blur": random.uniform(0, 0.3)
            }
        }
        
    def start_data_collection(self, interval: float = None) -> bool:
        """Start collecting data at regular intervals.
        
        Args:
            interval: The collection interval in seconds
            
        Returns:
            bool: True if data collection started successfully
        """
        if interval is not None:
            self.collection_interval = max(1.0, interval)
            
        self.data_collection_active = True
        self.last_collection_time = time.time()
        
        self.logger.info(f"Started data collection with interval {self.collection_interval} seconds")
        return True
        
    def stop_data_collection(self) -> bool:
        """Stop collecting data.
        
        Returns:
            bool: True if data collection stopped successfully
        """
        self.data_collection_active = False
        self.logger.info("Stopped data collection")
        return True
        
    def get_sensor(self, name: str) -> Optional[Sensor]:
        """Get a sensor by name.
        
        Args:
            name: The name of the sensor
            
        Returns:
            Optional[Sensor]: The sensor if found, None otherwise
        """
        return self.sensors.get(name)
        
    def get_latest_data(self) -> Optional[EnvironmentData]:
        """Get the latest collected data.
        
        Returns:
            Optional[EnvironmentData]: The latest data if available, None otherwise
        """
        if not self.collected_data:
            return None
        return self.collected_data[-1]
        
    def export_data(self, start_index: int = 0, count: int = None) -> str:
        """Export collected data to JSON.
        
        Args:
            start_index: The starting index
            count: The number of data points to export
            
        Returns:
            str: The exported data as a JSON string
        """
        if not self.collected_data:
            return "{}"
            
        # Determine the range to export
        if count is None:
            end_index = len(self.collected_data)
        else:
            end_index = min(start_index + count, len(self.collected_data))
            
        data_to_export = self.collected_data[start_index:end_index]
        
        # Convert to a serializable format
        export_data = []
        for env_data in data_to_export:
            data_dict = {
                "location_id": env_data.location_id,
                "timestamp": env_data.timestamp.isoformat(),
                "processed": env_data.processed,
                "readings": []
            }
            
            for reading in env_data.readings:
                reading_dict = {
                    "sensor_type": reading.sensor_type.value,
                    "timestamp": reading.timestamp.isoformat(),
                    "value": reading.value,
                    "confidence": reading.confidence,
                    "metadata": reading.metadata
                }
                data_dict["readings"].append(reading_dict)
                
            export_data.append(data_dict)
            
        # Save to file
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = f"{self.data_dir}/export_{timestamp_str}.json"
        
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        self.logger.info(f"Exported {len(export_data)} data points to {export_file}")
        
        return export_file
        
    # Event handlers
    
    async def on_state_changed(self, data: Dict[str, Any]) -> None:
        """Handle state changed events.
        
        Args:
            data: Event data
        """
        old_state = data["old_state"]
        new_state = data["new_state"]
        
        # Start data collection when exploring or collecting data
        if new_state in [AgentState.EXPLORING, AgentState.COLLECTING_DATA]:
            self.start_data_collection()
        # Stop data collection when idle or processing
        elif new_state in [AgentState.IDLE, AgentState.PROCESSING]:
            self.stop_data_collection()


# Example usage
if __name__ == "__main__":
    import asyncio
    from agent_core import TravelingAgent
    
    async def main():
        # Create the agent
        agent = TravelingAgent("PerceptionTest")
        
        # Create and register the perception module
        perception = PerceptionModule()
        agent.register_component(perception)
        
        # Initialize the agent
        await agent.initialize()
        
        # Start data collection
        perception.start_data_collection(interval=2.0)
        
        # Run the agent for a short time
        agent_task = asyncio.create_task(agent.run())
        
        # Wait for some data to be collected
        await asyncio.sleep(6)
        
        # Process the collected data
        results = await perception.process_data()
        print(f"Processing results: {results}")
        
        # Export the data
        export_file = perception.export_data()
        print(f"Data exported to: {export_file}")
        
        # Stop data collection
        perception.stop_data_collection()
        
        # Shutdown
        await agent.shutdown()
        await agent_task
        
    asyncio.run(main())
