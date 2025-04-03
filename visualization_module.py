#!/usr/bin/env python3
"""
Traveling AI Agent - Visualization Module

This module implements the Visualization component responsible for
creating visual representations of the agent's state, environment, and data.
"""

import logging
import asyncio
import os
import time
import json
import math
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agent_core import Component, AgentState


class VisualizationModule(Component):
    """Visualization component for creating visual representations of data."""
    
    def __init__(self):
        """Initialize the Visualization Module component."""
        super().__init__("VisualizationModule")
        self.output_dir = "visualizations"
        self.map_size = (800, 600)
        self.chart_size = (10, 6)  # inches
        self.dpi = 100
        self.color_scheme = {
            "background": "#f5f5f5",
            "primary": "#3498db",
            "secondary": "#2ecc71",
            "accent": "#e74c3c",
            "text": "#2c3e50",
            "grid": "#bdc3c7",
        }
        self.last_update_time = 0
        self.update_interval = 30.0  # seconds
        
    async def initialize(self) -> bool:
        """Initialize the Visualization Module.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.logger.info("Initializing Visualization Module")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Subscribe to relevant events
        if self.agent:
            self.agent.event_bus.subscribe("data_collected", self.on_data_collected)
            self.agent.event_bus.subscribe("data_processed", self.on_data_processed)
            self.agent.event_bus.subscribe("models_trained", self.on_models_trained)
            
        self.active = True
        self.last_update_time = time.time()
        
        return True
        
    async def shutdown(self) -> bool:
        """Shutdown the Visualization Module.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        self.logger.info("Shutting down Visualization Module")
        
        # Unsubscribe from events
        if self.agent:
            self.agent.event_bus.unsubscribe("data_collected", self.on_data_collected)
            self.agent.event_bus.unsubscribe("data_processed", self.on_data_processed)
            self.agent.event_bus.unsubscribe("models_trained", self.on_models_trained)
            
        self.active = False
        
        return True
        
    async def update(self) -> None:
        """Update the Visualization Module state."""
        current_time = time.time()
        
        # Update visualizations at regular intervals
        if current_time - self.last_update_time >= self.update_interval:
            await self.update_visualizations()
            self.last_update_time = current_time
            
        # Update state
        self.state = {
            "update_interval": self.update_interval,
            "last_update": datetime.fromtimestamp(self.last_update_time).isoformat(),
        }
        
    async def update_visualizations(self) -> None:
        """Update all visualizations."""
        if not self.agent:
            return
            
        self.logger.info("Updating visualizations")
        
        # Create agent status visualization
        await self.create_agent_status_visualization()
        
        # Create map visualization
        await self.create_map_visualization()
        
        # Create resource usage visualization
        await self.create_resource_usage_visualization()
        
        # Create environmental data visualization
        await self.create_environmental_data_visualization()
        
    async def create_agent_status_visualization(self) -> str:
        """Create a visualization of the agent's current status.
        
        Returns:
            str: The path to the created visualization
        """
        if not self.agent:
            return ""
            
        self.logger.info("Creating agent status visualization")
        
        # Get agent state
        agent_state = self.agent.get_state()
        
        # Create a figure
        plt.figure(figsize=self.chart_size, dpi=self.dpi)
        plt.style.use('ggplot')
        
        # Set background color
        plt.gca().set_facecolor(self.color_scheme["background"])
        plt.gcf().set_facecolor(self.color_scheme["background"])
        
        # Create a pie chart of component states
        components = agent_state.get("components", {})
        active_components = sum(1 for comp in components.values() if comp.get("active", False))
        inactive_components = len(components) - active_components
        
        labels = ['Active', 'Inactive']
        sizes = [active_components, inactive_components]
        colors = [self.color_scheme["secondary"], self.color_scheme["accent"]]
        explode = (0.1, 0)  # explode the 1st slice (Active)
        
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Add title and subtitle
        plt.title(f"Agent Status: {agent_state['agent']['state']}", 
                 fontsize=16, color=self.color_scheme["text"])
        plt.suptitle(f"Agent: {agent_state['agent']['name']}", 
                    fontsize=12, color=self.color_scheme["text"])
        
        # Add timestamp
        plt.figtext(0.5, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                   ha="center", fontsize=8, color=self.color_scheme["text"])
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.output_dir}/agent_status_{timestamp}.png"
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Agent status visualization saved to {output_path}")
        
        return output_path
        
    async def create_map_visualization(self) -> str:
        """Create a visualization of the agent's map and current location.
        
        Returns:
            str: The path to the created visualization
        """
        if not self.agent:
            return ""
            
        self.logger.info("Creating map visualization")
        
        # Get the navigation system
        nav_system = self.agent.get_component("NavigationSystem")
        if not nav_system:
            self.logger.error("Navigation System not found")
            return ""
            
        # Get the knowledge base
        kb = self.agent.get_component("KnowledgeBase")
        if not kb:
            self.logger.error("Knowledge Base not found")
            return ""
            
        # Get locations from knowledge base
        locations = kb.get_locations()
        
        if not locations:
            self.logger.warning("No locations found for map visualization")
            return ""
            
        # Create a new image
        img = Image.new('RGB', self.map_size, color=self.color_scheme["background"])
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, use default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 12)
            title_font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            title_font = ImageFont.load_default()
            
        # Find the bounding box of all locations
        min_lat = min(loc["latitude"] for loc in locations if loc["latitude"] is not None)
        max_lat = max(loc["latitude"] for loc in locations if loc["latitude"] is not None)
        min_lon = min(loc["longitude"] for loc in locations if loc["longitude"] is not None)
        max_lon = max(loc["longitude"] for loc in locations if loc["longitude"] is not None)
        
        # Add some padding
        lat_padding = (max_lat - min_lat) * 0.1
        lon_padding = (max_lon - min_lon) * 0.1
        min_lat -= lat_padding
        max_lat += lat_padding
        min_lon -= lon_padding
        max_lon += lon_padding
        
        # Ensure aspect ratio
        lat_range = max_lat - min_lat
        lon_range = max_lon - min_lon
        aspect_ratio = self.map_size[0] / self.map_size[1]
        
        if lon_range / lat_range > aspect_ratio:
            # Width is limiting factor
            center_lat = (min_lat + max_lat) / 2
            lat_range = lon_range / aspect_ratio
            min_lat = center_lat - lat_range / 2
            max_lat = center_lat + lat_range / 2
        else:
            # Height is limiting factor
            center_lon = (min_lon + max_lon) / 2
            lon_range = lat_range * aspect_ratio
            min_lon = center_lon - lon_range / 2
            max_lon = center_lon + lon_range / 2
            
        # Draw a grid
        grid_spacing = 0.1  # degrees
        
        # Round to nearest grid spacing
        min_lat = math.floor(min_lat / grid_spacing) * grid_spacing
        max_lat = math.ceil(max_lat / grid_spacing) * grid_spacing
        min_lon = math.floor(min_lon / grid_spacing) * grid_spacing
        max_lon = math.ceil(max_lon / grid_spacing) * grid_spacing
        
        # Draw latitude grid lines
        for lat in np.arange(min_lat, max_lat + grid_spacing, grid_spacing):
            y = int(self.map_size[1] - (lat - min_lat) / (max_lat - min_lat) * self.map_size[1])
            draw.line([(0, y), (self.map_size[0], y)], fill=self.color_scheme["grid"], width=1)
            draw.text((5, y), f"{lat:.1f}°", fill=self.color_scheme["text"], font=font)
            
        # Draw longitude grid lines
        for lon in np.arange(min_lon, max_lon + grid_spacing, grid_spacing):
            x = int((lon - min_lon) / (max_lon - min_lon) * self.map_size[0])
            draw.line([(x, 0), (x, self.map_size[1])], fill=self.color_scheme["grid"], width=1)
            draw.text((x, 5), f"{lon:.1f}°", fill=self.color_scheme["text"], font=font)
            
        # Draw locations
        for location in locations:
            if location["latitude"] is None or location["longitude"] is None:
                continue
                
            # Convert coordinates to pixel positions
            x = int((location["longitude"] - min_lon) / (max_lon - min_lon) * self.map_size[0])
            y = int(self.map_size[1] - (location["latitude"] - min_lat) / (max_lat - min_lat) * self.map_size[1])
            
            # Determine color based on visited status
            color = self.color_scheme["secondary"] if location["visited"] else self.color_scheme["primary"]
            
            # Draw location marker
            marker_size = 8
            draw.ellipse([(x - marker_size, y - marker_size), (x + marker_size, y + marker_size)], 
                        fill=color, outline=self.color_scheme["text"])
            
            # Draw location name
            draw.text((x + marker_size + 2, y - marker_size), location["name"], 
                     fill=self.color_scheme["text"], font=font)
            
        # Draw current location and destination if available
        current_location_id = nav_system.current_location_id
        destination_id = nav_system.destination_id
        
        if current_location_id:
            current_location = next((loc for loc in locations if loc["id"] == current_location_id), None)
            if current_location and current_location["latitude"] is not None and current_location["longitude"] is not None:
                x = int((current_location["longitude"] - min_lon) / (max_lon - min_lon) * self.map_size[0])
                y = int(self.map_size[1] - (current_location["latitude"] - min_lat) / (max_lat - min_lat) * self.map_size[1])
                
                # Draw current location marker (larger)
                marker_size = 12
                draw.ellipse([(x - marker_size, y - marker_size), (x + marker_size, y + marker_size)], 
                            fill=self.color_scheme["accent"], outline=self.color_scheme["text"])
                draw.text((x + marker_size + 2, y - marker_size), "Current Location", 
                         fill=self.color_scheme["text"], font=font)
                
        if destination_id:
            destination = next((loc for loc in locations if loc["id"] == destination_id), None)
            if destination and destination["latitude"] is not None and destination["longitude"] is not None:
                x = int((destination["longitude"] - min_lon) / (max_lon - min_lon) * self.map_size[0])
                y = int(self.map_size[1] - (destination["latitude"] - min_lat) / (max_lat - min_lat) * self.map_size[1])
                
                # Draw destination marker (star)
                marker_size = 10
                points = []
                for i in range(5):
                    angle = math.pi / 2 + i * 2 * math.pi / 5
                    points.append((x + marker_size * math.cos(angle), y - marker_size * math.sin(angle)))
                    angle += math.pi / 5
                    points.append((x + marker_size / 2 * math.cos(angle), y - marker_size / 2 * math.sin(angle)))
                draw.polygon(points, fill=self.color_scheme["primary"], outline=self.color_scheme["text"])
                draw.text((x + marker_size + 2, y - marker_size), "Destination", 
                         fill=self.color_scheme["text"], font=font)
                
        # Draw routes
        routes = kb.get_routes()
        for route in routes:
            start_location = next((loc for loc in locations if loc["id"] == route["start_location_id"]), None)
            end_location = next((loc for loc in locations if loc["id"] == route["end_location_id"]), None)
            
            if (start_location and end_location and 
                start_location["latitude"] is not None and start_location["longitude"] is not None and
                end_location["latitude"] is not None and end_location["longitude"] is not None):
                
                x1 = int((start_location["longitude"] - min_lon) / (max_lon - min_lon) * self.map_size[0])
                y1 = int(self.map_size[1] - (start_location["latitude"] - min_lat) / (max_lat - min_lat) * self.map_size[1])
                x2 = int((end_location["longitude"] - min_lon) / (max_lon - min_lon) * self.map_size[0])
                y2 = int(self.map_size[1] - (end_location["latitude"] - min_lat) / (max_lat - min_lat) * self.map_size[1])
                
                # Draw route line
                draw.line([(x1, y1), (x2, y2)], fill=self.color_scheme["primary"], width=2)
                
                # If this is the current route, highlight it
                if nav_system.current_route_id == route["id"]:
                    draw.line([(x1, y1), (x2, y2)], fill=self.color_scheme["accent"], width=3)
                    
                    # Draw progress along the route
                    if nav_system.route_progress > 0:
                        progress = nav_system.route_progress
                        px = int(x1 + (x2 - x1) * progress)
                        py = int(y1 + (y2 - y1) * progress)
                        
                        # Draw current position marker
                        marker_size = 6
                        draw.ellipse([(px - marker_size, py - marker_size), (px + marker_size, py + marker_size)], 
                                    fill=self.color_scheme["accent"], outline=self.color_scheme["text"])
                    
        # Draw title
        title = "Agent Map"
        draw.text((10, 10), title, fill=self.color_scheme["text"], font=title_font)
        
        # Draw legend
        legend_x = 10
        legend_y = 40
        
        # Visited location
        draw.ellipse([(legend_x, legend_y), (legend_x + 16, legend_y + 16)], 
                    fill=self.color_scheme["secondary"], outline=self.color_scheme["text"])
        draw.text((legend_x + 20, legend_y), "Visited Location", fill=self.color_scheme["text"], font=font)
        
        # Unvisited location
        legend_y += 20
        draw.ellipse([(legend_x, legend_y), (legend_x + 16, legend_y + 16)], 
                    fill=self.color_scheme["primary"], outline=self.color_scheme["text"])
        draw.text((legend_x + 20, legend_y), "Unvisited Location", fill=self.color_scheme["text"], font=font)
        
        # Current location
        legend_y += 20
        draw.ellipse([(legend_x, legend_y), (legend_x + 16, legend_y + 16)], 
                    fill=self.color_scheme["accent"], outline=self.color_scheme["text"])
        draw.text((legend_x + 20, legend_y), "Current Location", fill=self.color_scheme["text"], font=font)
        
        # Route
        legend_y += 20
        draw.line([(legend_x, legend_y + 8), (legend_x + 16, legend_y + 8)], 
                 fill=self.color_scheme["primary"], width=2)
        draw.text((legend_x + 20, legend_y), "Route", fill=self.color_scheme["text"], font=font)
        
        # Current route
        legend_y += 20
        draw.line([(legend_x, legend_y + 8), (legend_x + 16, legend_y + 8)], 
                 fill=self.color_scheme["accent"], width=3)
        draw.text((legend_x + 20, legend_y), "Current Route", fill=self.color_scheme["text"], font=font)
        
        # Draw timestamp
        timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        draw.text((10, self.map_size[1] - 20), timestamp_text, fill=self.color_scheme["text"], font=font)
        
        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.output_dir}/map_{timestamp}.png"
        img.save(output_path)
        
        self.logger.info(f"Map visualization saved to {output_path}")
        
        return output_path
        
    async def create_resource_usage_visualization(self) -> str:
        """Create a visualization of the agent's resource usage.
        
        Returns:
            str: The path to the created visualization
        """
        if not self.agent:
            return ""
            
        self.logger.info("Creating resource usage visualization")
        
        # Get the resource manager
        resource_manager = self.agent.get_component("ResourceManager")
        if not resource_manager:
            self.logger.error("Resource Manager not found")
            return ""
            
        # Get current resource levels
        resources = resource_manager.resources
        
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=self.chart_size, dpi=self.dpi)
        plt.style.use('ggplot')
        
        # Set background color
        fig.set_facecolor(self.color_scheme["background"])
        for ax in axs.flat:
            ax.set_facecolor(self.color_scheme["background"])
            
        # Energy gauge
        energy = resources.get("energy", 0)
        self._create_gauge(axs[0, 0], energy, "Energy", "%", 
                          warning_threshold=resource_manager.thresholds.get("energy_warning", 30),
                          critical_threshold=resource_manager.thresholds.get("energy_critical", 10))
        
        # CPU gauge
        cpu = resources.get("cpu", 0)
        self._create_gauge(axs[0, 1], cpu, "CPU", "%", 
                          warning_threshold=resource_manager.thresholds.get("cpu_warning", 80),
                          critical_threshold=resource_manager.thresholds.get("cpu_critical", 95))
        
        # Memory gauge
        memory = resources.get("memory", 0)
        self._create_gauge(axs[1, 0], memory, "Memory", "%", 
                          warning_threshold=resource_manager.thresholds.get("memory_warning", 80),
                          critical_threshold=resource_manager.thresholds.get("memory_critical", 95))
        
        # Disk gauge
        disk = resources.get("disk", 0)
        self._create_gauge(axs[1, 1], disk, "Disk", "%", 
                          warning_threshold=resource_manager.thresholds.get("disk_warning", 80),
                          critical_threshold=resource_manager.thresholds.get("disk_critical", 95))
        
        # Add title
        fig.suptitle("Agent Resource Usage", fontsize=16, color=self.color_scheme["text"])
        
        # Add timestamp
        plt.figtext(0.5, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                   ha="center", fontsize=8, color=self.color_scheme["text"])
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.output_dir}/resource_usage_{timestamp}.png"
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Resource usage visualization saved to {output_path}")
        
        return output_path
        
    def _create_gauge(self, ax, value, title, units, warning_threshold=None, critical_threshold=None):
        """Create a gauge chart.
        
        Args:
            ax: The matplotlib axis
            value: The value to display
            title: The title of the gauge
            units: The units of the value
            warning_threshold: The warning threshold
            critical_threshold: The critical threshold
        """
        # Determine color based on thresholds
        if critical_threshold is not None and value <= critical_threshold:
            color = self.color_scheme["accent"]
        elif warning_threshold is not None and value <= warning_threshold:
            color = "#f39c12"  # Orange
        else:
            color = self.color_scheme["secondary"]
            
        # Create gauge
        ax.set_aspect('equal')
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        
        # Draw background circle
        background = plt.Circle((0, 0), 1, fill=False, linewidth=2, color=self.color_scheme["grid"])
        ax.add_artist(background)
        
        # Draw filled arc
        angle = value / 100 * 360
        arc = plt.matplotlib.patches.Wedge((0, 0), 1, 0, angle, fill=True, color=color, alpha=0.7)
        ax.add_artist(arc)
        
        # Draw threshold markers if provided
        if warning_threshold is not None:
            warning_angle = warning_threshold / 100 * 360
            warning_x = math.cos(math.radians(warning_angle))
            warning_y = math.sin(math.radians(warning_angle))
            ax.plot([0, warning_x], [0, warning_y], color="#f39c12", linewidth=2)
            
        if critical_threshold is not None:
            critical_angle = critical_threshold / 100 * 360
            critical_x = math.cos(math.radians(critical_angle))
            critical_y = math.sin(math.radians(critical_angle))
            ax.plot([0, critical_x], [0, critical_y], color=self.color_scheme["accent"], linewidth=2)
            
        # Draw value text
        ax.text(0, -0.2, f"{value:.1f}{units}", ha="center", va="center", 
               fontsize=14, color=self.color_scheme["text"])
        
        # Draw title
        ax.text(0, 0.6, title, ha="center", va="center", 
               fontsize=12, color=self.color_scheme["text"])
        
        # Remove ticks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
            
    async def create_environmental_data_visualization(self) -> str:
        """Create a visualization of environmental data collected by the agent.
        
        Returns:
            str: The path to the created visualization
        """
        if not self.agent:
            return ""
            
        self.logger.info("Creating environmental data visualization")
        
        # Get the perception module
        perception = self.agent.get_component("PerceptionModule")
        if not perception:
            self.logger.error("Perception Module not found")
            return ""
            
        # Get collected data
        collected_data = perception.collected_data
        if not collected_data:
            self.logger.warning("No environmental data found for visualization")
            return ""
            
        # Extract environmental readings
        timestamps = []
        temperature_values = []
        humidity_values = []
        pressure_values = []
        light_values = []
        
        for data_point in collected_data:
            timestamps.append(data_point.timestamp)
            
            # Find readings for each sensor type
            temp_reading = next((r for r in data_point.readings if r.sensor_type.value == "temperature"), None)
            humidity_reading = next((r for r in data_point.readings if r.sensor_type.value == "humidity"), None)
            pressure_reading = next((r for r in data_point.readings if r.sensor_type.value == "pressure"), None)
            light_reading = next((r for r in data_point.readings if r.sensor_type.value == "light"), None)
            
            temperature_values.append(temp_reading.value if temp_reading else None)
            humidity_values.append(humidity_reading.value if humidity_reading else None)
            pressure_values.append(pressure_reading.value if pressure_reading else None)
            light_values.append(light_reading.value if light_reading else None)
            
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=self.chart_size, dpi=self.dpi, sharex=True)
        plt.style.use('ggplot')
        
        # Set background color
        fig.set_facecolor(self.color_scheme["background"])
        for ax in axs.flat:
            ax.set_facecolor(self.color_scheme["background"])
            
        # Plot temperature
        self._plot_time_series(axs[0, 0], timestamps, temperature_values, "Temperature", "°C")
        
        # Plot humidity
        self._plot_time_series(axs[0, 1], timestamps, humidity_values, "Humidity", "%")
        
        # Plot pressure
        self._plot_time_series(axs[1, 0], timestamps, pressure_values, "Pressure", "hPa")
        
        # Plot light
        self._plot_time_series(axs[1, 1], timestamps, light_values, "Light", "lux")
        
        # Format x-axis
        for ax in axs.flat:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            
        # Add title
        fig.suptitle("Environmental Data", fontsize=16, color=self.color_scheme["text"])
        
        # Add timestamp
        plt.figtext(0.5, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                   ha="center", fontsize=8, color=self.color_scheme["text"])
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.autofmt_xdate()
        
        # Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.output_dir}/environmental_data_{timestamp}.png"
        plt.savefig(output_path)
        plt.close()
        
        self.logger.info(f"Environmental data visualization saved to {output_path}")
        
        return output_path
        
    def _plot_time_series(self, ax, timestamps, values, title, units):
        """Plot a time series.
        
        Args:
            ax: The matplotlib axis
            timestamps: The timestamps
            values: The values
            title: The title of the plot
            units: The units of the values
        """
        # Filter out None values
        valid_data = [(t, v) for t, v in zip(timestamps, values) if v is not None]
        if not valid_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                   fontsize=12, color=self.color_scheme["text"], transform=ax.transAxes)
            ax.set_title(title, color=self.color_scheme["text"])
            return
            
        valid_timestamps, valid_values = zip(*valid_data)
        
        # Plot the data
        ax.plot(valid_timestamps, valid_values, marker='o', linestyle='-', 
               color=self.color_scheme["primary"], markersize=4)
        
        # Set title and labels
        ax.set_title(title, color=self.color_scheme["text"])
        ax.set_ylabel(units, color=self.color_scheme["text"])
        
        # Set grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set tick colors
        ax.tick_params(axis='both', colors=self.color_scheme["text"])
        
    async def create_dashboard(self) -> str:
        """Create a comprehensive dashboard with all visualizations.
        
        Returns:
            str: The path to the created dashboard
        """
        self.logger.info("Creating comprehensive dashboard")
        
        # Create all individual visualizations
        agent_status_path = await self.create_agent_status_visualization()
        map_path = await self.create_map_visualization()
        resource_path = await self.create_resource_usage_visualization()
        env_data_path = await self.create_environmental_data_visualization()
        
        # Load all images
        try:
            agent_status_img = Image.open(agent_status_path) if agent_status_path else None
            map_img = Image.open(map_path) if map_path else None
            resource_img = Image.open(resource_path) if resource_path else None
            env_data_img = Image.open(env_data_path) if env_data_path else None
        except Exception as e:
            self.logger.error(f"Error loading images for dashboard: {e}")
            return ""
            
        # Calculate dashboard size
        width = 1600
        height = 1200
        
        # Create a new image
        dashboard = Image.new('RGB', (width, height), color=self.color_scheme["background"])
        
        # Place images on dashboard
        if agent_status_img:
            agent_status_img = agent_status_img.resize((width // 2, height // 2), Image.LANCZOS)
            dashboard.paste(agent_status_img, (0, 0))
            
        if map_img:
            map_img = map_img.resize((width // 2, height // 2), Image.LANCZOS)
            dashboard.paste(map_img, (width // 2, 0))
            
        if resource_img:
            resource_img = resource_img.resize((width // 2, height // 2), Image.LANCZOS)
            dashboard.paste(resource_img, (0, height // 2))
            
        if env_data_img:
            env_data_img = env_data_img.resize((width // 2, height // 2), Image.LANCZOS)
            dashboard.paste(env_data_img, (width // 2, height // 2))
            
        # Add title and timestamp
        draw = ImageDraw.Draw(dashboard)
        try:
            title_font = ImageFont.truetype("arial.ttf", 24)
            timestamp_font = ImageFont.truetype("arial.ttf", 16)
        except IOError:
            title_font = ImageFont.load_default()
            timestamp_font = ImageFont.load_default()
            
        title = "Traveling AI Agent Dashboard"
        timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Draw title at the top center
        title_width = draw.textlength(title, font=title_font)
        draw.text(((width - title_width) // 2, 10), title, fill=self.color_scheme["text"], font=title_font)
        
        # Draw timestamp at the bottom center
        timestamp_width = draw.textlength(timestamp_text, font=timestamp_font)
        draw.text(((width - timestamp_width) // 2, height - 30), timestamp_text, 
                 fill=self.color_scheme["text"], font=timestamp_font)
        
        # Save the dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{self.output_dir}/dashboard_{timestamp}.png"
        dashboard.save(output_path)
        
        self.logger.info(f"Dashboard saved to {output_path}")
        
        return output_path
        
    # Event handlers
    
    async def on_data_collected(self, data: Dict[str, Any]) -> None:
        """Handle data collected events.
        
        Args:
            data: Event data
        """
        # We could update visualizations immediately, but we'll stick to the regular update interval
        pass
        
    async def on_data_processed(self, data: Dict[str, Any]) -> None:
        """Handle data processed events.
        
        Args:
            data: Event data
        """
        # We could update visualizations immediately, but we'll stick to the regular update interval
        pass
        
    async def on_models_trained(self, data: Dict[str, Any]) -> None:
        """Handle models trained events.
        
        Args:
            data: Event data
        """
        # We could update visualizations immediately, but we'll stick to the regular update interval
        pass


# Example usage
if __name__ == "__main__":
    import asyncio
    from agent_core import TravelingAgent
    from perception_module import PerceptionModule
    from resource_manager import ResourceManager
    from knowledge_base import KnowledgeBase
    from navigation_system import NavigationSystem
    
    async def main():
        # Create the agent
        agent = TravelingAgent("VisualizationTest")
        
        # Create and register components
        perception = PerceptionModule()
        resource_manager = ResourceManager()
        kb = KnowledgeBase(":memory:")  # In-memory database for testing
        nav = NavigationSystem()
        visualization = VisualizationModule()
        
        agent.register_component(perception)
        agent.register_component(resource_manager)
        agent.register_component(kb)
        agent.register_component(nav)
        agent.register_component(visualization)
        
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
        
        # Add a route
        kb.add_route(
            start_location_id=loc1,
            end_location_id=loc2,
            distance=5.0,
            estimated_time=60.0,
            difficulty=3
        )
        
        # Set current location
        nav.set_current_location(loc1)
        
        # Start data collection
        perception.start_data_collection(interval=1.0)
        
        # Run the agent for a short time
        agent_task = asyncio.create_task(agent.run())
        
        # Wait for some data to be collected
        await asyncio.sleep(5)
        
        # Create visualizations
        agent_status_path = await visualization.create_agent_status_visualization()
        print(f"Agent status visualization: {agent_status_path}")
        
        map_path = await visualization.create_map_visualization()
        print(f"Map visualization: {map_path}")
        
        resource_path = await visualization.create_resource_usage_visualization()
        print(f"Resource usage visualization: {resource_path}")
        
        env_data_path = await visualization.create_environmental_data_visualization()
        print(f"Environmental data visualization: {env_data_path}")
        
        dashboard_path = await visualization.create_dashboard()
        print(f"Dashboard: {dashboard_path}")
        
        # Stop data collection
        perception.stop_data_collection()
        
        # Shutdown
        await agent.shutdown()
        await agent_task
        
    asyncio.run(main())
