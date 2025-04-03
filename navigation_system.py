#!/usr/bin/env python3
"""
Traveling AI Agent - Navigation System Module

This module implements the Navigation System component responsible for
path planning, route optimization, location tracking, and map representation.
"""

import logging
import asyncio
import math
import random
import heapq
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime

from agent_core import Component, AgentState


@dataclass
class Coordinates:
    """Class representing geographic coordinates."""
    latitude: float
    longitude: float
    
    def distance_to(self, other: 'Coordinates') -> float:
        """Calculate the distance to another set of coordinates using the Haversine formula.
        
        Args:
            other: The other coordinates
            
        Returns:
            float: The distance in kilometers
        """
        # Earth radius in kilometers
        R = 6371.0
        
        lat1 = math.radians(self.latitude)
        lon1 = math.radians(self.longitude)
        lat2 = math.radians(other.latitude)
        lon2 = math.radians(other.longitude)
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return distance


@dataclass
class Location:
    """Class representing a location on the map."""
    id: int
    name: str
    coordinates: Coordinates
    description: Optional[str] = None
    visited: bool = False
    visit_count: int = 0
    last_visited: Optional[datetime] = None
    
    def distance_to(self, other: 'Location') -> float:
        """Calculate the distance to another location.
        
        Args:
            other: The other location
            
        Returns:
            float: The distance in kilometers
        """
        return self.coordinates.distance_to(other.coordinates)


@dataclass
class Route:
    """Class representing a route between two locations."""
    id: int
    start_location: Location
    end_location: Location
    distance: float
    estimated_time: float  # in minutes
    difficulty: int  # 0-10
    path: List[Coordinates] = None  # Waypoints along the route
    
    def __post_init__(self):
        """Initialize the route with default values if needed."""
        if self.path is None:
            self.path = [self.start_location.coordinates, self.end_location.coordinates]
        if self.distance <= 0:
            self.distance = self.start_location.distance_to(self.end_location)
        if self.estimated_time <= 0:
            # Assume average speed of 5 km/h for walking
            self.estimated_time = (self.distance / 5) * 60


@dataclass
class Map:
    """Class representing a map with locations and routes."""
    locations: Dict[int, Location]
    routes: Dict[int, Route]
    
    def add_location(self, location: Location) -> None:
        """Add a location to the map.
        
        Args:
            location: The location to add
        """
        self.locations[location.id] = location
        
    def add_route(self, route: Route) -> None:
        """Add a route to the map.
        
        Args:
            route: The route to add
        """
        self.routes[route.id] = route
        
    def get_location(self, location_id: int) -> Optional[Location]:
        """Get a location by ID.
        
        Args:
            location_id: The ID of the location to get
            
        Returns:
            Optional[Location]: The location if found, None otherwise
        """
        return self.locations.get(location_id)
        
    def get_route(self, route_id: int) -> Optional[Route]:
        """Get a route by ID.
        
        Args:
            route_id: The ID of the route to get
            
        Returns:
            Optional[Route]: The route if found, None otherwise
        """
        return self.routes.get(route_id)
        
    def find_route(self, start_location_id: int, end_location_id: int) -> Optional[Route]:
        """Find a route between two locations.
        
        Args:
            start_location_id: The ID of the starting location
            end_location_id: The ID of the ending location
            
        Returns:
            Optional[Route]: The route if found, None otherwise
        """
        for route in self.routes.values():
            if (route.start_location.id == start_location_id and 
                route.end_location.id == end_location_id):
                return route
        return None
        
    def find_nearest_location(self, coordinates: Coordinates) -> Optional[Location]:
        """Find the nearest location to a set of coordinates.
        
        Args:
            coordinates: The coordinates to find the nearest location to
            
        Returns:
            Optional[Location]: The nearest location if any exist, None otherwise
        """
        if not self.locations:
            return None
            
        nearest_location = None
        min_distance = float('inf')
        
        for location in self.locations.values():
            distance = coordinates.distance_to(location.coordinates)
            if distance < min_distance:
                min_distance = distance
                nearest_location = location
                
        return nearest_location


class NavigationSystem(Component):
    """Navigation System component for path planning and location tracking."""
    
    def __init__(self):
        """Initialize the Navigation System component."""
        super().__init__("NavigationSystem")
        self.map = Map({}, {})
        self.current_location_id = None
        self.destination_id = None
        self.current_route_id = None
        self.route_progress = 0.0  # 0.0 to 1.0
        self.navigation_active = False
        self.exploration_radius = 5.0  # km
        self.last_update_time = 0
        
    async def initialize(self) -> bool:
        """Initialize the Navigation System.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.logger.info("Initializing Navigation System")
        
        # Subscribe to relevant events
        if self.agent:
            self.agent.event_bus.subscribe("state_changed", self.on_state_changed)
            
        self.active = True
        self.last_update_time = datetime.now().timestamp()
        
        return True
        
    async def shutdown(self) -> bool:
        """Shutdown the Navigation System.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        self.logger.info("Shutting down Navigation System")
        
        # Unsubscribe from events
        if self.agent:
            self.agent.event_bus.unsubscribe("state_changed", self.on_state_changed)
            
        self.active = False
        
        return True
        
    async def update(self) -> None:
        """Update the Navigation System state."""
        current_time = datetime.now().timestamp()
        
        # Update navigation progress if active
        if self.navigation_active and self.current_route_id is not None:
            route = self.map.get_route(self.current_route_id)
            if route:
                # Simulate progress along the route
                progress_increment = (current_time - self.last_update_time) / (route.estimated_time * 60)
                self.route_progress += progress_increment
                
                # Check if we've reached the destination
                if self.route_progress >= 1.0:
                    await self._arrive_at_destination()
                    
        self.last_update_time = current_time
        
        # Update state
        self.state = {
            "current_location": self.current_location_id,
            "destination": self.destination_id,
            "current_route": self.current_route_id,
            "route_progress": self.route_progress,
            "navigation_active": self.navigation_active,
        }
        
    async def _arrive_at_destination(self) -> None:
        """Handle arrival at the destination."""
        if not self.destination_id or not self.agent:
            return
            
        destination = self.map.get_location(self.destination_id)
        if not destination:
            return
            
        self.logger.info(f"Arrived at destination: {destination.name}")
        
        # Update location
        self.current_location_id = self.destination_id
        self.destination_id = None
        self.current_route_id = None
        self.route_progress = 0.0
        self.navigation_active = False
        
        # Mark location as visited
        destination.visited = True
        destination.visit_count += 1
        destination.last_visited = datetime.now()
        
        # Update knowledge base
        kb = self.agent.get_component("KnowledgeBase")
        if kb:
            kb.mark_location_visited(destination.id)
            
        # Publish arrival event
        await self.agent.event_bus.publish("destination_reached", {
            "location_id": destination.id,
            "location_name": destination.name,
        })
        
        # Change agent state
        await self.agent.change_state(AgentState.ACTIVE)
        
    async def navigate_to(self, destination_id: int) -> bool:
        """Navigate to a destination.
        
        Args:
            destination_id: The ID of the destination location
            
        Returns:
            bool: True if navigation started successfully, False otherwise
        """
        if not self.agent:
            return False
            
        # Check if destination exists
        destination = self.map.get_location(destination_id)
        if not destination:
            self.logger.error(f"Destination location {destination_id} not found")
            return False
            
        # Check if we're already at the destination
        if self.current_location_id == destination_id:
            self.logger.info(f"Already at destination: {destination.name}")
            return True
            
        # Find a route to the destination
        route = None
        
        if self.current_location_id:
            route = self.map.find_route(self.current_location_id, destination_id)
            
        if not route:
            # Create a new route
            if self.current_location_id:
                start_location = self.map.get_location(self.current_location_id)
                if start_location:
                    route = self._plan_route(start_location, destination)
                    
        if not route:
            self.logger.error(f"Could not find or create a route to {destination.name}")
            return False
            
        # Start navigation
        self.destination_id = destination_id
        self.current_route_id = route.id
        self.route_progress = 0.0
        self.navigation_active = True
        
        # Change agent state
        await self.agent.change_state(AgentState.NAVIGATING)
        
        self.logger.info(f"Navigation started to {destination.name}")
        self.logger.info(f"Estimated time: {route.estimated_time:.1f} minutes")
        
        # Publish navigation started event
        await self.agent.event_bus.publish("navigation_started", {
            "destination_id": destination_id,
            "destination_name": destination.name,
            "route_id": route.id,
            "estimated_time": route.estimated_time,
        })
        
        return True
        
    def _plan_route(self, start_location: Location, end_location: Location) -> Route:
        """Plan a route between two locations.
        
        Args:
            start_location: The starting location
            end_location: The ending location
            
        Returns:
            Route: The planned route
        """
        # Calculate a direct route
        distance = start_location.distance_to(end_location)
        
        # Estimate time based on distance (assume 5 km/h walking speed)
        estimated_time = (distance / 5) * 60  # minutes
        
        # Assign a difficulty based on distance
        difficulty = min(10, int(distance / 5))
        
        # Generate a new route ID
        route_id = len(self.map.routes) + 1
        
        # Create the route
        route = Route(
            id=route_id,
            start_location=start_location,
            end_location=end_location,
            distance=distance,
            estimated_time=estimated_time,
            difficulty=difficulty
        )
        
        # Add to map
        self.map.add_route(route)
        
        return route
        
    async def explore_area(self, center_location_id: int = None, radius: float = None) -> bool:
        """Explore an area around a location.
        
        Args:
            center_location_id: The ID of the center location (defaults to current location)
            radius: The radius to explore in kilometers (defaults to exploration_radius)
            
        Returns:
            bool: True if exploration started successfully, False otherwise
        """
        if not self.agent:
            return False
            
        # Use current location if not specified
        if center_location_id is None:
            center_location_id = self.current_location_id
            
        # Check if center location exists
        center_location = None
        if center_location_id:
            center_location = self.map.get_location(center_location_id)
            
        if not center_location:
            self.logger.error("No center location for exploration")
            return False
            
        # Use default radius if not specified
        if radius is None:
            radius = self.exploration_radius
            
        # Change agent state
        await self.agent.change_state(AgentState.EXPLORING)
        
        self.logger.info(f"Starting exploration around {center_location.name} with radius {radius} km")
        
        # Publish exploration started event
        await self.agent.event_bus.publish("exploration_started", {
            "center_location_id": center_location_id,
            "center_location_name": center_location.name,
            "radius": radius,
        })
        
        # Simulate discovering new locations
        await self._discover_locations(center_location, radius)
        
        return True
        
    async def _discover_locations(self, center_location: Location, radius: float) -> None:
        """Discover new locations around a center location.
        
        Args:
            center_location: The center location
            radius: The radius to explore in kilometers
        """
        # Get the knowledge base component
        kb = self.agent.get_component("KnowledgeBase")
        if not kb:
            self.logger.error("Knowledge Base not found")
            return
            
        # Simulate discovering 1-3 new locations
        num_locations = random.randint(1, 3)
        
        for i in range(num_locations):
            # Generate random coordinates within the radius
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, radius)
            
            # Convert polar to Cartesian coordinates
            dx = distance * math.cos(angle)
            dy = distance * math.sin(angle)
            
            # Convert to latitude/longitude (approximate)
            # 1 degree of latitude is approximately 111 km
            # 1 degree of longitude varies with latitude
            lat_offset = dy / 111.0
            lon_offset = dx / (111.0 * math.cos(math.radians(center_location.coordinates.latitude)))
            
            new_lat = center_location.coordinates.latitude + lat_offset
            new_lon = center_location.coordinates.longitude + lon_offset
            
            # Generate a name for the location
            location_name = f"Discovered Location {len(self.map.locations) + 1}"
            
            # Add to knowledge base
            location_id = kb.add_location(
                name=location_name,
                latitude=new_lat,
                longitude=new_lon,
                description=f"Discovered during exploration on {datetime.now().isoformat()}"
            )
            
            # Add to map
            new_location = Location(
                id=location_id,
                name=location_name,
                coordinates=Coordinates(new_lat, new_lon),
                description=f"Discovered during exploration on {datetime.now().isoformat()}"
            )
            self.map.add_location(new_location)
            
            # Create a route to the new location
            self._plan_route(center_location, new_location)
            
            self.logger.info(f"Discovered new location: {location_name} at ({new_lat:.6f}, {new_lon:.6f})")
            
            # Add some points of interest
            num_pois = random.randint(0, 2)
            for j in range(num_pois):
                poi_name = f"Point of Interest {j+1} at {location_name}"
                poi_types = ["landmark", "building", "natural", "historical", "viewpoint"]
                poi_type = random.choice(poi_types)
                
                kb.add_point_of_interest(
                    name=poi_name,
                    location_id=location_id,
                    poi_type=poi_type,
                    description=f"Discovered during exploration",
                    importance=random.randint(1, 10)
                )
                
                self.logger.info(f"Discovered point of interest: {poi_name} ({poi_type})")
                
        # Publish discovery event
        await self.agent.event_bus.publish("locations_discovered", {
            "count": num_locations,
            "center_location_name": center_location.name,
            "radius": radius,
        })
        
    def find_path(self, start_location_id: int, end_location_id: int) -> List[int]:
        """Find the shortest path between two locations using A* algorithm.
        
        Args:
            start_location_id: The ID of the starting location
            end_location_id: The ID of the ending location
            
        Returns:
            List[int]: List of location IDs forming the path, or empty list if no path found
        """
        if start_location_id == end_location_id:
            return [start_location_id]
            
        start_location = self.map.get_location(start_location_id)
        end_location = self.map.get_location(end_location_id)
        
        if not start_location or not end_location:
            return []
            
        # Build a graph of connected locations
        graph = {}
        for route in self.map.routes.values():
            start_id = route.start_location.id
            end_id = route.end_location.id
            
            if start_id not in graph:
                graph[start_id] = {}
            if end_id not in graph:
                graph[end_id] = {}
                
            # Add bidirectional connections
            graph[start_id][end_id] = route.distance
            graph[end_id][start_id] = route.distance
            
        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start_location_id))
        
        came_from = {}
        
        g_score = {loc_id: float('inf') for loc_id in self.map.locations}
        g_score[start_location_id] = 0
        
        f_score = {loc_id: float('inf') for loc_id in self.map.locations}
        f_score[start_location_id] = self.map.get_location(start_location_id).distance_to(end_location)
        
        open_set_hash = {start_location_id}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            if current == end_location_id:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_location_id)
                path.reverse()
                return path
                
            if current not in graph:
                continue
                
            for neighbor, distance in graph[current].items():
                tentative_g_score = g_score[current] + distance
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.map.get_location(neighbor).distance_to(end_location)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
                        
        return []  # No path found
        
    def set_current_location(self, location_id: int) -> bool:
        """Set the current location.
        
        Args:
            location_id: The ID of the location to set as current
            
        Returns:
            bool: True if successful, False otherwise
        """
        location = self.map.get_location(location_id)
        if not location:
            self.logger.error(f"Location {location_id} not found")
            return False
            
        self.current_location_id = location_id
        self.logger.info(f"Current location set to: {location.name}")
        return True
        
    def get_current_location(self) -> Optional[Location]:
        """Get the current location.
        
        Returns:
            Optional[Location]: The current location if set, None otherwise
        """
        if self.current_location_id:
            return self.map.get_location(self.current_location_id)
        return None
        
    def import_map_from_knowledge_base(self) -> bool:
        """Import map data from the knowledge base.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.agent:
            return False
            
        kb = self.agent.get_component("KnowledgeBase")
        if not kb:
            self.logger.error("Knowledge Base not found")
            return False
            
        # Get locations from knowledge base
        kb_locations = kb.get_locations()
        
        # Convert to map locations
        for kb_loc in kb_locations:
            location = Location(
                id=kb_loc["id"],
                name=kb_loc["name"],
                coordinates=Coordinates(kb_loc["latitude"], kb_loc["longitude"]),
                description=kb_loc["description"],
                visited=kb_loc["visited"] == 1,
                visit_count=kb_loc["visit_count"],
                last_visited=datetime.fromisoformat(kb_loc["last_visited"]) if kb_loc["last_visited"] else None
            )
            self.map.add_location(location)
            
        # Get routes from knowledge base
        kb_routes = kb.get_routes()
        
        # Convert to map routes
        for kb_route in kb_routes:
            start_location = self.map.get_location(kb_route["start_location_id"])
            end_location = self.map.get_location(kb_route["end_location_id"])
            
            if start_location and end_location:
                route = Route(
                    id=kb_route["id"],
                    start_location=start_location,
                    end_location=end_location,
                    distance=kb_route["distance"],
                    estimated_time=kb_route["estimated_time"],
                    difficulty=kb_route["difficulty"]
                )
                self.map.add_route(route)
                
        self.logger.info(f"Imported {len(self.map.locations)} locations and {len(self.map.routes)} routes from knowledge base")
        return True
        
    def export_map_to_knowledge_base(self) -> bool:
        """Export map data to the knowledge base.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.agent:
            return False
            
        kb = self.agent.get_component("KnowledgeBase")
        if not kb:
            self.logger.error("Knowledge Base not found")
            return False
            
        # Export locations
        for location in self.map.locations.values():
            # Check if location exists in knowledge base
            kb_location = kb.get_location(location.id)
            
            if not kb_location:
                # Add new location
                kb.add_location(
                    name=location.name,
                    latitude=location.coordinates.latitude,
                    longitude=location.coordinates.longitude,
                    description=location.description
                )
                
                if location.visited:
                    kb.mark_location_visited(location.id)
                    
        # Export routes
        for route in self.map.routes.values():
            # Add route to knowledge base
            kb.add_route(
                start_location_id=route.start_location.id,
                end_location_id=route.end_location.id,
                distance=route.distance,
                estimated_time=route.estimated_time,
                difficulty=route.difficulty
            )
            
        self.logger.info(f"Exported {len(self.map.locations)} locations and {len(self.map.routes)} routes to knowledge base")
        return True
        
    # Event handlers
    
    async def on_state_changed(self, data: Dict[str, Any]) -> None:
        """Handle state changed events.
        
        Args:
            data: Event data
        """
        old_state = data["old_state"]
        new_state = data["new_state"]
        
        # If we're no longer navigating, stop navigation
        if old_state == AgentState.NAVIGATING and new_state != AgentState.NAVIGATING:
            self.navigation_active = False


# Example usage
if __name__ == "__main__":
    import asyncio
    from agent_core import TravelingAgent
    from knowledge_base import KnowledgeBase
    
    async def main():
        # Create the agent
        agent = TravelingAgent("NavigationTest")
        
        # Create and register components
        kb = KnowledgeBase(":memory:")  # In-memory database for testing
        nav = NavigationSystem()
        
        agent.register_component(kb)
        agent.register_component(nav)
        
        # Initialize the agent
        await agent.initialize()
        
        # Create some test locations
        home = Location(
            id=1,
            name="Home Base",
            coordinates=Coordinates(40.7128, -74.0060),
            description="Starting location"
        )
        
        destination = Location(
            id=2,
            name="Destination",
            coordinates=Coordinates(40.7484, -73.9857),
            description="Empire State Building"
        )
        
        # Add locations to map
        nav.map.add_location(home)
        nav.map.add_location(destination)
        
        # Set current location
        nav.set_current_location(1)
        
        # Plan a route
        route = nav._plan_route(home, destination)
        
        # Navigate to destination
        await nav.navigate_to(2)
        
        # Run the agent for a short time to simulate navigation
        agent_task = asyncio.create_task(agent.run())
        
        # Wait for navigation to complete
        await asyncio.sleep(5)
        
        # Explore the area
        await nav.explore_area(2)
        
        # Wait for exploration to complete
        await asyncio.sleep(5)
        
        # Export map to knowledge base
        nav.export_map_to_knowledge_base()
        
        # Shutdown
        await agent.shutdown()
        await agent_task
        
    asyncio.run(main())
