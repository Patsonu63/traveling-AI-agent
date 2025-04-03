#!/usr/bin/env python3
"""
Traveling AI Agent - Knowledge Base Module

This module implements the Knowledge Base component responsible for
storing, organizing, and retrieving information collected by the agent.
"""

import logging
import json
import os
import sqlite3
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from agent_core import Component


class KnowledgeBase(Component):
    """Knowledge Base component for storing and retrieving information."""
    
    def __init__(self, db_path: str = "knowledge.db"):
        """Initialize the Knowledge Base component.
        
        Args:
            db_path: Path to the SQLite database file
        """
        super().__init__("KnowledgeBase")
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.memory_cache = {}
        self.max_cache_size = 1000
        
    async def initialize(self) -> bool:
        """Initialize the Knowledge Base.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        self.logger.info("Initializing Knowledge Base")
        
        try:
            # Create database directory if it doesn't exist
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                
            # Connect to the database
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create tables if they don't exist
            self._create_tables()
            
            self.active = True
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Knowledge Base: {e}")
            return False
            
    async def shutdown(self) -> bool:
        """Shutdown the Knowledge Base.
        
        Returns:
            bool: True if shutdown was successful, False otherwise
        """
        self.logger.info("Shutting down Knowledge Base")
        
        try:
            if self.conn:
                self.conn.close()
                
            self.active = False
            return True
            
        except Exception as e:
            self.logger.error(f"Error shutting down Knowledge Base: {e}")
            return False
            
    def _create_tables(self) -> None:
        """Create the necessary database tables."""
        # Locations table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            latitude REAL,
            longitude REAL,
            description TEXT,
            visited BOOLEAN DEFAULT 0,
            visit_count INTEGER DEFAULT 0,
            last_visited TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Points of interest table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS points_of_interest (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_id INTEGER,
            name TEXT NOT NULL,
            type TEXT,
            description TEXT,
            importance INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (location_id) REFERENCES locations (id)
        )
        ''')
        
        # Facts table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Images table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_id INTEGER,
            path TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (location_id) REFERENCES locations (id)
        )
        ''')
        
        # Routes table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS routes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_location_id INTEGER,
            end_location_id INTEGER,
            distance REAL,
            estimated_time REAL,
            difficulty INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (start_location_id) REFERENCES locations (id),
            FOREIGN KEY (end_location_id) REFERENCES locations (id)
        )
        ''')
        
        # Memory table (for short-term memory)
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE,
            value TEXT NOT NULL,
            expiry TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        self.conn.commit()
        
    async def update(self) -> None:
        """Update the Knowledge Base state."""
        # Clean up expired memory entries
        self._cleanup_expired_memory()
        
    def _cleanup_expired_memory(self) -> None:
        """Clean up expired memory entries."""
        try:
            current_time = datetime.now().isoformat()
            self.cursor.execute(
                "DELETE FROM memory WHERE expiry IS NOT NULL AND expiry < ?",
                (current_time,)
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Error cleaning up expired memory: {e}")
            
    # Location methods
    
    def add_location(self, name: str, latitude: float = None, longitude: float = None, 
                    description: str = None) -> int:
        """Add a new location to the knowledge base.
        
        Args:
            name: The name of the location
            latitude: The latitude coordinate
            longitude: The longitude coordinate
            description: A description of the location
            
        Returns:
            int: The ID of the newly added location
        """
        try:
            self.cursor.execute(
                "INSERT INTO locations (name, latitude, longitude, description) VALUES (?, ?, ?, ?)",
                (name, latitude, longitude, description)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Error adding location: {e}")
            return -1
            
    def get_location(self, location_id: int) -> Optional[Dict[str, Any]]:
        """Get a location by ID.
        
        Args:
            location_id: The ID of the location to get
            
        Returns:
            Optional[Dict[str, Any]]: The location data if found, None otherwise
        """
        try:
            self.cursor.execute("SELECT * FROM locations WHERE id = ?", (location_id,))
            row = self.cursor.fetchone()
            if row:
                columns = [desc[0] for desc in self.cursor.description]
                return dict(zip(columns, row))
            return None
        except Exception as e:
            self.logger.error(f"Error getting location: {e}")
            return None
            
    def get_locations(self, visited: bool = None) -> List[Dict[str, Any]]:
        """Get all locations, optionally filtered by visited status.
        
        Args:
            visited: If provided, filter by visited status
            
        Returns:
            List[Dict[str, Any]]: List of location data
        """
        try:
            if visited is not None:
                self.cursor.execute("SELECT * FROM locations WHERE visited = ?", (1 if visited else 0,))
            else:
                self.cursor.execute("SELECT * FROM locations")
                
            rows = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            self.logger.error(f"Error getting locations: {e}")
            return []
            
    def mark_location_visited(self, location_id: int) -> bool:
        """Mark a location as visited.
        
        Args:
            location_id: The ID of the location to mark
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            current_time = datetime.now().isoformat()
            self.cursor.execute(
                "UPDATE locations SET visited = 1, visit_count = visit_count + 1, last_visited = ? WHERE id = ?",
                (current_time, location_id)
            )
            self.conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error marking location as visited: {e}")
            return False
            
    # Points of interest methods
    
    def add_point_of_interest(self, name: str, location_id: int = None, poi_type: str = None,
                             description: str = None, importance: int = 0) -> int:
        """Add a new point of interest.
        
        Args:
            name: The name of the point of interest
            location_id: The ID of the associated location
            poi_type: The type of point of interest
            description: A description of the point of interest
            importance: The importance level (0-10)
            
        Returns:
            int: The ID of the newly added point of interest
        """
        try:
            self.cursor.execute(
                "INSERT INTO points_of_interest (location_id, name, type, description, importance) VALUES (?, ?, ?, ?, ?)",
                (location_id, name, poi_type, description, importance)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Error adding point of interest: {e}")
            return -1
            
    def get_points_of_interest(self, location_id: int = None) -> List[Dict[str, Any]]:
        """Get points of interest, optionally filtered by location.
        
        Args:
            location_id: If provided, filter by location ID
            
        Returns:
            List[Dict[str, Any]]: List of point of interest data
        """
        try:
            if location_id is not None:
                self.cursor.execute("SELECT * FROM points_of_interest WHERE location_id = ?", (location_id,))
            else:
                self.cursor.execute("SELECT * FROM points_of_interest")
                
            rows = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            self.logger.error(f"Error getting points of interest: {e}")
            return []
            
    # Fact methods
    
    def add_fact(self, subject: str, predicate: str, obj: str, confidence: float = 1.0, source: str = None) -> int:
        """Add a new fact to the knowledge base.
        
        Args:
            subject: The subject of the fact
            predicate: The predicate of the fact
            obj: The object of the fact
            confidence: The confidence level (0.0-1.0)
            source: The source of the fact
            
        Returns:
            int: The ID of the newly added fact
        """
        try:
            self.cursor.execute(
                "INSERT INTO facts (subject, predicate, object, confidence, source) VALUES (?, ?, ?, ?, ?)",
                (subject, predicate, obj, confidence, source)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Error adding fact: {e}")
            return -1
            
    def get_facts(self, subject: str = None, predicate: str = None, obj: str = None) -> List[Dict[str, Any]]:
        """Get facts, optionally filtered by subject, predicate, or object.
        
        Args:
            subject: If provided, filter by subject
            predicate: If provided, filter by predicate
            obj: If provided, filter by object
            
        Returns:
            List[Dict[str, Any]]: List of fact data
        """
        try:
            query = "SELECT * FROM facts WHERE 1=1"
            params = []
            
            if subject is not None:
                query += " AND subject = ?"
                params.append(subject)
                
            if predicate is not None:
                query += " AND predicate = ?"
                params.append(predicate)
                
            if obj is not None:
                query += " AND object = ?"
                params.append(obj)
                
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            self.logger.error(f"Error getting facts: {e}")
            return []
            
    # Memory methods
    
    def remember(self, key: str, value: Any, expiry: datetime = None) -> bool:
        """Store a value in memory.
        
        Args:
            key: The key to store the value under
            value: The value to store
            expiry: Optional expiry time
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Store in memory cache
            self.memory_cache[key] = value
            
            # Trim cache if it gets too large
            if len(self.memory_cache) > self.max_cache_size:
                # Remove oldest items
                keys_to_remove = list(self.memory_cache.keys())[:-self.max_cache_size]
                for k in keys_to_remove:
                    del self.memory_cache[k]
                    
            # Store in database
            value_json = json.dumps(value)
            expiry_str = expiry.isoformat() if expiry else None
            
            self.cursor.execute(
                "INSERT OR REPLACE INTO memory (key, value, expiry) VALUES (?, ?, ?)",
                (key, value_json, expiry_str)
            )
            self.conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error storing memory: {e}")
            return False
            
    def recall(self, key: str) -> Any:
        """Recall a value from memory.
        
        Args:
            key: The key to recall
            
        Returns:
            Any: The recalled value, or None if not found
        """
        try:
            # Check memory cache first
            if key in self.memory_cache:
                return self.memory_cache[key]
                
            # Check database
            self.cursor.execute("SELECT value FROM memory WHERE key = ?", (key,))
            row = self.cursor.fetchone()
            
            if row:
                value = json.loads(row[0])
                # Update cache
                self.memory_cache[key] = value
                return value
                
            return None
        except Exception as e:
            self.logger.error(f"Error recalling memory: {e}")
            return None
            
    def forget(self, key: str) -> bool:
        """Remove a value from memory.
        
        Args:
            key: The key to forget
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Remove from cache
            if key in self.memory_cache:
                del self.memory_cache[key]
                
            # Remove from database
            self.cursor.execute("DELETE FROM memory WHERE key = ?", (key,))
            self.conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error forgetting memory: {e}")
            return False
            
    # Route methods
    
    def add_route(self, start_location_id: int, end_location_id: int, 
                 distance: float = None, estimated_time: float = None, 
                 difficulty: int = 0) -> int:
        """Add a new route between locations.
        
        Args:
            start_location_id: The ID of the starting location
            end_location_id: The ID of the ending location
            distance: The distance of the route
            estimated_time: The estimated time to travel the route
            difficulty: The difficulty level of the route (0-10)
            
        Returns:
            int: The ID of the newly added route
        """
        try:
            self.cursor.execute(
                "INSERT INTO routes (start_location_id, end_location_id, distance, estimated_time, difficulty) VALUES (?, ?, ?, ?, ?)",
                (start_location_id, end_location_id, distance, estimated_time, difficulty)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Error adding route: {e}")
            return -1
            
    def get_routes(self, start_location_id: int = None, end_location_id: int = None) -> List[Dict[str, Any]]:
        """Get routes, optionally filtered by start or end location.
        
        Args:
            start_location_id: If provided, filter by start location ID
            end_location_id: If provided, filter by end location ID
            
        Returns:
            List[Dict[str, Any]]: List of route data
        """
        try:
            query = "SELECT * FROM routes WHERE 1=1"
            params = []
            
            if start_location_id is not None:
                query += " AND start_location_id = ?"
                params.append(start_location_id)
                
            if end_location_id is not None:
                query += " AND end_location_id = ?"
                params.append(end_location_id)
                
            self.cursor.execute(query, params)
            rows = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            self.logger.error(f"Error getting routes: {e}")
            return []
            
    def export_knowledge(self, format: str = "json") -> str:
        """Export the knowledge base to a specified format.
        
        Args:
            format: The export format (currently only 'json' is supported)
            
        Returns:
            str: The exported knowledge base
        """
        if format.lower() != "json":
            self.logger.error(f"Unsupported export format: {format}")
            return ""
            
        try:
            export_data = {
                "locations": self.get_locations(),
                "points_of_interest": self.get_points_of_interest(),
                "facts": self.get_facts(),
                "routes": self.get_routes()
            }
            
            return json.dumps(export_data, indent=2)
        except Exception as e:
            self.logger.error(f"Error exporting knowledge: {e}")
            return ""
            
    def import_knowledge(self, data: str, format: str = "json") -> bool:
        """Import knowledge from a specified format.
        
        Args:
            data: The data to import
            format: The import format (currently only 'json' is supported)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if format.lower() != "json":
            self.logger.error(f"Unsupported import format: {format}")
            return False
            
        try:
            import_data = json.loads(data)
            
            # Import locations
            for location in import_data.get("locations", []):
                self.add_location(
                    name=location["name"],
                    latitude=location.get("latitude"),
                    longitude=location.get("longitude"),
                    description=location.get("description")
                )
                
            # Import points of interest
            for poi in import_data.get("points_of_interest", []):
                self.add_point_of_interest(
                    name=poi["name"],
                    location_id=poi.get("location_id"),
                    poi_type=poi.get("type"),
                    description=poi.get("description"),
                    importance=poi.get("importance", 0)
                )
                
            # Import facts
            for fact in import_data.get("facts", []):
                self.add_fact(
                    subject=fact["subject"],
                    predicate=fact["predicate"],
                    obj=fact["object"],
                    confidence=fact.get("confidence", 1.0),
                    source=fact.get("source")
                )
                
            # Import routes
            for route in import_data.get("routes", []):
                self.add_route(
                    start_location_id=route["start_location_id"],
                    end_location_id=route["end_location_id"],
                    distance=route.get("distance"),
                    estimated_time=route.get("estimated_time"),
                    difficulty=route.get("difficulty", 0)
                )
                
            return True
        except Exception as e:
            self.logger.error(f"Error importing knowledge: {e}")
            return False


# Example usage
if __name__ == "__main__":
    import asyncio
    from agent_core import TravelingAgent
    
    async def main():
        # Create the agent
        agent = TravelingAgent("KnowledgeTest")
        
        # Create and register the knowledge base
        kb = KnowledgeBase("test_knowledge.db")
        agent.register_component(kb)
        
        # Initialize the agent
        await agent.initialize()
        
        # Add some test data
        location_id = kb.add_location(
            name="New York City",
            latitude=40.7128,
            longitude=-74.0060,
            description="The largest city in the United States"
        )
        
        kb.add_point_of_interest(
            name="Empire State Building",
            location_id=location_id,
            poi_type="landmark",
            description="Famous skyscraper",
            importance=8
        )
        
        kb.add_fact(
            subject="New York City",
            predicate="population",
            obj="8.4 million",
            confidence=0.95,
            source="US Census Bureau"
        )
        
        # Test memory
        kb.remember("last_location", "New York City")
        recalled = kb.recall("last_location")
        print(f"Recalled: {recalled}")
        
        # Export knowledge
        export = kb.export_knowledge()
        print(f"Exported knowledge: {export}")
        
        # Shutdown the agent
        await agent.shutdown()
        
    asyncio.run(main())
