"""
Database connection and management utilities.
"""

import os
from typing import Optional
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# SQLAlchemy Base
Base = declarative_base()


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'quant_trading'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', '')
        }
        
        # Connection pool for psycopg2
        self.connection_pool: Optional[pool.ThreadedConnectionPool] = None
        
        # SQLAlchemy engine and session
        self.engine = None
        self.Session = None
        
    def initialize_pool(self, min_conn=1, max_conn=10):
        """Initialize connection pool."""
        try:
            self.connection_pool = pool.ThreadedConnectionPool(
                min_conn,
                max_conn,
                **self.db_config
            )
            print("✓ Connection pool created successfully")
        except Exception as e:
            print(f"✗ Error creating connection pool: {e}")
            raise
    
    def initialize_sqlalchemy(self):
        """Initialize SQLAlchemy engine and session."""
        db_url = (
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}"
            f"@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        
        try:
            self.engine = create_engine(db_url, pool_pre_ping=True)
            self.Session = sessionmaker(bind=self.engine)
            print("✓ SQLAlchemy engine created successfully")
        except Exception as e:
            print(f"✗ Error creating SQLAlchemy engine: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool (psycopg2)."""
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                self.connection_pool.putconn(conn)
    
    @contextmanager
    def get_session(self):
        """Get a SQLAlchemy session."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def execute_script(self, script_path: str):
        """Execute SQL script file."""
        with open(script_path, 'r') as f:
            sql_script = f.read()
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(sql_script)
                conn.commit()
                print(f"✓ Script {script_path} executed successfully")
            except Exception as e:
                conn.rollback()
                print(f"✗ Error executing script: {e}")
                raise
            finally:
                cursor.close()
    
    def create_tables(self):
        """Create all tables using SQLAlchemy."""
        try:
            Base.metadata.create_all(self.engine)
            print("✓ All tables created successfully")
        except Exception as e:
            print(f"✗ Error creating tables: {e}")
            raise
    
    def close(self):
        """Close all connections."""
        if self.connection_pool:
            self.connection_pool.closeall()
        if self.engine:
            self.engine.dispose()
        print("✓ All database connections closed")


# Global database manager instance
db_manager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return db_manager


if __name__ == "__main__":
    # Test database connection
    db = get_db_manager()
    
    try:
        db.initialize_pool()
        db.initialize_sqlalchemy()
        
        # Test connection
        with db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            print(f"PostgreSQL version: {version[0]}")
            cursor.close()
        
        print("✓ Database connection test successful!")
        
    except Exception as e:
        print(f"✗ Database connection test failed: {e}")
    finally:
        db.close()
