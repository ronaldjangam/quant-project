"""
Utility functions for the trading system.
"""

import yaml
import logging
from typing import Dict, Any
from datetime import datetime


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_file: str = None, level: str = 'INFO'):
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    log_level = getattr(logging, level.upper())
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def format_currency(value: float) -> str:
    """Format value as currency."""
    return f"${value:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage."""
    return f"{value*100:.2f}%"


def calculate_annual_return(total_return: float, num_days: int) -> float:
    """
    Calculate annualized return.
    
    Args:
        total_return: Total return (e.g., 0.20 for 20%)
        num_days: Number of days
        
    Returns:
        Annualized return
    """
    if num_days == 0:
        return 0.0
    
    years = num_days / 252  # Trading days
    annual_return = (1 + total_return) ** (1 / years) - 1
    
    return annual_return


def validate_date_range(start_date: str, end_date: str) -> bool:
    """
    Validate date range.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        True if valid, False otherwise
    """
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        return start < end
    except:
        return False
