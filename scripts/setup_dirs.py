"""
Create necessary directories for the project.
"""

import os


def create_directories():
    """Create all necessary directories."""
    directories = [
        'data/raw',
        'data/processed',
        'data/external',
        'models/saved',
        'logs',
        'results',
        'plots',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
        # Create .gitkeep files
        gitkeep_path = os.path.join(directory, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            with open(gitkeep_path, 'w') as f:
                pass
    
    print("✓ Created all necessary directories")


if __name__ == "__main__":
    create_directories()
