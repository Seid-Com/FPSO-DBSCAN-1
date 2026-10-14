#!/usr/bin/env python3
"""
Launcher script for the Adaptive Fuzzy-PSO DBSCAN application
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application"""
    try:
        # Change to the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Launch Streamlit
        cmd = [
            sys.executable, 
            "-m", "streamlit", 
            "run", 
            "app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ]
        
        print("Starting Adaptive Fuzzy-PSO DBSCAN application...")
        print("The application will be available at: http://localhost:8501")
        print("Press Ctrl+C to stop the application")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()