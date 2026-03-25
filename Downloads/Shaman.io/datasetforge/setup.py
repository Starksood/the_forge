#!/usr/bin/env python3
"""Setup script for DatasetForge development environment."""
import os
import subprocess
import sys
import venv

def create_virtual_environment():
    """Create a Python virtual environment."""
    venv_path = "venv"
    
    if os.path.exists(venv_path):
        print(f"Virtual environment already exists at {venv_path}")
        return venv_path
    
    print(f"Creating virtual environment at {venv_path}...")
    venv.create(venv_path, with_pip=True)
    print("Virtual environment created successfully!")
    return venv_path

def install_dependencies(venv_path):
    """Install Python dependencies in the virtual environment."""
    if os.name == 'nt':  # Windows
        pip_path = os.path.join(venv_path, "Scripts", "pip")
        python_path = os.path.join(venv_path, "Scripts", "python")
    else:  # Unix/Linux/macOS
        pip_path = os.path.join(venv_path, "bin", "pip")
        python_path = os.path.join(venv_path, "bin", "python")
    
    print("Installing Python dependencies...")
    subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
    print("Python dependencies installed successfully!")

def setup_frontend():
    """Set up the frontend build."""
    frontend_dir = "frontend"
    dist_dir = os.path.join(frontend_dir, "dist")
    
    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir, exist_ok=True)
        print("Created frontend dist directory")
    
    # The static HTML file is already created
    print("Frontend setup complete (using static HTML)")

def main():
    """Main setup function."""
    print("Setting up DatasetForge development environment...")
    
    # Create virtual environment
    venv_path = create_virtual_environment()
    
    # Install dependencies
    install_dependencies(venv_path)
    
    # Setup frontend
    setup_frontend()
    
    # Create sessions directory
    os.makedirs("sessions", exist_ok=True)
    print("Created sessions directory")
    
    print("\n" + "="*50)
    print("Setup complete!")
    print("="*50)
    print("\nTo run DatasetForge:")
    if os.name == 'nt':  # Windows
        print("1. Activate virtual environment: venv\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("1. Activate virtual environment: source venv/bin/activate")
    print("2. Start the application: python app.py")
    print("3. Open http://localhost:8000 in your browser")
    print("\nMake sure Ollama is running with gemma3:4b model!")

if __name__ == "__main__":
    main()