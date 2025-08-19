#!/usr/bin/env python3
"""
AudioDoc Server Starter
Automatically runs the server with appropriate timeout settings for long processing tasks.
"""

import subprocess
import sys
import os
import signal

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import gunicorn
        return True
    except ImportError:
        print("Installing gunicorn...")
        subprocess.run([sys.executable, "-m", "pip", "install", "gunicorn"], check=True)
        return True

def start_server():
    """Start the server with appropriate settings"""
    if not check_dependencies():
        print("Failed to install dependencies")
        return False
    
    print("🎵 Starting AudioDoc Server with extended timeout for Whisper processing...")
    print("⏱️  Configured for 10-minute request timeout (perfect for large audio files)")
    print("🚀 Server will be available at: http://localhost:5001")
    print("📝 Large-v3 Whisper model will be downloaded on first use (~1.5GB)")
    print("\n" + "="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        # Run gunicorn with extended timeout and single worker for memory efficiency
        cmd = [
            "gunicorn",
            "--bind", "0.0.0.0:5001",
            "--timeout", "600",  # 10 minutes
            "--workers", "1",    # Single worker for memory efficiency
            "--threads", "2",    # Allow some concurrency
            "--worker-class", "sync",
            "--max-requests", "100",  # Restart worker after 100 requests to prevent memory leaks
            "--max-requests-jitter", "10",
            "--preload",  # Preload app for better memory usage
            "--access-logfile", "-",  # Log to stdout
            "--error-logfile", "-",   # Error log to stdout
            "app:app"
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = start_server()
    sys.exit(0 if success else 1)
