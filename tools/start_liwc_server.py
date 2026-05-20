#!/usr/bin/env python3
import subprocess
import time
import sys

def check_liwc_running():
    try:
        result = subprocess.run(['pgrep', '-f', 'LIWC-22-license-server'], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def start_liwc_server():
    """Start LIWC license server if not already running."""
    if check_liwc_running():
        print("✅ LIWC-22-license-server is already running")
        return True
    
    print("🚀 Starting LIWC-22-license-server...")
    
    # Full path to LIWC license server
    liwc_server_path = '/Applications/LIWC-22.app/Contents/MacOS/LIWC-22-license-server'
    
    try:
        process = subprocess.Popen([liwc_server_path])
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ LIWC server started successfully")
            return True
        else:
            print("❌ Server failed to start")
            return False
            
    except FileNotFoundError:
        print(f"❌ Error: LIWC-22-license-server not found at {liwc_server_path}")
        return False
    except Exception as e:
        print(f"❌ Error starting LIWC server: {e}")
        return False

def main():
    return start_liwc_server()

if __name__ == "__main__":
    main()

