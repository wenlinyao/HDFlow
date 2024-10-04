import psutil
import os
import time

def kill_process_if_exceeds_memory_limit(memory_limit=10):
    """Kill processes exceeding a specified memory limit percentage."""
    # Get the total memory available
    total_memory = psutil.virtual_memory().total
    
    # Calculate the memory usage threshold in bytes
    threshold = (total_memory * memory_limit) / 100
    
    # Iterate through all running processes
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            # Memory usage of the process
            process_memory = proc.info['memory_info'].rss
            
            # Check if the process exceeds the memory usage threshold
            if process_memory > threshold:
                print(f"Process {proc.info['name']} (PID {proc.info['pid']}) is using {process_memory} bytes. Killing it.")
                # Terminate the process
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Handle errors if the process has been killed or access is denied
            pass

if __name__ == "__main__":
    print("start monitoring...")
    while True:
        kill_process_if_exceeds_memory_limit()
        time.sleep(15)  # Check every 10 seconds
