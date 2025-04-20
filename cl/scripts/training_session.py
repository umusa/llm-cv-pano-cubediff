"""
Utility script to manage training sessions with 8-hour time limits.
This script monitors and manages the training process, automatically shutting down
after the specified time limit and setting up for resuming in the next session.
"""

import os
import time
import subprocess
import argparse
import json
import signal
import sys
from datetime import datetime, timedelta

def parse_args():
    parser = argparse.ArgumentParser(description="CubeDiff Training Session Manager")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--session_length_hours", 
        type=float, 
        default=8.0,
        help="Maximum session length in hours"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--resume_from", 
        type=str, 
        help="Path to checkpoint to resume from"
    )
    
    return parser.parse_args()

def get_latest_checkpoint(output_dir):
    """Get the path to the latest checkpoint in the output directory."""
    checkpoints = []
    
    # Look for checkpoint directories
    for d in os.listdir(output_dir):
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d)):
            try:
                step = int(d.split("-")[1])
                checkpoints.append((step, os.path.join(output_dir, d)))
            except ValueError:
                continue
    
    if not checkpoints:
        return None
    
    # Sort by step number and return the latest
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]

def save_session_state(output_dir, checkpoint_path, total_steps, remaining_steps):
    """Save the current session state to a JSON file."""
    state = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint_path": checkpoint_path,
        "total_steps": total_steps,
        "remaining_steps": remaining_steps,
    }
    
    state_path = os.path.join(output_dir, "session_state.json")
    with open(state_path, "w") as f:
        json.dump(state, f, indent=4)
    
    return state_path

def load_session_state(output_dir):
    """Load the session state from a JSON file."""
    state_path = os.path.join(output_dir, "session_state.json")
    
    if not os.path.exists(state_path):
        return None
    
    with open(state_path, "r") as f:
        state = json.load(f)
    
    return state

def run_training_session(args):
    """Run a training session with time limit."""
    # Determine output directory
    output_dir = args.output_dir
    if not output_dir:
        # Extract from config if not provided
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        output_dir = config.get("training", {}).get("output_dir", "outputs/cubediff")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine checkpoint to resume from
    checkpoint_path = args.resume_from
    if not checkpoint_path:
        # Try to find latest checkpoint
        state = load_session_state(output_dir)
        if state:
            checkpoint_path = state["checkpoint_path"]
            remaining_steps = state["remaining_steps"]
            total_steps = state["total_steps"]
            print(f"Resuming from checkpoint: {checkpoint_path}")
            print(f"Remaining steps: {remaining_steps} / {total_steps}")
        else:
            # Get total steps from config
            import yaml
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)
            
            total_steps = config.get("training", {}).get("train_steps", 30000)
            remaining_steps = total_steps
            print(f"Starting new training run for {total_steps} steps")
    
    # Calculate session end time
    session_end_time = datetime.now() + timedelta(hours=args.session_length_hours)
    print(f"Session will end at: {session_end_time}")
    
    # Start the training process
    cmd = [
        "python", "main.py",
        "--config", args.config,
        "--mode", "train",
        "--output_dir", output_dir
    ]
    
    if checkpoint_path:
        cmd.extend(["--checkpoint", checkpoint_path])
    
    # Set up process
    process = subprocess.Popen(cmd)
    
    try:
        # Monitor training time
        while process.poll() is None:
            # Check if time limit is reached
            if datetime.now() >= session_end_time:
                print("Time limit reached. Gracefully stopping training...")
                
                # Send SIGTERM to allow graceful shutdown
                process.terminate()
                
                # Wait for process to finish (with timeout)
                try:
                    process.wait(timeout=300)  # Wait up to 5 minutes
                except subprocess.TimeoutExpired:
                    print("Training process did not terminate in time. Killing...")
                    process.kill()
                
                break
            
            # Check every minute
            time.sleep(60)
        
        # Training process finished or was terminated
        exit_code = process.returncode
        print(f"Training process exited with code: {exit_code}")
        
        # Get the latest checkpoint
        latest_checkpoint = get_latest_checkpoint(output_dir)
        if latest_checkpoint:
            print(f"Latest checkpoint: {latest_checkpoint}")
            
            # Update session state
            # Note: In a real implementation, we would need a way to 
            # determine how many steps were completed
            # For now, we'll just use a placeholder
            completed_steps = 0  # Placeholder
            remaining_steps = total_steps - completed_steps
            
            state_path = save_session_state(
                output_dir, 
                latest_checkpoint, 
                total_steps, 
                remaining_steps
            )
            print(f"Session state saved to: {state_path}")
            
            # If training is not complete, suggest next session
            if remaining_steps > 0:
                print(f"\nTraining not complete. {remaining_steps} steps remaining.")
                print("To continue training in the next session, run:")
                print(f"python scripts/training_session.py --config {args.config} --output_dir {output_dir}")
        else:
            print("No checkpoint found. Training may have failed.")
    
    except KeyboardInterrupt:
        print("Interrupted by user. Stopping training...")
        process.terminate()
        try:
            process.wait(timeout=300)  # Wait up to 5 minutes
        except subprocess.TimeoutExpired:
            print("Training process did not terminate in time. Killing...")
            process.kill()

def main():
    args = parse_args()
    run_training_session(args)

if __name__ == "__main__":
    main()