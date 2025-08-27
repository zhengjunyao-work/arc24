#!/usr/bin/env python3
"""
Simple script to toggle GPU usage in training configuration
"""

import os

def toggle_gpu():
    """Toggle GPU usage in training_config.py"""
    config_file = "training_config.py"
    
    if not os.path.exists(config_file):
        print(f"❌ Configuration file {config_file} not found!")
        return
    
    # Read current configuration
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Check current GPU setting
    if "USE_GPU = True" in content:
        # Disable GPU
        new_content = content.replace("USE_GPU = True", "USE_GPU = False")
        print("🖥️  Disabling GPU usage...")
    elif "USE_GPU = False" in content:
        # Enable GPU
        new_content = content.replace("USE_GPU = False", "USE_GPU = True")
        print("✅ Enabling GPU usage...")
    else:
        print("❌ Could not find USE_GPU setting in configuration file")
        return
    
    # Write updated configuration
    with open(config_file, 'w') as f:
        f.write(new_content)
    
    print(f"✅ Configuration updated successfully!")
    
    # Show current status
    if "USE_GPU = True" in new_content:
        print("🖥️  Current status: GPU ENABLED")
    else:
        print("🖥️  Current status: GPU DISABLED")

def show_status():
    """Show current GPU configuration status"""
    config_file = "training_config.py"
    
    if not os.path.exists(config_file):
        print(f"❌ Configuration file {config_file} not found!")
        return
    
    with open(config_file, 'r') as f:
        content = f.read()
    
    if "USE_GPU = True" in content:
        print("✅ GPU Usage: ENABLED")
        print("   Training will use GPU if available")
    else:
        print("🖥️  GPU Usage: DISABLED")
        print("   Training will use CPU only")

def main():
    """Main function"""
    print("GPU Configuration Toggle")
    print("=" * 30)
    
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ['toggle', 'switch', 'change']:
            toggle_gpu()
        elif command in ['status', 'show', 'check']:
            show_status()
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: toggle, status")
    else:
        # Interactive mode
        show_status()
        print("\nOptions:")
        print("1. Toggle GPU usage")
        print("2. Show current status")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            toggle_gpu()
        elif choice == '2':
            show_status()
        elif choice == '3':
            print("Goodbye!")
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
