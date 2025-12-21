import sys
import argparse
import logging
import os

# Ensure the current directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Configuration first
import config

def parse_args():
    """
    Parses command-line arguments to determine the execution mode.
    """
    parser = argparse.ArgumentParser(description="Flow State (Pipe Dream) - Manager")
    
    # --- Execution Mode ---
    parser.add_argument("--mode", choices=["dashboard", "sim", "builder"], default="dashboard",
                       help="Which component to launch (default: dashboard)")
    
    # --- Global Flags ---
    parser.add_argument("--debug", action="store_true", help="Enable debug logging and visual overlays")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
    
    # --- Window Configuration ---
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    parser.add_argument("--fullscreen", action="store_true", help="Run in fullscreen mode")
    
    return parser.parse_args()

def setup_logging(debug_mode):
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("FlowState")

def run_simulation(args, logger):
    """Initializes and runs the main Simulation App."""
    try:
        # Apply Configuration
        config.DEBUG = args.debug
        config.SCREEN_WIDTH = args.width
        config.SCREEN_HEIGHT = args.height
        config.FULLSCREEN = args.fullscreen
        
        logger.info(f"Launching Simulation (Resolution: {args.width}x{args.height}, Debug: {args.debug})")
        
        # Import late to prevent Pygame load during Dashboard mode
        from flow_state_app import FlowStateApp
        
        app = FlowStateApp()
        app.run()
        
    except ImportError as e:
        logger.critical(f"Failed to import Simulation modules: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Simulation crashed: {e}", exc_info=True)
        sys.exit(1)

def run_builder(args, logger):
    """Initializes and runs the Model Builder."""
    try:
        # Apply Configuration
        config.DEBUG = args.debug
        config.SCREEN_WIDTH = args.width
        config.SCREEN_HEIGHT = args.height
        config.FULLSCREEN = args.fullscreen
        
        logger.info(f"Launching Model Builder (Resolution: {args.width}x{args.height})")
        
        from flow_state_app import FlowStateApp
        
        # Initialize app in EDITOR mode as requested
        # We try to use config.MODE_EDITOR, defaulting to 'editor' string if not found
        start_mode = getattr(config, 'MODE_EDITOR', 'editor')
        
        app = FlowStateApp(start_mode=start_mode)
        app.run()
        
    except ImportError as e:
        logger.critical(f"Failed to import Application modules: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Builder crashed: {e}", exc_info=True)
        sys.exit(1)

def run_dashboard(logger):
    """Initializes and runs the Dashboard GUI."""
    try:
        logger.info("Launching Dashboard...")
        import dashboard
        dashboard.run()
    except Exception as e:
        logger.critical(f"Dashboard crashed: {e}", exc_info=True)
        sys.exit(1)

def main():
    args = parse_args()
    logger = setup_logging(args.debug)
    
    if args.mode == "dashboard":
        run_dashboard(logger)
    elif args.mode == "sim":
        run_simulation(args, logger)
    elif args.mode == "builder":
        run_builder(args, logger)
    else:
        logger.error(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()