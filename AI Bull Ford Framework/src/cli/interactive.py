"""Interactive CLI for AI Bull Ford Framework.

Provides an interactive command-line interface for framework management,
pipeline execution, and real-time monitoring.
"""

import asyncio
import logging
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path

try:
    import readline
except ImportError:
    # readline not available on Windows by default
    readline = None

from .commands import CLICommands
from .utils import CLIUtils

logger = logging.getLogger(__name__)


class InteractiveCLI:
    """Interactive command-line interface for AIBF framework."""
    
    def __init__(self, framework):
        """Initialize the interactive CLI.
        
        Args:
            framework: The AIBullFordFramework instance
        """
        self.framework = framework
        self.commands = CLICommands(framework)
        self.utils = CLIUtils()
        self.running = False
        self.history_file = Path.home() / ".aibf_history"
        
        # Setup readline if available
        if readline:
            self._setup_readline()
    
    def _setup_readline(self):
        """Setup readline for command history and completion."""
        try:
            # Load command history
            if self.history_file.exists():
                readline.read_history_file(str(self.history_file))
            
            # Set history length
            readline.set_history_length(1000)
            
            # Setup tab completion
            readline.set_completer(self._complete_command)
            readline.parse_and_bind("tab: complete")
            
        except Exception as e:
            logger.warning(f"Failed to setup readline: {e}")
    
    def _complete_command(self, text: str, state: int) -> Optional[str]:
        """Tab completion for commands."""
        try:
            commands = list(self.commands.get_available_commands().keys())
            matches = [cmd for cmd in commands if cmd.startswith(text)]
            
            if state < len(matches):
                return matches[state]
            return None
            
        except Exception:
            return None
    
    def _save_history(self):
        """Save command history to file."""
        if readline:
            try:
                readline.write_history_file(str(self.history_file))
            except Exception as e:
                logger.warning(f"Failed to save command history: {e}")
    
    async def start(self):
        """Start the interactive CLI session."""
        self.running = True
        
        # Display welcome message
        self._display_welcome()
        
        try:
            while self.running:
                try:
                    # Get user input
                    prompt = self.utils.get_colored_prompt()
                    user_input = input(prompt).strip()
                    
                    if not user_input:
                        continue
                    
                    # Parse and execute command
                    await self._execute_command(user_input)
                    
                except KeyboardInterrupt:
                    print("\n\nUse 'exit' or 'quit' to leave the CLI.")
                    continue
                except EOFError:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    logger.error(f"CLI error: {e}")
                    print(f"Error: {e}")
        
        finally:
            self._save_history()
            await self._cleanup()
    
    def _display_welcome(self):
        """Display welcome message and help information."""
        welcome_text = """
╔══════════════════════════════════════════════════════════════╗
║                  AI Bull Ford Framework                      ║
║                   Interactive CLI Mode                       ║
╚══════════════════════════════════════════════════════════════╝

Welcome to the AI Bull Ford Framework Interactive CLI!

Type 'help' to see available commands.
Type 'exit' or 'quit' to leave the CLI.
Use Tab for command completion.

        """
        print(welcome_text)
    
    async def _execute_command(self, user_input: str):
        """Parse and execute a user command.
        
        Args:
            user_input: The command string entered by the user
        """
        # Parse command and arguments
        parts = user_input.split()
        if not parts:
            return
        
        command = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        # Handle built-in commands
        if command in ['exit', 'quit', 'q']:
            print("Goodbye!")
            self.running = False
            return
        
        elif command in ['help', 'h', '?']:
            self._display_help(args)
            return
        
        elif command == 'clear':
            self.utils.clear_screen()
            return
        
        # Execute framework commands
        try:
            result = await self.commands.execute_command(command, args)
            if result is not None:
                self._display_result(result)
        
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            print(f"Error executing command '{command}': {e}")
    
    def _display_help(self, args: List[str]):
        """Display help information.
        
        Args:
            args: Optional command to get specific help for
        """
        if args:
            # Show help for specific command
            command = args[0]
            help_text = self.commands.get_command_help(command)
            if help_text:
                print(f"\nHelp for '{command}':")
                print(help_text)
            else:
                print(f"No help available for command '{command}'")
        else:
            # Show general help
            self._display_general_help()
    
    def _display_general_help(self):
        """Display general help and available commands."""
        help_text = """
Available Commands:

╔════════════════════════════════════════════════════════════════╗
║                        Framework Commands                      ║
╠════════════════════════════════════════════════════════════════╣
║ status              - Show framework status                    ║
║ config              - Display current configuration            ║
║ modules             - List available modules                   ║
║ pipelines           - Manage pipelines                         ║
║ models              - Manage AI models                         ║
║ metrics             - Show performance metrics                 ║
║ logs                - View framework logs                      ║
╠════════════════════════════════════════════════════════════════╣
║                        Pipeline Commands                       ║
╠════════════════════════════════════════════════════════════════╣
║ run <pipeline>      - Execute a pipeline                       ║
║ create <pipeline>   - Create a new pipeline                    ║
║ list                - List available pipelines                 ║
║ stop <pipeline>     - Stop a running pipeline                  ║
╠════════════════════════════════════════════════════════════════╣
║                        Server Commands                         ║
╠════════════════════════════════════════════════════════════════╣
║ start <service>     - Start API service                        ║
║ stop <service>      - Stop API service                         ║
║ restart <service>   - Restart API service                      ║
║ services            - List running services                    ║
╠════════════════════════════════════════════════════════════════╣
║                        Utility Commands                        ║
╠════════════════════════════════════════════════════════════════╣
║ help [command]      - Show help (for specific command)         ║
║ clear               - Clear the screen                         ║
║ exit/quit           - Exit the CLI                             ║
╚════════════════════════════════════════════════════════════════╝

Examples:
  status                    - Show current framework status
  run healthcare_pipeline   - Execute the healthcare pipeline
  start rest                - Start the REST API server
  help run                  - Get help for the 'run' command

For detailed help on any command, type: help <command>
        """
        print(help_text)
    
    def _display_result(self, result: Any):
        """Display command execution result.
        
        Args:
            result: The result to display
        """
        if isinstance(result, dict):
            self.utils.print_json(result)
        elif isinstance(result, list):
            self.utils.print_table(result)
        elif isinstance(result, str):
            print(result)
        else:
            print(f"Result: {result}")
    
    async def _cleanup(self):
        """Cleanup resources before exiting."""
        try:
            logger.info("Cleaning up CLI resources...")
            # Any cleanup operations can be added here
        except Exception as e:
            logger.error(f"Error during CLI cleanup: {e}")


class CLISession:
    """Manages CLI session state and context."""
    
    def __init__(self):
        self.session_id = None
        self.start_time = None
        self.command_count = 0
        self.context = {}
    
    def start_session(self):
        """Start a new CLI session."""
        import uuid
        from datetime import datetime
        
        self.session_id = str(uuid.uuid4())[:8]
        self.start_time = datetime.now()
        self.command_count = 0
        logger.info(f"Started CLI session {self.session_id}")
    
    def end_session(self):
        """End the current CLI session."""
        if self.start_time:
            from datetime import datetime
            duration = datetime.now() - self.start_time
            logger.info(
                f"Ended CLI session {self.session_id} "
                f"(duration: {duration}, commands: {self.command_count})"
            )
    
    def increment_command_count(self):
        """Increment the command counter."""
        self.command_count += 1
    
    def set_context(self, key: str, value: Any):
        """Set a context variable."""
        self.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get a context variable."""
        return self.context.get(key, default)