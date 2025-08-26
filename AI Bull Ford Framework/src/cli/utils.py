"""CLI Utilities for AI Bull Ford Framework.

Provides utility functions for CLI formatting, display, and interaction.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime


class CLIUtils:
    """Utility functions for CLI operations."""
    
    def __init__(self):
        """Initialize CLI utilities."""
        self.colors = self._setup_colors()
    
    def _setup_colors(self) -> Dict[str, str]:
        """Setup color codes for terminal output."""
        # Check if terminal supports colors
        if not self._supports_color():
            return {key: '' for key in [
                'reset', 'bold', 'dim', 'red', 'green', 'yellow', 
                'blue', 'magenta', 'cyan', 'white'
            ]}
        
        return {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'dim': '\033[2m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
            'bg_red': '\033[41m',
            'bg_green': '\033[42m',
            'bg_yellow': '\033[43m',
            'bg_blue': '\033[44m'
        }
    
    def _supports_color(self) -> bool:
        """Check if terminal supports color output."""
        # Check for common color support indicators
        if os.getenv('NO_COLOR'):
            return False
        
        if os.getenv('FORCE_COLOR'):
            return True
        
        # Check if stdout is a TTY
        if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
            return False
        
        # Check TERM environment variable
        term = os.getenv('TERM', '')
        if 'color' in term or term in ['xterm', 'xterm-256color', 'screen']:
            return True
        
        # Windows terminal support
        if sys.platform == 'win32':
            try:
                import colorama
                colorama.init()
                return True
            except ImportError:
                return False
        
        return False
    
    def colorize(self, text: str, color: str, bold: bool = False) -> str:
        """Apply color formatting to text.
        
        Args:
            text: Text to colorize
            color: Color name
            bold: Whether to make text bold
            
        Returns:
            Formatted text string
        """
        if color not in self.colors:
            return text
        
        result = self.colors[color] + text
        if bold:
            result = self.colors['bold'] + result
        result += self.colors['reset']
        
        return result
    
    def get_colored_prompt(self) -> str:
        """Get a colored CLI prompt."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        prompt_parts = [
            self.colorize('[', 'dim'),
            self.colorize(timestamp, 'cyan'),
            self.colorize(']', 'dim'),
            ' ',
            self.colorize('AIBF', 'blue', bold=True),
            self.colorize('>', 'green'),
            ' '
        ]
        return ''.join(prompt_parts)
    
    def print_success(self, message: str):
        """Print a success message."""
        print(self.colorize('✓ ', 'green') + message)
    
    def print_error(self, message: str):
        """Print an error message."""
        print(self.colorize('✗ ', 'red') + self.colorize(message, 'red'))
    
    def print_warning(self, message: str):
        """Print a warning message."""
        print(self.colorize('⚠ ', 'yellow') + self.colorize(message, 'yellow'))
    
    def print_info(self, message: str):
        """Print an info message."""
        print(self.colorize('ℹ ', 'blue') + message)
    
    def print_json(self, data: Dict[str, Any], indent: int = 2):
        """Print JSON data with syntax highlighting.
        
        Args:
            data: Dictionary to print as JSON
            indent: Indentation level
        """
        try:
            json_str = json.dumps(data, indent=indent, ensure_ascii=False)
            
            # Apply basic syntax highlighting if colors are supported
            if self.colors['reset']:
                json_str = self._highlight_json(json_str)
            
            print(json_str)
            
        except Exception as e:
            self.print_error(f"Failed to format JSON: {e}")
            print(str(data))
    
    def _highlight_json(self, json_str: str) -> str:
        """Apply basic syntax highlighting to JSON string."""
        import re
        
        # Highlight strings (values)
        json_str = re.sub(
            r'"([^"\\]*(\\.[^"\\]*)*)"\s*:',
            lambda m: self.colorize(f'"{m.group(1)}"', 'cyan') + ':',
            json_str
        )
        
        # Highlight string values
        json_str = re.sub(
            r':\s*"([^"\\]*(\\.[^"\\]*)*)"',
            lambda m: ': ' + self.colorize(f'"{m.group(1)}"', 'green'),
            json_str
        )
        
        # Highlight numbers
        json_str = re.sub(
            r':\s*(-?\d+\.?\d*)',
            lambda m: ': ' + self.colorize(m.group(1), 'yellow'),
            json_str
        )
        
        # Highlight booleans and null
        json_str = re.sub(
            r'\b(true|false|null)\b',
            lambda m: self.colorize(m.group(1), 'magenta'),
            json_str
        )
        
        return json_str
    
    def print_table(self, data: List[Dict[str, Any]], headers: Optional[List[str]] = None):
        """Print data as a formatted table.
        
        Args:
            data: List of dictionaries to display
            headers: Optional list of column headers
        """
        if not data:
            self.print_info("No data to display")
            return
        
        # Determine headers
        if headers is None:
            headers = list(data[0].keys()) if data else []
        
        # Calculate column widths
        col_widths = {}
        for header in headers:
            col_widths[header] = len(header)
            for row in data:
                value = str(row.get(header, ''))
                col_widths[header] = max(col_widths[header], len(value))
        
        # Print header
        header_row = ' | '.join(
            self.colorize(header.ljust(col_widths[header]), 'cyan', bold=True)
            for header in headers
        )
        print(header_row)
        
        # Print separator
        separator = '-+-'.join('-' * col_widths[header] for header in headers)
        print(self.colorize(separator, 'dim'))
        
        # Print data rows
        for row in data:
            row_str = ' | '.join(
                str(row.get(header, '')).ljust(col_widths[header])
                for header in headers
            )
            print(row_str)
    
    def print_progress_bar(self, current: int, total: int, width: int = 50, 
                          prefix: str = '', suffix: str = ''):
        """Print a progress bar.
        
        Args:
            current: Current progress value
            total: Total progress value
            width: Width of progress bar
            prefix: Text before progress bar
            suffix: Text after progress bar
        """
        if total == 0:
            percent = 100
        else:
            percent = int(100 * current / total)
        
        filled_width = int(width * current / total) if total > 0 else 0
        bar = '█' * filled_width + '░' * (width - filled_width)
        
        progress_text = f'{prefix} |{bar}| {percent}% {suffix}'
        
        # Use carriage return to overwrite previous line
        print(f'\r{progress_text}', end='', flush=True)
        
        # Print newline when complete
        if current >= total:
            print()
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_terminal_size(self) -> tuple:
        """Get terminal size (width, height)."""
        try:
            size = os.get_terminal_size()
            return size.columns, size.lines
        except OSError:
            return 80, 24  # Default size
    
    def truncate_text(self, text: str, max_length: int, suffix: str = '...') -> str:
        """Truncate text to fit within max_length.
        
        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add when truncating
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    def format_bytes(self, bytes_value: int) -> str:
        """Format bytes into human-readable string.
        
        Args:
            bytes_value: Number of bytes
            
        Returns:
            Formatted string (e.g., '1.5 MB')
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f} {unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f} PB"
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human-readable string.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted string (e.g., '1h 30m 45s')
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        
        minutes = int(seconds // 60)
        seconds = seconds % 60
        
        if minutes < 60:
            return f"{minutes}m {seconds:.0f}s"
        
        hours = minutes // 60
        minutes = minutes % 60
        
        if hours < 24:
            return f"{hours}h {minutes}m"
        
        days = hours // 24
        hours = hours % 24
        
        return f"{days}d {hours}h"
    
    def confirm(self, message: str, default: bool = False) -> bool:
        """Ask for user confirmation.
        
        Args:
            message: Confirmation message
            default: Default value if user just presses Enter
            
        Returns:
            True if user confirms, False otherwise
        """
        suffix = ' [Y/n]' if default else ' [y/N]'
        
        try:
            response = input(message + suffix + ' ').strip().lower()
            
            if not response:
                return default
            
            return response in ['y', 'yes', 'true', '1']
            
        except (KeyboardInterrupt, EOFError):
            print()  # New line
            return False
    
    def get_input(self, prompt: str, default: Optional[str] = None, 
                  validator: Optional[callable] = None) -> str:
        """Get user input with validation.
        
        Args:
            prompt: Input prompt
            default: Default value
            validator: Optional validation function
            
        Returns:
            User input string
        """
        full_prompt = prompt
        if default:
            full_prompt += f" [{default}]"
        full_prompt += ": "
        
        while True:
            try:
                user_input = input(full_prompt).strip()
                
                if not user_input and default:
                    user_input = default
                
                if validator:
                    if validator(user_input):
                        return user_input
                    else:
                        self.print_error("Invalid input. Please try again.")
                        continue
                
                return user_input
                
            except (KeyboardInterrupt, EOFError):
                print()  # New line
                raise
    
    def print_banner(self, title: str, width: Optional[int] = None):
        """Print a decorative banner.
        
        Args:
            title: Banner title
            width: Banner width (auto-detected if None)
        """
        if width is None:
            terminal_width, _ = self.get_terminal_size()
            width = min(terminal_width - 4, 80)
        
        # Ensure minimum width
        width = max(width, len(title) + 4)
        
        border = '═' * (width - 2)
        padding = ' ' * ((width - 2 - len(title)) // 2)
        
        banner = f"""
╔{border}╗
║{padding}{title}{padding}║
╚{border}╝
        """.strip()
        
        print(self.colorize(banner, 'cyan', bold=True))
    
    def print_section(self, title: str):
        """Print a section header.
        
        Args:
            title: Section title
        """
        print()
        print(self.colorize(f"── {title} ", 'blue', bold=True) + 
              self.colorize('─' * (50 - len(title)), 'dim'))
        print()


class ProgressTracker:
    """Track and display progress for long-running operations."""
    
    def __init__(self, total: int, description: str = ""):
        """Initialize progress tracker.
        
        Args:
            total: Total number of items to process
            description: Description of the operation
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
        self.utils = CLIUtils()
    
    def update(self, increment: int = 1):
        """Update progress.
        
        Args:
            increment: Amount to increment progress
        """
        self.current = min(self.current + increment, self.total)
        self._display_progress()
    
    def set_progress(self, current: int):
        """Set absolute progress value.
        
        Args:
            current: Current progress value
        """
        self.current = min(current, self.total)
        self._display_progress()
    
    def _display_progress(self):
        """Display current progress."""
        elapsed = datetime.now() - self.start_time
        elapsed_str = self.utils.format_duration(elapsed.total_seconds())
        
        if self.current > 0:
            eta_seconds = (elapsed.total_seconds() / self.current) * (self.total - self.current)
            eta_str = self.utils.format_duration(eta_seconds)
            suffix = f"({elapsed_str} elapsed, {eta_str} remaining)"
        else:
            suffix = f"({elapsed_str} elapsed)"
        
        self.utils.print_progress_bar(
            self.current, self.total, 
            prefix=self.description, 
            suffix=suffix
        )
    
    def finish(self):
        """Mark progress as complete."""
        self.current = self.total
        elapsed = datetime.now() - self.start_time
        elapsed_str = self.utils.format_duration(elapsed.total_seconds())
        
        self.utils.print_progress_bar(
            self.current, self.total,
            prefix=self.description,
            suffix=f"(completed in {elapsed_str})"
        )
        print()  # New line after completion