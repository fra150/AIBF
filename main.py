#!/usr/bin/env python3
"""
AI Bull Ford - Main Entry Point

A comprehensive modular AI framework with assembly line architecture for building
advanced AI systems with configurable pipelines, multimodal processing,
specialized applications, and robust API services.

Author: AI Bull Ford Team
Version: 1.0.0
"""

import sys
import os
import asyncio
import argparse
import logging
import signal
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Core framework imports
from assembly_line.pipeline import Pipeline
from assembly_line.module_registry import ModuleRegistry
from assembly_line.workflow_definition import WorkflowDefinition

# Configuration and monitoring
from monitoring.logging_config import setup_logging
from monitoring.metrics import MetricsCollector
from monitoring.health_check import HealthChecker

# API services
from api.rest import RESTServer
from api.websocket import WebSocketServer
from api.grpc import GRPCServer
from api.graphql import GraphQLServer

# Applications
from applications import initialize_applications, shutdown_applications

# Security
from security.authentication import AuthenticationManager
from security.authorization import AuthorizationManager

# Global logger
logger = logging.getLogger(__name__)


class AIBullFordFramework:
    """Main framework orchestrator class."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.module_registry = ModuleRegistry()
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker()
        self.auth_manager = AuthenticationManager(config.get("authentication", {}))
        self.authz_manager = AuthorizationManager(config.get("authorization", {}))
        
        # API servers
        self.rest_server: Optional[RESTServer] = None
        self.websocket_server: Optional[WebSocketServer] = None
        self.grpc_server: Optional[GRPCServer] = None
        self.graphql_server: Optional[GraphQLServer] = None
        
        # Framework state
        self.running = False
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    async def initialize(self) -> bool:
        """Initialize all framework components."""
        try:
            logger.info("Initializing AI Bull Ford Framework...")
            
            # Initialize applications
            logger.info("Initializing specialized applications...")
            app_success = await initialize_applications()
            if not app_success:
                logger.error("Failed to initialize applications")
                return False
            
            # Initialize monitoring
            logger.info("Starting monitoring services...")
            await self.metrics_collector.start()
            await self.health_checker.start()
            
            # Initialize API servers based on configuration
            api_config = self.config.get("api", {})
            
            if api_config.get("rest", {}).get("enabled", True):
                logger.info("Initializing REST API server...")
                self.rest_server = RESTServer(
                    config=api_config.get("rest", {}),
                    auth_manager=self.auth_manager,
                    authz_manager=self.authz_manager
                )
                await self.rest_server.initialize()
            
            if api_config.get("websocket", {}).get("enabled", False):
                logger.info("Initializing WebSocket server...")
                self.websocket_server = WebSocketServer(
                    config=api_config.get("websocket", {}),
                    auth_manager=self.auth_manager
                )
                await self.websocket_server.initialize()
            
            if api_config.get("grpc", {}).get("enabled", False):
                logger.info("Initializing gRPC server...")
                self.grpc_server = GRPCServer(
                    config=api_config.get("grpc", {}),
                    auth_manager=self.auth_manager
                )
                await self.grpc_server.initialize()
            
            if api_config.get("graphql", {}).get("enabled", False):
                logger.info("Initializing GraphQL server...")
                self.graphql_server = GraphQLServer(
                    config=api_config.get("graphql", {}),
                    auth_manager=self.auth_manager
                )
                await self.graphql_server.initialize()
            
            logger.info("AI Bull Ford Framework initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize framework: {e}")
            return False
    
    async def start_servers(self):
        """Start all configured API servers."""
        tasks = []
        
        if self.rest_server:
            logger.info("Starting REST API server...")
            tasks.append(asyncio.create_task(self.rest_server.start()))
        
        if self.websocket_server:
            logger.info("Starting WebSocket server...")
            tasks.append(asyncio.create_task(self.websocket_server.start()))
        
        if self.grpc_server:
            logger.info("Starting gRPC server...")
            tasks.append(asyncio.create_task(self.grpc_server.start()))
        
        if self.graphql_server:
            logger.info("Starting GraphQL server...")
            tasks.append(asyncio.create_task(self.graphql_server.start()))
        
        if tasks:
            # Wait for all servers to start
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("All API servers started successfully")
        else:
            logger.info("No API servers configured to start")
    
    async def run_pipeline(self, pipeline_config: Dict[str, Any]):
        """Run a specific pipeline configuration."""
        try:
            logger.info("Starting pipeline execution...")
            
            # Create workflow definition from config
            workflow_def = WorkflowDefinition.from_config(pipeline_config)
            
            # Create and execute pipeline
            pipeline = Pipeline.from_workflow_definition(workflow_def, self.module_registry)
            
            # Execute pipeline with input data
            input_data = pipeline_config.get("input_data", {})
            result = await pipeline.process_async(input_data)
            
            logger.info(f"Pipeline execution completed successfully: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    async def run_server_mode(self):
        """Run in server mode with API endpoints."""
        try:
            # Start all configured servers
            await self.start_servers()
            
            self.running = True
            logger.info("AI Bull Ford Framework is running in server mode")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Server mode execution failed: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown all framework components."""
        logger.info("Shutting down AI Bull Ford Framework...")
        
        try:
            # Stop API servers
            shutdown_tasks = []
            
            if self.rest_server:
                logger.info("Stopping REST API server...")
                shutdown_tasks.append(asyncio.create_task(self.rest_server.stop()))
            
            if self.websocket_server:
                logger.info("Stopping WebSocket server...")
                shutdown_tasks.append(asyncio.create_task(self.websocket_server.stop()))
            
            if self.grpc_server:
                logger.info("Stopping gRPC server...")
                shutdown_tasks.append(asyncio.create_task(self.grpc_server.stop()))
            
            if self.graphql_server:
                logger.info("Stopping GraphQL server...")
                shutdown_tasks.append(asyncio.create_task(self.graphql_server.stop()))
            
            if shutdown_tasks:
                await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
            # Stop monitoring services
            logger.info("Stopping monitoring services...")
            await self.metrics_collector.stop()
            await self.health_checker.stop()
            
            # Shutdown applications
            logger.info("Shutting down specialized applications...")
            shutdown_applications()
            
            self.running = False
            logger.info("AI Bull Ford Framework shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


def load_configuration(config_path: str) -> Dict[str, Any]:
    """Load framework configuration from file."""
    import json
    import yaml
    
    config_file = Path(config_path)
    
    if not config_file.exists():
        logger.warning(f"Configuration file {config_path} not found, using defaults")
        return get_default_configuration()
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            if config_file.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_file.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_file.suffix}")
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        logger.info("Using default configuration")
        return get_default_configuration()


def get_default_configuration() -> Dict[str, Any]:
    """Get default framework configuration."""
    return {
        "framework": {
            "name": "AI Bull Ford",
            "version": "1.0.0",
            "mode": "development"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "logs/aibf.log"
        },
        "api": {
            "rest": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": 8080,
                "workers": 4
            },
            "websocket": {
                "enabled": False,
                "host": "0.0.0.0",
                "port": 8081
            },
            "grpc": {
                "enabled": False,
                "host": "0.0.0.0",
                "port": 8082
            },
            "graphql": {
                "enabled": False,
                "host": "0.0.0.0",
                "port": 8083
            }
        },
        "authentication": {
            "enabled": True,
            "method": "jwt",
            "secret_key": "your-secret-key-here",
            "token_expiry": 3600
        },
        "authorization": {
            "enabled": True,
            "default_role": "user",
            "admin_role": "admin"
        },
        "monitoring": {
            "metrics": {
                "enabled": True,
                "collection_interval": 60
            },
            "health_check": {
                "enabled": True,
                "check_interval": 30
            }
        },
        "applications": {
            "healthcare": {
                "enabled": True
            },
            "financial": {
                "enabled": True
            },
            "educational": {
                "enabled": True
            }
        }
    }


def main():
    """Main entry point for AI Bull Ford framework."""
    parser = argparse.ArgumentParser(
        description="AI Bull Ford - Comprehensive Modular AI Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Execution Modes:
  pipeline    - Execute a specific pipeline configuration
  server      - Run as API server with all configured endpoints
  interactive - Start interactive CLI mode
  test        - Run framework tests

Examples:
  python main.py --mode server --config config/production.yaml
  python main.py --mode pipeline --config config/pipelines/healthcare.yaml
  python main.py --mode interactive --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="server",
        choices=["pipeline", "server", "interactive", "test"],
        help="Execution mode"
    )
    parser.add_argument(
        "--pipeline-config",
        type=str,
        help="Path to pipeline-specific configuration (for pipeline mode)"
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Override API server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Override API server port"
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Number of worker processes (for server mode)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Load configuration
    config = load_configuration(args.config)
    
    # Apply command line overrides
    if args.host:
        config.setdefault("api", {}).setdefault("rest", {})["host"] = args.host
    if args.port:
        config.setdefault("api", {}).setdefault("rest", {})["port"] = args.port
    if args.workers:
        config.setdefault("api", {}).setdefault("rest", {})["workers"] = args.workers
    
    # Override logging level if specified
    if args.log_level:
        config.setdefault("logging", {})["level"] = args.log_level
    
    try:
        if args.mode == "pipeline":
            # Pipeline execution mode
            pipeline_config_path = args.pipeline_config or args.config
            pipeline_config = load_configuration(pipeline_config_path)
            
            # Run pipeline synchronously
            async def run_pipeline_mode():
                framework = AIBullFordFramework(config)
                success = await framework.initialize()
                if not success:
                    logger.error("Failed to initialize framework")
                    return False
                
                result = await framework.run_pipeline(pipeline_config)
                logger.info(f"Pipeline execution result: {result}")
                await framework.shutdown()
                return True
            
            # Run the async pipeline
            success = asyncio.run(run_pipeline_mode())
            if not success:
                exit(1)
        
        elif args.mode == "server":
            # API server mode
            async def run_server_mode():
                framework = AIBullFordFramework(config)
                success = await framework.initialize()
                if not success:
                    logger.error("Failed to initialize framework")
                    return False
                
                await framework.run_server_mode()
                return True
            
            # Run the async server
            success = asyncio.run(run_server_mode())
            if not success:
                exit(1)
        
        elif args.mode == "interactive":
            # Interactive CLI mode
            logger.info("Starting interactive mode...")
            
            async def run_interactive_mode():
                framework = AIBullFordFramework(config)
                success = await framework.initialize()
                if not success:
                    logger.error("Failed to initialize framework")
                    return False
                
                # Start interactive CLI
                from src.cli import InteractiveCLI
                cli = InteractiveCLI(framework)
                await cli.start()
                
                await framework.shutdown()
                return True
            
            try:
                success = asyncio.run(run_interactive_mode())
                if not success:
                    exit(1)
            except KeyboardInterrupt:
                logger.info("Interactive mode interrupted by user")
        
        elif args.mode == "test":
            # Test execution mode
            logger.info("Running framework tests...")
            
            import subprocess
            import sys
            
            # Run pytest with coverage
            test_cmd = [
                sys.executable, "-m", "pytest",
                "tests/",
                "-v",
                "--cov=src",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--tb=short"
            ]
            
            result = subprocess.run(test_cmd, cwd=Path.cwd())
            if result.returncode != 0:
                logger.error("Tests failed")
                exit(1)
            else:
                logger.info("All tests passed successfully")
        
        else:
            logger.error(f"Unknown execution mode: {args.mode}")
            exit(1)
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application failed with error: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()