"""CLI Commands for AI Bull Ford Framework.

Implements all available CLI commands for framework management,
pipeline execution, and monitoring.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class CLICommands:
    """Implements CLI commands for the AIBF framework."""
    
    def __init__(self, framework):
        """Initialize CLI commands.
        
        Args:
            framework: The AIBullFordFramework instance
        """
        self.framework = framework
        self.command_registry = self._build_command_registry()
    
    def _build_command_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build the command registry with all available commands."""
        return {
            # Framework commands
            'status': {
                'handler': self._cmd_status,
                'description': 'Show framework status and health',
                'usage': 'status',
                'examples': ['status']
            },
            'config': {
                'handler': self._cmd_config,
                'description': 'Display current configuration',
                'usage': 'config [section]',
                'examples': ['config', 'config api', 'config logging']
            },
            'modules': {
                'handler': self._cmd_modules,
                'description': 'List available modules',
                'usage': 'modules [filter]',
                'examples': ['modules', 'modules core', 'modules api']
            },
            'metrics': {
                'handler': self._cmd_metrics,
                'description': 'Show performance metrics',
                'usage': 'metrics [type]',
                'examples': ['metrics', 'metrics cpu', 'metrics memory']
            },
            'logs': {
                'handler': self._cmd_logs,
                'description': 'View framework logs',
                'usage': 'logs [level] [lines]',
                'examples': ['logs', 'logs error', 'logs info 50']
            },
            
            # Pipeline commands
            'run': {
                'handler': self._cmd_run_pipeline,
                'description': 'Execute a pipeline',
                'usage': 'run <pipeline_name> [config_file]',
                'examples': ['run healthcare_pipeline', 'run custom_pipeline config/custom.yaml']
            },
            'pipelines': {
                'handler': self._cmd_list_pipelines,
                'description': 'List available pipelines',
                'usage': 'pipelines',
                'examples': ['pipelines']
            },
            'create': {
                'handler': self._cmd_create_pipeline,
                'description': 'Create a new pipeline',
                'usage': 'create <pipeline_name> <template>',
                'examples': ['create my_pipeline basic', 'create ml_pipeline advanced']
            },
            'stop': {
                'handler': self._cmd_stop_pipeline,
                'description': 'Stop a running pipeline',
                'usage': 'stop <pipeline_id>',
                'examples': ['stop pipeline_123', 'stop all']
            },
            
            # Server commands
            'start': {
                'handler': self._cmd_start_service,
                'description': 'Start API service',
                'usage': 'start <service>',
                'examples': ['start rest', 'start websocket', 'start all']
            },
            'restart': {
                'handler': self._cmd_restart_service,
                'description': 'Restart API service',
                'usage': 'restart <service>',
                'examples': ['restart rest', 'restart all']
            },
            'services': {
                'handler': self._cmd_list_services,
                'description': 'List running services',
                'usage': 'services',
                'examples': ['services']
            },
            
            # Model commands
            'models': {
                'handler': self._cmd_list_models,
                'description': 'List available AI models',
                'usage': 'models [category]',
                'examples': ['models', 'models healthcare', 'models nlp']
            },
            'load': {
                'handler': self._cmd_load_model,
                'description': 'Load an AI model',
                'usage': 'load <model_name> [model_path]',
                'examples': ['load bert_model', 'load custom_model /path/to/model']
            },
            'unload': {
                'handler': self._cmd_unload_model,
                'description': 'Unload an AI model',
                'usage': 'unload <model_name>',
                'examples': ['unload bert_model', 'unload all']
            },
            
            # Utility commands
            'test': {
                'handler': self._cmd_test,
                'description': 'Run framework tests',
                'usage': 'test [test_type] [module]',
                'examples': ['test', 'test unit', 'test integration core']
            },
            'benchmark': {
                'handler': self._cmd_benchmark,
                'description': 'Run performance benchmarks',
                'usage': 'benchmark [component]',
                'examples': ['benchmark', 'benchmark neural_networks', 'benchmark api']
            },
            'export': {
                'handler': self._cmd_export,
                'description': 'Export framework data',
                'usage': 'export <type> <output_file>',
                'examples': ['export config config.json', 'export metrics metrics.csv']
            }
        }
    
    def get_available_commands(self) -> Dict[str, str]:
        """Get a dictionary of available commands and their descriptions."""
        return {cmd: info['description'] for cmd, info in self.command_registry.items()}
    
    def get_command_help(self, command: str) -> Optional[str]:
        """Get detailed help for a specific command."""
        if command not in self.command_registry:
            return None
        
        cmd_info = self.command_registry[command]
        help_text = f"""
Command: {command}
Description: {cmd_info['description']}
Usage: {cmd_info['usage']}

Examples:
"""
        for example in cmd_info['examples']:
            help_text += f"  {example}\n"
        
        return help_text
    
    async def execute_command(self, command: str, args: List[str]) -> Any:
        """Execute a CLI command.
        
        Args:
            command: The command name
            args: Command arguments
            
        Returns:
            Command execution result
        """
        if command not in self.command_registry:
            raise ValueError(f"Unknown command: {command}")
        
        handler = self.command_registry[command]['handler']
        
        try:
            # Execute the command handler
            if asyncio.iscoroutinefunction(handler):
                result = await handler(args)
            else:
                result = handler(args)
            
            return result
            
        except Exception as e:
            logger.error(f"Command '{command}' failed: {e}")
            raise
    
    # Framework commands
    async def _cmd_status(self, args: List[str]) -> Dict[str, Any]:
        """Show framework status."""
        status = {
            'framework': {
                'name': 'AI Bull Ford',
                'version': '1.0.0',
                'running': self.framework.running,
                'uptime': self._get_uptime()
            },
            'services': await self._get_services_status(),
            'health': await self._get_health_status(),
            'resources': await self._get_resource_usage()
        }
        return status
    
    def _cmd_config(self, args: List[str]) -> Dict[str, Any]:
        """Display current configuration."""
        config = self.framework.config
        
        if args:
            # Show specific section
            section = args[0]
            if section in config:
                return {section: config[section]}
            else:
                return {'error': f'Configuration section "{section}" not found'}
        
        return config
    
    def _cmd_modules(self, args: List[str]) -> List[Dict[str, Any]]:
        """List available modules."""
        modules = []
        
        # Get registered modules from module registry
        if hasattr(self.framework, 'module_registry'):
            registered_modules = self.framework.module_registry.list_modules()
            for module_name, module_info in registered_modules.items():
                modules.append({
                    'name': module_name,
                    'type': module_info.get('type', 'unknown'),
                    'status': module_info.get('status', 'unknown'),
                    'description': module_info.get('description', '')
                })
        
        # Filter if requested
        if args:
            filter_term = args[0].lower()
            modules = [m for m in modules if filter_term in m['name'].lower() or filter_term in m['type'].lower()]
        
        return modules
    
    async def _cmd_metrics(self, args: List[str]) -> Dict[str, Any]:
        """Show performance metrics."""
        metrics = {}
        
        if hasattr(self.framework, 'metrics_collector'):
            try:
                all_metrics = await self.framework.metrics_collector.get_metrics()
                
                if args:
                    # Show specific metric type
                    metric_type = args[0]
                    if metric_type in all_metrics:
                        metrics = {metric_type: all_metrics[metric_type]}
                    else:
                        metrics = {'error': f'Metric type "{metric_type}" not found'}
                else:
                    metrics = all_metrics
                    
            except Exception as e:
                metrics = {'error': f'Failed to retrieve metrics: {e}'}
        else:
            metrics = {'error': 'Metrics collector not available'}
        
        return metrics
    
    def _cmd_logs(self, args: List[str]) -> List[str]:
        """View framework logs."""
        log_level = 'INFO'
        max_lines = 100
        
        if args:
            if args[0].upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                log_level = args[0].upper()
                if len(args) > 1 and args[1].isdigit():
                    max_lines = int(args[1])
            elif args[0].isdigit():
                max_lines = int(args[0])
        
        # Read logs from file or memory handler
        logs = self._read_logs(log_level, max_lines)
        return logs
    
    # Pipeline commands
    async def _cmd_run_pipeline(self, args: List[str]) -> Dict[str, Any]:
        """Execute a pipeline."""
        if not args:
            return {'error': 'Pipeline name required'}
        
        pipeline_name = args[0]
        config_file = args[1] if len(args) > 1 else None
        
        try:
            # Load pipeline configuration
            if config_file:
                from main import load_configuration
                pipeline_config = load_configuration(config_file)
            else:
                # Use default pipeline configuration
                pipeline_config = self._get_default_pipeline_config(pipeline_name)
            
            # Execute pipeline
            result = await self.framework.run_pipeline(pipeline_config)
            
            return {
                'pipeline': pipeline_name,
                'status': 'completed',
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'pipeline': pipeline_name,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _cmd_list_pipelines(self, args: List[str]) -> List[Dict[str, Any]]:
        """List available pipelines."""
        pipelines = []
        
        # Scan for pipeline configurations
        config_dir = Path('config/pipelines')
        if config_dir.exists():
            for config_file in config_dir.glob('*.yaml'):
                pipelines.append({
                    'name': config_file.stem,
                    'file': str(config_file),
                    'modified': datetime.fromtimestamp(config_file.stat().st_mtime).isoformat()
                })
        
        return pipelines
    
    def _cmd_create_pipeline(self, args: List[str]) -> Dict[str, Any]:
        """Create a new pipeline."""
        if len(args) < 2:
            return {'error': 'Pipeline name and template required'}
        
        pipeline_name = args[0]
        template = args[1]
        
        # Create pipeline configuration from template
        config = self._create_pipeline_from_template(pipeline_name, template)
        
        # Save configuration file
        config_dir = Path('config/pipelines')
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = config_dir / f'{pipeline_name}.yaml'
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return {
            'pipeline': pipeline_name,
            'template': template,
            'config_file': str(config_file),
            'status': 'created'
        }
    
    def _cmd_stop_pipeline(self, args: List[str]) -> Dict[str, Any]:
        """Stop a running pipeline."""
        if not args:
            return {'error': 'Pipeline ID required'}
        
        pipeline_id = args[0]
        
        # TODO: Implement pipeline stopping logic
        return {
            'pipeline_id': pipeline_id,
            'status': 'stopped',
            'message': 'Pipeline stopping not yet implemented'
        }
    
    # Service commands
    async def _cmd_start_service(self, args: List[str]) -> Dict[str, Any]:
        """Start API service."""
        if not args:
            return {'error': 'Service name required'}
        
        service = args[0]
        
        if service == 'all':
            await self.framework.start_servers()
            return {'status': 'started', 'services': 'all'}
        
        # Start specific service
        result = await self._start_specific_service(service)
        return result
    
    async def _cmd_restart_service(self, args: List[str]) -> Dict[str, Any]:
        """Restart API service."""
        if not args:
            return {'error': 'Service name required'}
        
        service = args[0]
        
        # TODO: Implement service restart logic
        return {
            'service': service,
            'status': 'restarted',
            'message': 'Service restart not yet implemented'
        }
    
    async def _cmd_list_services(self, args: List[str]) -> List[Dict[str, Any]]:
        """List running services."""
        services = []
        
        # Check each service type
        if hasattr(self.framework, 'rest_server') and self.framework.rest_server:
            services.append({
                'name': 'REST API',
                'type': 'rest',
                'status': 'running' if self.framework.rest_server else 'stopped',
                'port': self.framework.config.get('api', {}).get('rest', {}).get('port', 8080)
            })
        
        if hasattr(self.framework, 'websocket_server') and self.framework.websocket_server:
            services.append({
                'name': 'WebSocket',
                'type': 'websocket',
                'status': 'running' if self.framework.websocket_server else 'stopped',
                'port': self.framework.config.get('api', {}).get('websocket', {}).get('port', 8081)
            })
        
        return services
    
    # Model commands
    def _cmd_list_models(self, args: List[str]) -> List[Dict[str, Any]]:
        """List available AI models."""
        models = []
        
        # Scan models directory
        models_dir = Path('models')
        if models_dir.exists():
            for model_file in models_dir.rglob('*.pt'):
                category = model_file.parent.name if model_file.parent != models_dir else 'general'
                models.append({
                    'name': model_file.stem,
                    'category': category,
                    'file': str(model_file),
                    'size': f"{model_file.stat().st_size / (1024*1024):.1f} MB"
                })
        
        # Filter by category if specified
        if args:
            category_filter = args[0].lower()
            models = [m for m in models if category_filter in m['category'].lower()]
        
        return models
    
    def _cmd_load_model(self, args: List[str]) -> Dict[str, Any]:
        """Load an AI model."""
        if not args:
            return {'error': 'Model name required'}
        
        model_name = args[0]
        model_path = args[1] if len(args) > 1 else None
        
        # TODO: Implement model loading logic
        return {
            'model': model_name,
            'path': model_path,
            'status': 'loaded',
            'message': 'Model loading not yet implemented'
        }
    
    def _cmd_unload_model(self, args: List[str]) -> Dict[str, Any]:
        """Unload an AI model."""
        if not args:
            return {'error': 'Model name required'}
        
        model_name = args[0]
        
        # TODO: Implement model unloading logic
        return {
            'model': model_name,
            'status': 'unloaded',
            'message': 'Model unloading not yet implemented'
        }
    
    # Utility commands
    async def _cmd_test(self, args: List[str]) -> Dict[str, Any]:
        """Run framework tests."""
        test_type = args[0] if args else 'all'
        module = args[1] if len(args) > 1 else None
        
        # TODO: Implement test execution
        return {
            'test_type': test_type,
            'module': module,
            'status': 'completed',
            'message': 'Test execution not yet implemented'
        }
    
    async def _cmd_benchmark(self, args: List[str]) -> Dict[str, Any]:
        """Run performance benchmarks."""
        component = args[0] if args else 'all'
        
        # TODO: Implement benchmarking
        return {
            'component': component,
            'status': 'completed',
            'message': 'Benchmarking not yet implemented'
        }
    
    def _cmd_export(self, args: List[str]) -> Dict[str, Any]:
        """Export framework data."""
        if len(args) < 2:
            return {'error': 'Export type and output file required'}
        
        export_type = args[0]
        output_file = args[1]
        
        try:
            if export_type == 'config':
                # Export configuration
                with open(output_file, 'w') as f:
                    json.dump(self.framework.config, f, indent=2)
            
            elif export_type == 'metrics':
                # Export metrics (placeholder)
                with open(output_file, 'w') as f:
                    f.write('timestamp,metric,value\n')
                    f.write(f'{datetime.now().isoformat()},placeholder,0\n')
            
            else:
                return {'error': f'Unknown export type: {export_type}'}
            
            return {
                'type': export_type,
                'file': output_file,
                'status': 'exported'
            }
            
        except Exception as e:
            return {'error': f'Export failed: {e}'}
    
    # Helper methods
    def _get_uptime(self) -> str:
        """Get framework uptime."""
        # TODO: Implement uptime calculation
        return "0:00:00"
    
    async def _get_services_status(self) -> Dict[str, str]:
        """Get status of all services."""
        status = {}
        
        if hasattr(self.framework, 'rest_server'):
            status['rest'] = 'running' if self.framework.rest_server else 'stopped'
        
        if hasattr(self.framework, 'websocket_server'):
            status['websocket'] = 'running' if self.framework.websocket_server else 'stopped'
        
        return status
    
    async def _get_health_status(self) -> Dict[str, Any]:
        """Get health status."""
        if hasattr(self.framework, 'health_checker'):
            try:
                return await self.framework.health_checker.get_health_status()
            except Exception:
                return {'status': 'unknown'}
        
        return {'status': 'healthy'}
    
    async def _get_resource_usage(self) -> Dict[str, Any]:
        """Get resource usage information."""
        import psutil
        
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent
        }
    
    def _read_logs(self, level: str, max_lines: int) -> List[str]:
        """Read logs from file."""
        logs = []
        
        log_file = Path('logs/aibf.log')
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    # Filter by level and return last max_lines
                    filtered_lines = [line.strip() for line in lines if level in line]
                    logs = filtered_lines[-max_lines:] if filtered_lines else []
            except Exception as e:
                logs = [f'Error reading logs: {e}']
        else:
            logs = ['Log file not found']
        
        return logs
    
    def _get_default_pipeline_config(self, pipeline_name: str) -> Dict[str, Any]:
        """Get default configuration for a pipeline."""
        return {
            'name': pipeline_name,
            'description': f'Default configuration for {pipeline_name}',
            'modules': [],
            'input_data': {},
            'output_format': 'json'
        }
    
    def _create_pipeline_from_template(self, name: str, template: str) -> Dict[str, Any]:
        """Create pipeline configuration from template."""
        templates = {
            'basic': {
                'name': name,
                'description': f'Basic pipeline: {name}',
                'modules': [
                    {'type': 'data_loader', 'config': {}},
                    {'type': 'processor', 'config': {}},
                    {'type': 'output', 'config': {}}
                ]
            },
            'advanced': {
                'name': name,
                'description': f'Advanced pipeline: {name}',
                'modules': [
                    {'type': 'data_loader', 'config': {}},
                    {'type': 'preprocessor', 'config': {}},
                    {'type': 'neural_network', 'config': {}},
                    {'type': 'postprocessor', 'config': {}},
                    {'type': 'output', 'config': {}}
                ]
            }
        }
        
        return templates.get(template, templates['basic'])
    
    async def _start_specific_service(self, service: str) -> Dict[str, Any]:
        """Start a specific service."""
        try:
            if service == 'rest' and hasattr(self.framework, 'rest_server'):
                if not self.framework.rest_server:
                    # Initialize REST server
                    from src.api.rest import RESTServer
                    api_config = self.framework.config.get('api', {})
                    self.framework.rest_server = RESTServer(
                        config=api_config.get('rest', {}),
                        auth_manager=self.framework.auth_manager,
                        authz_manager=self.framework.authz_manager
                    )
                    await self.framework.rest_server.initialize()
                
                await self.framework.rest_server.start()
                return {'service': service, 'status': 'started'}
            
            else:
                return {'error': f'Unknown or unsupported service: {service}'}
                
        except Exception as e:
            return {'error': f'Failed to start {service}: {e}'}