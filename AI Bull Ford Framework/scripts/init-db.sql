-- AI Bull Ford (AIBF) - Database Initialization Script
-- PostgreSQL database setup and initial schema

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS aibf_core;
CREATE SCHEMA IF NOT EXISTS aibf_auth;
CREATE SCHEMA IF NOT EXISTS aibf_monitoring;
CREATE SCHEMA IF NOT EXISTS aibf_analytics;
CREATE SCHEMA IF NOT EXISTS aibf_ml;

-- Set search path
SET search_path TO aibf_core, aibf_auth, aibf_monitoring, aibf_analytics, aibf_ml, public;

-- Core tables

-- Users table
CREATE TABLE IF NOT EXISTS aibf_auth.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    is_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

-- API Keys table
CREATE TABLE IF NOT EXISTS aibf_auth.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES aibf_auth.users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    permissions JSONB DEFAULT '[]',
    is_active BOOLEAN DEFAULT true,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP WITH TIME ZONE
);

-- Sessions table
CREATE TABLE IF NOT EXISTS aibf_auth.sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES aibf_auth.users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Models table
CREATE TABLE IF NOT EXISTS aibf_ml.models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    model_type VARCHAR(100) NOT NULL,
    framework VARCHAR(50) NOT NULL,
    description TEXT,
    config JSONB DEFAULT '{}',
    metrics JSONB DEFAULT '{}',
    artifacts JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'created',
    created_by UUID REFERENCES aibf_auth.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, version)
);

-- Training Jobs table
CREATE TABLE IF NOT EXISTS aibf_ml.training_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID REFERENCES aibf_ml.models(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    progress FLOAT DEFAULT 0.0,
    metrics JSONB DEFAULT '{}',
    logs TEXT,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by UUID REFERENCES aibf_auth.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Datasets table
CREATE TABLE IF NOT EXISTS aibf_ml.datasets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    dataset_type VARCHAR(100) NOT NULL,
    format VARCHAR(50) NOT NULL,
    size_bytes BIGINT,
    num_samples INTEGER,
    schema JSONB,
    metadata JSONB DEFAULT '{}',
    storage_path TEXT,
    checksum VARCHAR(255),
    created_by UUID REFERENCES aibf_auth.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Experiments table
CREATE TABLE IF NOT EXISTS aibf_ml.experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'created',
    metrics JSONB DEFAULT '{}',
    artifacts JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    created_by UUID REFERENCES aibf_auth.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Monitoring tables

-- System Metrics table
CREATE TABLE IF NOT EXISTS aibf_monitoring.system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    unit VARCHAR(50),
    tags JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    host VARCHAR(255),
    service VARCHAR(255)
);

-- Application Logs table
CREATE TABLE IF NOT EXISTS aibf_monitoring.application_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    logger VARCHAR(255),
    module VARCHAR(255),
    function VARCHAR(255),
    line_number INTEGER,
    user_id UUID REFERENCES aibf_auth.users(id),
    request_id UUID,
    session_id UUID,
    extra JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    host VARCHAR(255),
    service VARCHAR(255)
);

-- Audit Logs table
CREATE TABLE IF NOT EXISTS aibf_monitoring.audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES aibf_auth.users(id),
    action VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN DEFAULT true,
    error_message TEXT
);

-- Analytics tables

-- Events table
CREATE TABLE IF NOT EXISTS aibf_analytics.events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(255) NOT NULL,
    event_name VARCHAR(255) NOT NULL,
    user_id UUID REFERENCES aibf_auth.users(id),
    session_id UUID,
    properties JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT,
    referrer TEXT,
    page_url TEXT
);

-- Performance Metrics table
CREATE TABLE IF NOT EXISTS aibf_analytics.performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(255) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER NOT NULL,
    response_time_ms DOUBLE PRECISION NOT NULL,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    user_id UUID REFERENCES aibf_auth.users(id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Core application tables

-- Configurations table
CREATE TABLE IF NOT EXISTS aibf_core.configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    key VARCHAR(255) UNIQUE NOT NULL,
    value JSONB NOT NULL,
    description TEXT,
    is_secret BOOLEAN DEFAULT false,
    created_by UUID REFERENCES aibf_auth.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tasks table (for async task management)
CREATE TABLE IF NOT EXISTS aibf_core.tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    config JSONB DEFAULT '{}',
    result JSONB,
    error_message TEXT,
    progress FLOAT DEFAULT 0.0,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by UUID REFERENCES aibf_auth.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Notifications table
CREATE TABLE IF NOT EXISTS aibf_core.notifications (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES aibf_auth.users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    notification_type VARCHAR(50) DEFAULT 'info',
    is_read BOOLEAN DEFAULT false,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    read_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for better performance

-- Users indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON aibf_auth.users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON aibf_auth.users(email);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON aibf_auth.users(created_at);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON aibf_auth.users(is_active);

-- API Keys indexes
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON aibf_auth.api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON aibf_auth.api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_is_active ON aibf_auth.api_keys(is_active);

-- Sessions indexes
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON aibf_auth.sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON aibf_auth.sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON aibf_auth.sessions(expires_at);

-- Models indexes
CREATE INDEX IF NOT EXISTS idx_models_name ON aibf_ml.models(name);
CREATE INDEX IF NOT EXISTS idx_models_type ON aibf_ml.models(model_type);
CREATE INDEX IF NOT EXISTS idx_models_status ON aibf_ml.models(status);
CREATE INDEX IF NOT EXISTS idx_models_created_at ON aibf_ml.models(created_at);

-- Training Jobs indexes
CREATE INDEX IF NOT EXISTS idx_training_jobs_model_id ON aibf_ml.training_jobs(model_id);
CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON aibf_ml.training_jobs(status);
CREATE INDEX IF NOT EXISTS idx_training_jobs_created_at ON aibf_ml.training_jobs(created_at);

-- System Metrics indexes
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON aibf_monitoring.system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON aibf_monitoring.system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_host ON aibf_monitoring.system_metrics(host);
CREATE INDEX IF NOT EXISTS idx_system_metrics_service ON aibf_monitoring.system_metrics(service);

-- Application Logs indexes
CREATE INDEX IF NOT EXISTS idx_application_logs_level ON aibf_monitoring.application_logs(level);
CREATE INDEX IF NOT EXISTS idx_application_logs_timestamp ON aibf_monitoring.application_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_application_logs_user_id ON aibf_monitoring.application_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_application_logs_request_id ON aibf_monitoring.application_logs(request_id);

-- Events indexes
CREATE INDEX IF NOT EXISTS idx_events_type ON aibf_analytics.events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_name ON aibf_analytics.events(event_name);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON aibf_analytics.events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_user_id ON aibf_analytics.events(user_id);

-- Performance Metrics indexes
CREATE INDEX IF NOT EXISTS idx_performance_metrics_endpoint ON aibf_analytics.performance_metrics(endpoint);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON aibf_analytics.performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_status_code ON aibf_analytics.performance_metrics(status_code);

-- Tasks indexes
CREATE INDEX IF NOT EXISTS idx_tasks_status ON aibf_core.tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_type ON aibf_core.tasks(task_type);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON aibf_core.tasks(priority);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON aibf_core.tasks(created_at);

-- Notifications indexes
CREATE INDEX IF NOT EXISTS idx_notifications_user_id ON aibf_core.notifications(user_id);
CREATE INDEX IF NOT EXISTS idx_notifications_is_read ON aibf_core.notifications(is_read);
CREATE INDEX IF NOT EXISTS idx_notifications_created_at ON aibf_core.notifications(created_at);

-- Create triggers for updated_at timestamps

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers to tables with updated_at columns
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON aibf_auth.users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON aibf_ml.models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_datasets_updated_at BEFORE UPDATE ON aibf_ml.datasets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_experiments_updated_at BEFORE UPDATE ON aibf_ml.experiments
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_configurations_updated_at BEFORE UPDATE ON aibf_core.configurations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_tasks_updated_at BEFORE UPDATE ON aibf_core.tasks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default configurations
INSERT INTO aibf_core.configurations (key, value, description) VALUES
('app.name', '"AI Bull Ford"', 'Application name'),
('app.version', '"0.1.0"', 'Application version'),
('app.environment', '"development"', 'Application environment'),
('security.session_timeout', '3600', 'Session timeout in seconds'),
('security.max_login_attempts', '5', 'Maximum login attempts before lockout'),
('ml.default_model_timeout', '300', 'Default model inference timeout in seconds'),
('monitoring.metrics_retention_days', '30', 'Metrics retention period in days'),
('analytics.event_batch_size', '100', 'Event processing batch size')
ON CONFLICT (key) DO NOTHING;

-- Create default admin user (password: admin123)
INSERT INTO aibf_auth.users (username, email, password_hash, first_name, last_name, is_superuser, is_verified)
VALUES (
    'admin',
    'admin@aibf.ai',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3QJX9VQvyG',  -- admin123
    'Admin',
    'User',
    true,
    true
)
ON CONFLICT (username) DO NOTHING;

-- Grant permissions
GRANT USAGE ON SCHEMA aibf_core TO aibf;
GRANT USAGE ON SCHEMA aibf_auth TO aibf;
GRANT USAGE ON SCHEMA aibf_monitoring TO aibf;
GRANT USAGE ON SCHEMA aibf_analytics TO aibf;
GRANT USAGE ON SCHEMA aibf_ml TO aibf;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA aibf_core TO aibf;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA aibf_auth TO aibf;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA aibf_monitoring TO aibf;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA aibf_analytics TO aibf;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA aibf_ml TO aibf;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA aibf_core TO aibf;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA aibf_auth TO aibf;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA aibf_monitoring TO aibf;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA aibf_analytics TO aibf;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA aibf_ml TO aibf;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA aibf_core GRANT ALL ON TABLES TO aibf;
ALTER DEFAULT PRIVILEGES IN SCHEMA aibf_auth GRANT ALL ON TABLES TO aibf;
ALTER DEFAULT PRIVILEGES IN SCHEMA aibf_monitoring GRANT ALL ON TABLES TO aibf;
ALTER DEFAULT PRIVILEGES IN SCHEMA aibf_analytics GRANT ALL ON TABLES TO aibf;
ALTER DEFAULT PRIVILEGES IN SCHEMA aibf_ml GRANT ALL ON TABLES TO aibf;

ALTER DEFAULT PRIVILEGES IN SCHEMA aibf_core GRANT ALL ON SEQUENCES TO aibf;
ALTER DEFAULT PRIVILEGES IN SCHEMA aibf_auth GRANT ALL ON SEQUENCES TO aibf;
ALTER DEFAULT PRIVILEGES IN SCHEMA aibf_monitoring GRANT ALL ON SEQUENCES TO aibf;
ALTER DEFAULT PRIVILEGES IN SCHEMA aibf_analytics GRANT ALL ON SEQUENCES TO aibf;
ALTER DEFAULT PRIVILEGES IN SCHEMA aibf_ml GRANT ALL ON SEQUENCES TO aibf;

-- Create views for common queries

-- Active users view
CREATE OR REPLACE VIEW aibf_auth.active_users AS
SELECT 
    id,
    username,
    email,
    first_name,
    last_name,
    created_at,
    last_login
FROM aibf_auth.users
WHERE is_active = true;

-- Recent training jobs view
CREATE OR REPLACE VIEW aibf_ml.recent_training_jobs AS
SELECT 
    tj.id,
    tj.name,
    m.name as model_name,
    m.version as model_version,
    tj.status,
    tj.progress,
    tj.created_at,
    tj.started_at,
    tj.completed_at
FROM aibf_ml.training_jobs tj
JOIN aibf_ml.models m ON tj.model_id = m.id
ORDER BY tj.created_at DESC;

-- System health view
CREATE OR REPLACE VIEW aibf_monitoring.system_health AS
SELECT 
    metric_name,
    AVG(value) as avg_value,
    MAX(value) as max_value,
    MIN(value) as min_value,
    COUNT(*) as sample_count,
    MAX(timestamp) as last_updated
FROM aibf_monitoring.system_metrics
WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '1 hour'
GROUP BY metric_name;

-- Commit the transaction
COMMIT;

-- Vacuum and analyze for optimal performance
VACUUM ANALYZE;

-- Success message
\echo 'AIBF database initialization completed successfully!'