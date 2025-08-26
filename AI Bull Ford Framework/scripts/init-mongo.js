// AI Bull Ford (AIBF) - MongoDB Initialization Script
// MongoDB database setup and initial collections

// Switch to AIBF database
db = db.getSiblingDB('aibf');

// Create collections with validation schemas

// Users collection (for additional user data)
db.createCollection('users', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['user_id', 'profile'],
            properties: {
                user_id: {
                    bsonType: 'string',
                    description: 'User ID from PostgreSQL'
                },
                profile: {
                    bsonType: 'object',
                    properties: {
                        avatar_url: { bsonType: 'string' },
                        bio: { bsonType: 'string' },
                        preferences: { bsonType: 'object' },
                        social_links: { bsonType: 'object' }
                    }
                },
                settings: {
                    bsonType: 'object',
                    properties: {
                        theme: { bsonType: 'string' },
                        language: { bsonType: 'string' },
                        notifications: { bsonType: 'object' },
                        privacy: { bsonType: 'object' }
                    }
                },
                created_at: {
                    bsonType: 'date',
                    description: 'Creation timestamp'
                },
                updated_at: {
                    bsonType: 'date',
                    description: 'Last update timestamp'
                }
            }
        }
    }
});

// Documents collection (for RAG and document storage)
db.createCollection('documents', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['title', 'content', 'document_type'],
            properties: {
                title: {
                    bsonType: 'string',
                    description: 'Document title'
                },
                content: {
                    bsonType: 'string',
                    description: 'Document content'
                },
                document_type: {
                    bsonType: 'string',
                    enum: ['text', 'pdf', 'html', 'markdown', 'code', 'image', 'audio', 'video'],
                    description: 'Type of document'
                },
                metadata: {
                    bsonType: 'object',
                    properties: {
                        author: { bsonType: 'string' },
                        source: { bsonType: 'string' },
                        language: { bsonType: 'string' },
                        tags: { bsonType: 'array' },
                        category: { bsonType: 'string' },
                        file_size: { bsonType: 'number' },
                        file_path: { bsonType: 'string' },
                        checksum: { bsonType: 'string' }
                    }
                },
                embeddings: {
                    bsonType: 'object',
                    properties: {
                        model: { bsonType: 'string' },
                        vectors: { bsonType: 'array' },
                        dimensions: { bsonType: 'number' }
                    }
                },
                chunks: {
                    bsonType: 'array',
                    items: {
                        bsonType: 'object',
                        properties: {
                            text: { bsonType: 'string' },
                            start_index: { bsonType: 'number' },
                            end_index: { bsonType: 'number' },
                            embedding: { bsonType: 'array' }
                        }
                    }
                },
                status: {
                    bsonType: 'string',
                    enum: ['pending', 'processing', 'indexed', 'failed'],
                    description: 'Processing status'
                },
                created_at: {
                    bsonType: 'date',
                    description: 'Creation timestamp'
                },
                updated_at: {
                    bsonType: 'date',
                    description: 'Last update timestamp'
                }
            }
        }
    }
});

// Conversations collection (for chat and dialogue systems)
db.createCollection('conversations', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['user_id', 'messages'],
            properties: {
                user_id: {
                    bsonType: 'string',
                    description: 'User ID'
                },
                session_id: {
                    bsonType: 'string',
                    description: 'Session identifier'
                },
                title: {
                    bsonType: 'string',
                    description: 'Conversation title'
                },
                messages: {
                    bsonType: 'array',
                    items: {
                        bsonType: 'object',
                        required: ['role', 'content', 'timestamp'],
                        properties: {
                            role: {
                                bsonType: 'string',
                                enum: ['user', 'assistant', 'system'],
                                description: 'Message role'
                            },
                            content: {
                                bsonType: 'string',
                                description: 'Message content'
                            },
                            metadata: {
                                bsonType: 'object',
                                properties: {
                                    model: { bsonType: 'string' },
                                    tokens: { bsonType: 'number' },
                                    confidence: { bsonType: 'number' },
                                    processing_time: { bsonType: 'number' }
                                }
                            },
                            timestamp: {
                                bsonType: 'date',
                                description: 'Message timestamp'
                            }
                        }
                    }
                },
                context: {
                    bsonType: 'object',
                    properties: {
                        model_config: { bsonType: 'object' },
                        system_prompt: { bsonType: 'string' },
                        temperature: { bsonType: 'number' },
                        max_tokens: { bsonType: 'number' }
                    }
                },
                status: {
                    bsonType: 'string',
                    enum: ['active', 'archived', 'deleted'],
                    description: 'Conversation status'
                },
                created_at: {
                    bsonType: 'date',
                    description: 'Creation timestamp'
                },
                updated_at: {
                    bsonType: 'date',
                    description: 'Last update timestamp'
                }
            }
        }
    }
});

// Knowledge Base collection
db.createCollection('knowledge_base', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['topic', 'content'],
            properties: {
                topic: {
                    bsonType: 'string',
                    description: 'Knowledge topic'
                },
                content: {
                    bsonType: 'string',
                    description: 'Knowledge content'
                },
                category: {
                    bsonType: 'string',
                    description: 'Knowledge category'
                },
                tags: {
                    bsonType: 'array',
                    items: { bsonType: 'string' },
                    description: 'Knowledge tags'
                },
                confidence: {
                    bsonType: 'number',
                    minimum: 0,
                    maximum: 1,
                    description: 'Confidence score'
                },
                sources: {
                    bsonType: 'array',
                    items: {
                        bsonType: 'object',
                        properties: {
                            url: { bsonType: 'string' },
                            title: { bsonType: 'string' },
                            author: { bsonType: 'string' },
                            date: { bsonType: 'date' }
                        }
                    }
                },
                embeddings: {
                    bsonType: 'object',
                    properties: {
                        model: { bsonType: 'string' },
                        vector: { bsonType: 'array' },
                        dimensions: { bsonType: 'number' }
                    }
                },
                created_at: {
                    bsonType: 'date',
                    description: 'Creation timestamp'
                },
                updated_at: {
                    bsonType: 'date',
                    description: 'Last update timestamp'
                }
            }
        }
    }
});

// Model Artifacts collection
db.createCollection('model_artifacts', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['model_id', 'artifact_type', 'data'],
            properties: {
                model_id: {
                    bsonType: 'string',
                    description: 'Model ID from PostgreSQL'
                },
                artifact_type: {
                    bsonType: 'string',
                    enum: ['weights', 'config', 'tokenizer', 'metadata', 'logs', 'metrics'],
                    description: 'Type of artifact'
                },
                data: {
                    bsonType: 'object',
                    description: 'Artifact data'
                },
                version: {
                    bsonType: 'string',
                    description: 'Artifact version'
                },
                size_bytes: {
                    bsonType: 'number',
                    description: 'Artifact size in bytes'
                },
                checksum: {
                    bsonType: 'string',
                    description: 'Artifact checksum'
                },
                storage_path: {
                    bsonType: 'string',
                    description: 'Storage path for large artifacts'
                },
                created_at: {
                    bsonType: 'date',
                    description: 'Creation timestamp'
                }
            }
        }
    }
});

// Analytics Events collection
db.createCollection('analytics_events', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['event_type', 'timestamp'],
            properties: {
                event_type: {
                    bsonType: 'string',
                    description: 'Type of event'
                },
                user_id: {
                    bsonType: 'string',
                    description: 'User ID'
                },
                session_id: {
                    bsonType: 'string',
                    description: 'Session ID'
                },
                properties: {
                    bsonType: 'object',
                    description: 'Event properties'
                },
                context: {
                    bsonType: 'object',
                    properties: {
                        ip_address: { bsonType: 'string' },
                        user_agent: { bsonType: 'string' },
                        referrer: { bsonType: 'string' },
                        page_url: { bsonType: 'string' },
                        device_type: { bsonType: 'string' },
                        browser: { bsonType: 'string' },
                        os: { bsonType: 'string' }
                    }
                },
                timestamp: {
                    bsonType: 'date',
                    description: 'Event timestamp'
                }
            }
        }
    }
});

// Cache collection (for application caching)
db.createCollection('cache', {
    validator: {
        $jsonSchema: {
            bsonType: 'object',
            required: ['key', 'value'],
            properties: {
                key: {
                    bsonType: 'string',
                    description: 'Cache key'
                },
                value: {
                    description: 'Cache value (any type)'
                },
                ttl: {
                    bsonType: 'date',
                    description: 'Time to live'
                },
                tags: {
                    bsonType: 'array',
                    items: { bsonType: 'string' },
                    description: 'Cache tags for grouping'
                },
                created_at: {
                    bsonType: 'date',
                    description: 'Creation timestamp'
                }
            }
        }
    }
});

// Create indexes for better performance

// Users collection indexes
db.users.createIndex({ 'user_id': 1 }, { unique: true });
db.users.createIndex({ 'created_at': 1 });
db.users.createIndex({ 'updated_at': 1 });

// Documents collection indexes
db.documents.createIndex({ 'title': 'text', 'content': 'text' });
db.documents.createIndex({ 'document_type': 1 });
db.documents.createIndex({ 'metadata.tags': 1 });
db.documents.createIndex({ 'metadata.category': 1 });
db.documents.createIndex({ 'status': 1 });
db.documents.createIndex({ 'created_at': 1 });
db.documents.createIndex({ 'metadata.checksum': 1 });

// Conversations collection indexes
db.conversations.createIndex({ 'user_id': 1 });
db.conversations.createIndex({ 'session_id': 1 });
db.conversations.createIndex({ 'status': 1 });
db.conversations.createIndex({ 'created_at': 1 });
db.conversations.createIndex({ 'updated_at': 1 });
db.conversations.createIndex({ 'messages.timestamp': 1 });

// Knowledge Base collection indexes
db.knowledge_base.createIndex({ 'topic': 'text', 'content': 'text' });
db.knowledge_base.createIndex({ 'category': 1 });
db.knowledge_base.createIndex({ 'tags': 1 });
db.knowledge_base.createIndex({ 'confidence': 1 });
db.knowledge_base.createIndex({ 'created_at': 1 });

// Model Artifacts collection indexes
db.model_artifacts.createIndex({ 'model_id': 1 });
db.model_artifacts.createIndex({ 'artifact_type': 1 });
db.model_artifacts.createIndex({ 'version': 1 });
db.model_artifacts.createIndex({ 'created_at': 1 });
db.model_artifacts.createIndex({ 'model_id': 1, 'artifact_type': 1, 'version': 1 }, { unique: true });

// Analytics Events collection indexes
db.analytics_events.createIndex({ 'event_type': 1 });
db.analytics_events.createIndex({ 'user_id': 1 });
db.analytics_events.createIndex({ 'session_id': 1 });
db.analytics_events.createIndex({ 'timestamp': 1 });
db.analytics_events.createIndex({ 'event_type': 1, 'timestamp': 1 });

// Cache collection indexes
db.cache.createIndex({ 'key': 1 }, { unique: true });
db.cache.createIndex({ 'ttl': 1 }, { expireAfterSeconds: 0 });
db.cache.createIndex({ 'tags': 1 });
db.cache.createIndex({ 'created_at': 1 });

// Insert sample data

// Sample knowledge base entries
db.knowledge_base.insertMany([
    {
        topic: 'Machine Learning Basics',
        content: 'Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.',
        category: 'AI/ML',
        tags: ['machine learning', 'AI', 'basics', 'introduction'],
        confidence: 0.95,
        sources: [{
            url: 'https://example.com/ml-basics',
            title: 'Introduction to Machine Learning',
            author: 'AI Expert',
            date: new Date()
        }],
        created_at: new Date(),
        updated_at: new Date()
    },
    {
        topic: 'Neural Networks',
        content: 'Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information.',
        category: 'Deep Learning',
        tags: ['neural networks', 'deep learning', 'neurons', 'AI'],
        confidence: 0.92,
        sources: [{
            url: 'https://example.com/neural-networks',
            title: 'Understanding Neural Networks',
            author: 'Deep Learning Researcher',
            date: new Date()
        }],
        created_at: new Date(),
        updated_at: new Date()
    },
    {
        topic: 'Transformers Architecture',
        content: 'Transformers are a type of neural network architecture that has revolutionized natural language processing. They use self-attention mechanisms to process sequences.',
        category: 'NLP',
        tags: ['transformers', 'attention', 'NLP', 'BERT', 'GPT'],
        confidence: 0.98,
        sources: [{
            url: 'https://example.com/transformers',
            title: 'Attention Is All You Need',
            author: 'Vaswani et al.',
            date: new Date()
        }],
        created_at: new Date(),
        updated_at: new Date()
    }
]);

// Sample documents
db.documents.insertMany([
    {
        title: 'AIBF Getting Started Guide',
        content: 'This guide will help you get started with the AI Bull Ford framework. Follow these steps to set up your first AI application...',
        document_type: 'markdown',
        metadata: {
            author: 'AIBF Team',
            source: 'documentation',
            language: 'en',
            tags: ['guide', 'getting-started', 'documentation'],
            category: 'documentation'
        },
        status: 'indexed',
        created_at: new Date(),
        updated_at: new Date()
    },
    {
        title: 'API Reference',
        content: 'Complete API reference for the AIBF framework. This document covers all available endpoints, parameters, and response formats...',
        document_type: 'html',
        metadata: {
            author: 'AIBF Team',
            source: 'api-docs',
            language: 'en',
            tags: ['api', 'reference', 'documentation'],
            category: 'reference'
        },
        status: 'indexed',
        created_at: new Date(),
        updated_at: new Date()
    }
]);

// Sample cache entries
db.cache.insertMany([
    {
        key: 'app:config:default',
        value: {
            theme: 'light',
            language: 'en',
            timezone: 'UTC'
        },
        tags: ['config', 'default'],
        created_at: new Date()
    },
    {
        key: 'models:popular',
        value: [
            'gpt-3.5-turbo',
            'claude-3-sonnet',
            'llama-2-70b'
        ],
        tags: ['models', 'popular'],
        created_at: new Date()
    }
]);

// Create admin user for MongoDB
db.createUser({
    user: 'aibf_admin',
    pwd: 'aibf_admin_password',
    roles: [
        { role: 'readWrite', db: 'aibf' },
        { role: 'dbAdmin', db: 'aibf' }
    ]
});

// Create application user for MongoDB
db.createUser({
    user: 'aibf_app',
    pwd: 'aibf_app_password',
    roles: [
        { role: 'readWrite', db: 'aibf' }
    ]
});

// Print success message
print('AIBF MongoDB initialization completed successfully!');
print('Collections created: ' + db.getCollectionNames().length);
print('Sample data inserted.');
print('Users created: aibf_admin, aibf_app');