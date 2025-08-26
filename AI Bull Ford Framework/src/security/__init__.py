"""Security layer for AI Bull Ford.

Provides authentication, authorization, validation, encryption, and audit logging.
"""

from .authentication import (
    AuthenticationManager,
    UserManager,
    TokenManager,
    User,
    AuthToken,
    UserRole,
    UserStatus,
    TokenType
)

from .authorization import (
    AuthorizationManager,
    PermissionManager,
    RoleManager,
    Permission,
    Role,
    AccessRequest,
    AccessResult,
    PermissionType,
    ResourceType,
    require_permission
)

from .validation import (
    ValidationEngine,
    SchemaValidator,
    Validator,
    Sanitizer,
    ValidationRule,
    ValidationResult,
    ValidationType,
    SanitizationType,
    get_validation_engine,
    get_schema_validator
)

from .encryption import (
    CryptographyManager,
    SymmetricEncryption,
    AsymmetricEncryption,
    Hasher,
    KeyStore,
    MemoryKeyStore,
    FileKeyStore,
    EncryptionKey,
    EncryptedData,
    HashResult,
    EncryptionAlgorithm,
    HashAlgorithm,
    KeyDerivationFunction,
    get_crypto_manager,
    initialize_crypto_manager
)

from .audit import (
    AuditLogger,
    AuditStore,
    SQLiteAuditStore,
    FileAuditStore,
    AuditEvent,
    AuditFilter,
    EventType,
    EventSeverity,
    EventStatus,
    audit_decorator,
    audit_context,
    get_audit_logger,
    initialize_audit_logger,
    shutdown_audit_logger
)

__all__ = [
    # Authentication
    'AuthenticationManager',
    'UserManager', 
    'TokenManager',
    'User',
    'AuthToken',
    'UserRole',
    'UserStatus',
    'TokenType',
    
    # Authorization
    'AuthorizationManager',
    'PermissionManager',
    'RoleManager',
    'Permission',
    'Role',
    'AccessRequest',
    'AccessResult',
    'PermissionType',
    'ResourceType',
    'require_permission',
    
    # Validation
    'ValidationEngine',
    'SchemaValidator',
    'Validator',
    'Sanitizer',
    'ValidationRule',
    'ValidationResult',
    'ValidationType',
    'SanitizationType',
    'get_validation_engine',
    'get_schema_validator',
    
    # Encryption
    'CryptographyManager',
    'SymmetricEncryption',
    'AsymmetricEncryption',
    'Hasher',
    'KeyStore',
    'MemoryKeyStore',
    'FileKeyStore',
    'EncryptionKey',
    'EncryptedData',
    'HashResult',
    'EncryptionAlgorithm',
    'HashAlgorithm',
    'KeyDerivationFunction',
    'get_crypto_manager',
    'initialize_crypto_manager',
    
    # Audit
    'AuditLogger',
    'AuditStore',
    'SQLiteAuditStore',
    'FileAuditStore',
    'AuditEvent',
    'AuditFilter',
    'EventType',
    'EventSeverity',
    'EventStatus',
    'audit_decorator',
    'audit_context',
    'get_audit_logger',
    'initialize_audit_logger',
    'shutdown_audit_logger'
]