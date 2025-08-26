"""Encryption module for AI Bull Ford security layer.

Provides encryption, decryption, hashing, and cryptographic utilities.
"""

import os
import base64
import hashlib
import hmac
import secrets
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import struct
import time

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import hashes, serialization, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
    from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
    from cryptography.hazmat.backends import default_backend
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logging.warning("Cryptography library not available. Some encryption features will be limited.")


class EncryptionAlgorithm(Enum):
    """Encryption algorithms."""
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    FERNET = "fernet"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    CHACHA20_POLY1305 = "chacha20_poly1305"


class HashAlgorithm(Enum):
    """Hash algorithms."""
    SHA256 = "sha256"
    SHA512 = "sha512"
    SHA3_256 = "sha3_256"
    SHA3_512 = "sha3_512"
    BLAKE2B = "blake2b"
    BLAKE2S = "blake2s"
    MD5 = "md5"  # For compatibility only, not recommended


class KeyDerivationFunction(Enum):
    """Key derivation functions."""
    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"
    ARGON2 = "argon2"


@dataclass
class EncryptionKey:
    """Encryption key container."""
    key_id: str
    algorithm: EncryptionAlgorithm
    key_data: bytes
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if key is expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert key to dictionary (without sensitive data)."""
        return {
            'key_id': self.key_id,
            'algorithm': self.algorithm.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'metadata': self.metadata
        }


@dataclass
class EncryptedData:
    """Encrypted data container."""
    ciphertext: bytes
    algorithm: EncryptionAlgorithm
    key_id: str
    iv: Optional[bytes] = None
    tag: Optional[bytes] = None
    salt: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'ciphertext': base64.b64encode(self.ciphertext).decode('utf-8'),
            'algorithm': self.algorithm.value,
            'key_id': self.key_id,
            'iv': base64.b64encode(self.iv).decode('utf-8') if self.iv else None,
            'tag': base64.b64encode(self.tag).decode('utf-8') if self.tag else None,
            'salt': base64.b64encode(self.salt).decode('utf-8') if self.salt else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedData':
        """Create from dictionary."""
        return cls(
            ciphertext=base64.b64decode(data['ciphertext']),
            algorithm=EncryptionAlgorithm(data['algorithm']),
            key_id=data['key_id'],
            iv=base64.b64decode(data['iv']) if data.get('iv') else None,
            tag=base64.b64decode(data['tag']) if data.get('tag') else None,
            salt=base64.b64decode(data['salt']) if data.get('salt') else None,
            metadata=data.get('metadata', {})
        )


@dataclass
class HashResult:
    """Hash result container."""
    hash_value: str
    algorithm: HashAlgorithm
    salt: Optional[str] = None
    iterations: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'hash_value': self.hash_value,
            'algorithm': self.algorithm.value,
            'salt': self.salt,
            'iterations': self.iterations,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HashResult':
        """Create from dictionary."""
        return cls(
            hash_value=data['hash_value'],
            algorithm=HashAlgorithm(data['algorithm']),
            salt=data.get('salt'),
            iterations=data.get('iterations'),
            metadata=data.get('metadata', {})
        )


class KeyStore(ABC):
    """Abstract key storage interface."""
    
    @abstractmethod
    def store_key(self, key: EncryptionKey) -> bool:
        """Store an encryption key."""
        pass
    
    @abstractmethod
    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get an encryption key."""
        pass
    
    @abstractmethod
    def list_keys(self) -> List[EncryptionKey]:
        """List all keys."""
        pass
    
    @abstractmethod
    def delete_key(self, key_id: str) -> bool:
        """Delete a key."""
        pass
    
    @abstractmethod
    def rotate_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Rotate a key."""
        pass


class MemoryKeyStore(KeyStore):
    """In-memory key storage (for development/testing only)."""
    
    def __init__(self):
        """Initialize memory key store."""
        self.keys: Dict[str, EncryptionKey] = {}
        self.logger = logging.getLogger("security.keystore.memory")
        self.logger.warning("Using in-memory key store - not suitable for production!")
    
    def store_key(self, key: EncryptionKey) -> bool:
        """Store an encryption key."""
        self.keys[key.key_id] = key
        self.logger.info(f"Key stored: {key.key_id}")
        return True
    
    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get an encryption key."""
        key = self.keys.get(key_id)
        if key and key.is_expired():
            self.logger.warning(f"Key {key_id} is expired")
            return None
        return key
    
    def list_keys(self) -> List[EncryptionKey]:
        """List all keys."""
        return [key for key in self.keys.values() if not key.is_expired()]
    
    def delete_key(self, key_id: str) -> bool:
        """Delete a key."""
        if key_id in self.keys:
            del self.keys[key_id]
            self.logger.info(f"Key deleted: {key_id}")
            return True
        return False
    
    def rotate_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Rotate a key."""
        old_key = self.keys.get(key_id)
        if not old_key:
            return None
        
        # Create new key with same algorithm
        new_key = EncryptionKey(
            key_id=f"{key_id}_rotated_{int(time.time())}",
            algorithm=old_key.algorithm,
            key_data=self._generate_key(old_key.algorithm),
            metadata=old_key.metadata.copy()
        )
        
        self.store_key(new_key)
        self.logger.info(f"Key rotated: {key_id} -> {new_key.key_id}")
        return new_key
    
    def _generate_key(self, algorithm: EncryptionAlgorithm) -> bytes:
        """Generate a new key for the given algorithm."""
        if algorithm in [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.AES_256_CBC]:
            return secrets.token_bytes(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.FERNET:
            return Fernet.generate_key()
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return secrets.token_bytes(32)  # 256 bits
        else:
            return secrets.token_bytes(32)  # Default


class FileKeyStore(KeyStore):
    """File-based key storage (encrypted)."""
    
    def __init__(self, key_file: str, master_password: str):
        """Initialize file key store.
        
        Args:
            key_file: Path to key storage file
            master_password: Master password for encrypting the key file
        """
        self.key_file = Path(key_file)
        self.master_password = master_password
        self.logger = logging.getLogger("security.keystore.file")
        
        # Derive master key from password
        self.master_key = self._derive_master_key(master_password)
        
        # Load existing keys
        self.keys: Dict[str, EncryptionKey] = {}
        self._load_keys()
    
    def _derive_master_key(self, password: str) -> bytes:
        """Derive master key from password."""
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback to simple hash (not secure)
            return hashlib.sha256(password.encode()).digest()
        
        salt = b'aibf_keystore_salt'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(password.encode())
    
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data with master key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback to simple XOR (not secure)
            return bytes(a ^ b for a, b in zip(data, (self.master_key * (len(data) // len(self.master_key) + 1))[:len(data)]))
        
        fernet = Fernet(base64.urlsafe_b64encode(self.master_key))
        return fernet.encrypt(data)
    
    def _decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data with master key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            # Fallback to simple XOR (not secure)
            return bytes(a ^ b for a, b in zip(encrypted_data, (self.master_key * (len(encrypted_data) // len(self.master_key) + 1))[:len(encrypted_data)]))
        
        fernet = Fernet(base64.urlsafe_b64encode(self.master_key))
        return fernet.decrypt(encrypted_data)
    
    def _load_keys(self) -> None:
        """Load keys from file."""
        if not self.key_file.exists():
            return
        
        try:
            with open(self.key_file, 'rb') as f:
                encrypted_data = f.read()
            
            if not encrypted_data:
                return
            
            decrypted_data = self._decrypt_data(encrypted_data)
            keys_data = json.loads(decrypted_data.decode())
            
            for key_data in keys_data:
                key = EncryptionKey(
                    key_id=key_data['key_id'],
                    algorithm=EncryptionAlgorithm(key_data['algorithm']),
                    key_data=base64.b64decode(key_data['key_data']),
                    created_at=datetime.fromisoformat(key_data['created_at']),
                    expires_at=datetime.fromisoformat(key_data['expires_at']) if key_data.get('expires_at') else None,
                    metadata=key_data.get('metadata', {})
                )
                self.keys[key.key_id] = key
            
            self.logger.info(f"Loaded {len(self.keys)} keys from {self.key_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to load keys: {e}")
    
    def _save_keys(self) -> None:
        """Save keys to file."""
        try:
            keys_data = []
            for key in self.keys.values():
                key_data = {
                    'key_id': key.key_id,
                    'algorithm': key.algorithm.value,
                    'key_data': base64.b64encode(key.key_data).decode(),
                    'created_at': key.created_at.isoformat(),
                    'expires_at': key.expires_at.isoformat() if key.expires_at else None,
                    'metadata': key.metadata
                }
                keys_data.append(key_data)
            
            json_data = json.dumps(keys_data).encode()
            encrypted_data = self._encrypt_data(json_data)
            
            # Ensure directory exists
            self.key_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.key_file, 'wb') as f:
                f.write(encrypted_data)
            
            self.logger.info(f"Saved {len(self.keys)} keys to {self.key_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save keys: {e}")
    
    def store_key(self, key: EncryptionKey) -> bool:
        """Store an encryption key."""
        self.keys[key.key_id] = key
        self._save_keys()
        return True
    
    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get an encryption key."""
        key = self.keys.get(key_id)
        if key and key.is_expired():
            self.logger.warning(f"Key {key_id} is expired")
            return None
        return key
    
    def list_keys(self) -> List[EncryptionKey]:
        """List all keys."""
        return [key for key in self.keys.values() if not key.is_expired()]
    
    def delete_key(self, key_id: str) -> bool:
        """Delete a key."""
        if key_id in self.keys:
            del self.keys[key_id]
            self._save_keys()
            self.logger.info(f"Key deleted: {key_id}")
            return True
        return False
    
    def rotate_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Rotate a key."""
        old_key = self.keys.get(key_id)
        if not old_key:
            return None
        
        # Create new key with same algorithm
        new_key = EncryptionKey(
            key_id=f"{key_id}_rotated_{int(time.time())}",
            algorithm=old_key.algorithm,
            key_data=self._generate_key(old_key.algorithm),
            metadata=old_key.metadata.copy()
        )
        
        self.store_key(new_key)
        self.logger.info(f"Key rotated: {key_id} -> {new_key.key_id}")
        return new_key
    
    def _generate_key(self, algorithm: EncryptionAlgorithm) -> bytes:
        """Generate a new key for the given algorithm."""
        if algorithm in [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.AES_256_CBC]:
            return secrets.token_bytes(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.FERNET:
            return Fernet.generate_key()
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return secrets.token_bytes(32)  # 256 bits
        else:
            return secrets.token_bytes(32)  # Default


class Hasher:
    """Cryptographic hasher."""
    
    def __init__(self):
        """Initialize hasher."""
        self.logger = logging.getLogger("security.hasher")
    
    def hash_data(self, 
                  data: Union[str, bytes], 
                  algorithm: HashAlgorithm = HashAlgorithm.SHA256,
                  salt: Optional[Union[str, bytes]] = None,
                  iterations: int = 1) -> HashResult:
        """Hash data.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm
            salt: Optional salt
            iterations: Number of iterations for key stretching
            
        Returns:
            Hash result
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if salt is None:
            salt = secrets.token_bytes(16)
        elif isinstance(salt, str):
            salt = salt.encode('utf-8')
        
        # Combine data and salt
        salted_data = data + salt
        
        # Apply iterations
        hash_value = salted_data
        for _ in range(iterations):
            hash_value = self._hash_bytes(hash_value, algorithm)
        
        return HashResult(
            hash_value=hash_value.hex(),
            algorithm=algorithm,
            salt=salt.hex(),
            iterations=iterations
        )
    
    def verify_hash(self, data: Union[str, bytes], hash_result: HashResult) -> bool:
        """Verify data against hash.
        
        Args:
            data: Data to verify
            hash_result: Hash result to verify against
            
        Returns:
            True if data matches hash
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        salt = bytes.fromhex(hash_result.salt) if hash_result.salt else b''
        
        # Recreate hash
        salted_data = data + salt
        hash_value = salted_data
        
        iterations = hash_result.iterations or 1
        for _ in range(iterations):
            hash_value = self._hash_bytes(hash_value, hash_result.algorithm)
        
        return hash_value.hex() == hash_result.hash_value
    
    def _hash_bytes(self, data: bytes, algorithm: HashAlgorithm) -> bytes:
        """Hash bytes with specified algorithm."""
        if algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256(data).digest()
        elif algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512(data).digest()
        elif algorithm == HashAlgorithm.SHA3_256:
            return hashlib.sha3_256(data).digest()
        elif algorithm == HashAlgorithm.SHA3_512:
            return hashlib.sha3_512(data).digest()
        elif algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b(data).digest()
        elif algorithm == HashAlgorithm.BLAKE2S:
            return hashlib.blake2s(data).digest()
        elif algorithm == HashAlgorithm.MD5:
            return hashlib.md5(data).digest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    def hmac_sign(self, 
                  data: Union[str, bytes], 
                  key: Union[str, bytes],
                  algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> str:
        """Create HMAC signature.
        
        Args:
            data: Data to sign
            key: HMAC key
            algorithm: Hash algorithm
            
        Returns:
            HMAC signature (hex)
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
        if isinstance(key, str):
            key = key.encode('utf-8')
        
        if algorithm == HashAlgorithm.SHA256:
            hash_func = hashlib.sha256
        elif algorithm == HashAlgorithm.SHA512:
            hash_func = hashlib.sha512
        else:
            hash_func = hashlib.sha256  # Default
        
        signature = hmac.new(key, data, hash_func).hexdigest()
        return signature
    
    def hmac_verify(self, 
                    data: Union[str, bytes], 
                    key: Union[str, bytes],
                    signature: str,
                    algorithm: HashAlgorithm = HashAlgorithm.SHA256) -> bool:
        """Verify HMAC signature.
        
        Args:
            data: Data to verify
            key: HMAC key
            signature: HMAC signature to verify
            algorithm: Hash algorithm
            
        Returns:
            True if signature is valid
        """
        expected_signature = self.hmac_sign(data, key, algorithm)
        return hmac.compare_digest(signature, expected_signature)


class SymmetricEncryption:
    """Symmetric encryption handler."""
    
    def __init__(self, key_store: KeyStore):
        """Initialize symmetric encryption.
        
        Args:
            key_store: Key storage backend
        """
        self.key_store = key_store
        self.logger = logging.getLogger("security.symmetric")
    
    def generate_key(self, 
                    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM,
                    key_id: Optional[str] = None,
                    expires_in: Optional[timedelta] = None) -> EncryptionKey:
        """Generate a new encryption key.
        
        Args:
            algorithm: Encryption algorithm
            key_id: Optional key ID
            expires_in: Optional expiration time
            
        Returns:
            Generated encryption key
        """
        if key_id is None:
            key_id = f"key_{secrets.token_urlsafe(16)}"
        
        # Generate key data based on algorithm
        if algorithm in [EncryptionAlgorithm.AES_256_GCM, EncryptionAlgorithm.AES_256_CBC]:
            key_data = secrets.token_bytes(32)  # 256 bits
        elif algorithm == EncryptionAlgorithm.FERNET:
            key_data = Fernet.generate_key()
        elif algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            key_data = secrets.token_bytes(32)  # 256 bits
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        expires_at = None
        if expires_in:
            expires_at = datetime.now() + expires_in
        
        key = EncryptionKey(
            key_id=key_id,
            algorithm=algorithm,
            key_data=key_data,
            expires_at=expires_at
        )
        
        self.key_store.store_key(key)
        self.logger.info(f"Generated key: {key_id} ({algorithm.value})")
        return key
    
    def encrypt(self, 
               data: Union[str, bytes], 
               key_id: str,
               associated_data: Optional[bytes] = None) -> EncryptedData:
        """Encrypt data.
        
        Args:
            data: Data to encrypt
            key_id: Key ID to use for encryption
            associated_data: Optional associated data for AEAD
            
        Returns:
            Encrypted data
        """
        key = self.key_store.get_key(key_id)
        if not key:
            raise ValueError(f"Key not found: {key_id}")
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        if key.algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._encrypt_aes_gcm(data, key, associated_data)
        elif key.algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._encrypt_aes_cbc(data, key)
        elif key.algorithm == EncryptionAlgorithm.FERNET:
            return self._encrypt_fernet(data, key)
        elif key.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._encrypt_chacha20_poly1305(data, key, associated_data)
        else:
            raise ValueError(f"Unsupported algorithm: {key.algorithm}")
    
    def decrypt(self, 
               encrypted_data: EncryptedData,
               associated_data: Optional[bytes] = None) -> bytes:
        """Decrypt data.
        
        Args:
            encrypted_data: Encrypted data
            associated_data: Optional associated data for AEAD
            
        Returns:
            Decrypted data
        """
        key = self.key_store.get_key(encrypted_data.key_id)
        if not key:
            raise ValueError(f"Key not found: {encrypted_data.key_id}")
        
        if encrypted_data.algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._decrypt_aes_gcm(encrypted_data, key, associated_data)
        elif encrypted_data.algorithm == EncryptionAlgorithm.AES_256_CBC:
            return self._decrypt_aes_cbc(encrypted_data, key)
        elif encrypted_data.algorithm == EncryptionAlgorithm.FERNET:
            return self._decrypt_fernet(encrypted_data, key)
        elif encrypted_data.algorithm == EncryptionAlgorithm.CHACHA20_POLY1305:
            return self._decrypt_chacha20_poly1305(encrypted_data, key, associated_data)
        else:
            raise ValueError(f"Unsupported algorithm: {encrypted_data.algorithm}")
    
    def _encrypt_aes_gcm(self, data: bytes, key: EncryptionKey, associated_data: Optional[bytes]) -> EncryptedData:
        """Encrypt with AES-256-GCM."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library required for AES-GCM")
        
        iv = secrets.token_bytes(12)  # 96-bit IV for GCM
        cipher = Cipher(algorithms.AES(key.key_data), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return EncryptedData(
            ciphertext=ciphertext,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            key_id=key.key_id,
            iv=iv,
            tag=encryptor.tag
        )
    
    def _decrypt_aes_gcm(self, encrypted_data: EncryptedData, key: EncryptionKey, associated_data: Optional[bytes]) -> bytes:
        """Decrypt with AES-256-GCM."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library required for AES-GCM")
        
        cipher = Cipher(algorithms.AES(key.key_data), modes.GCM(encrypted_data.iv, encrypted_data.tag), backend=default_backend())
        decryptor = cipher.decryptor()
        
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)
        
        return decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()
    
    def _encrypt_aes_cbc(self, data: bytes, key: EncryptionKey) -> EncryptedData:
        """Encrypt with AES-256-CBC."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library required for AES-CBC")
        
        # Pad data to block size
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        iv = secrets.token_bytes(16)  # 128-bit IV for CBC
        cipher = Cipher(algorithms.AES(key.key_data), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        return EncryptedData(
            ciphertext=ciphertext,
            algorithm=EncryptionAlgorithm.AES_256_CBC,
            key_id=key.key_id,
            iv=iv
        )
    
    def _decrypt_aes_cbc(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt with AES-256-CBC."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library required for AES-CBC")
        
        cipher = Cipher(algorithms.AES(key.key_data), modes.CBC(encrypted_data.iv), backend=default_backend())
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()
        
        # Remove padding
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()
    
    def _encrypt_fernet(self, data: bytes, key: EncryptionKey) -> EncryptedData:
        """Encrypt with Fernet."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library required for Fernet")
        
        fernet = Fernet(key.key_data)
        ciphertext = fernet.encrypt(data)
        
        return EncryptedData(
            ciphertext=ciphertext,
            algorithm=EncryptionAlgorithm.FERNET,
            key_id=key.key_id
        )
    
    def _decrypt_fernet(self, encrypted_data: EncryptedData, key: EncryptionKey) -> bytes:
        """Decrypt with Fernet."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library required for Fernet")
        
        fernet = Fernet(key.key_data)
        return fernet.decrypt(encrypted_data.ciphertext)
    
    def _encrypt_chacha20_poly1305(self, data: bytes, key: EncryptionKey, associated_data: Optional[bytes]) -> EncryptedData:
        """Encrypt with ChaCha20-Poly1305."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library required for ChaCha20-Poly1305")
        
        nonce = secrets.token_bytes(12)  # 96-bit nonce
        cipher = Cipher(algorithms.ChaCha20(key.key_data, nonce), None, backend=default_backend())
        encryptor = cipher.encryptor()
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        return EncryptedData(
            ciphertext=ciphertext,
            algorithm=EncryptionAlgorithm.CHACHA20_POLY1305,
            key_id=key.key_id,
            iv=nonce
        )
    
    def _decrypt_chacha20_poly1305(self, encrypted_data: EncryptedData, key: EncryptionKey, associated_data: Optional[bytes]) -> bytes:
        """Decrypt with ChaCha20-Poly1305."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library required for ChaCha20-Poly1305")
        
        cipher = Cipher(algorithms.ChaCha20(key.key_data, encrypted_data.iv), None, backend=default_backend())
        decryptor = cipher.decryptor()
        
        return decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()


class AsymmetricEncryption:
    """Asymmetric encryption handler."""
    
    def __init__(self):
        """Initialize asymmetric encryption."""
        self.logger = logging.getLogger("security.asymmetric")
    
    def generate_key_pair(self, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.RSA_2048) -> Tuple[bytes, bytes]:
        """Generate a key pair.
        
        Args:
            algorithm: Asymmetric algorithm
            
        Returns:
            Tuple of (private_key, public_key) in PEM format
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library required for asymmetric encryption")
        
        if algorithm == EncryptionAlgorithm.RSA_2048:
            key_size = 2048
        elif algorithm == EncryptionAlgorithm.RSA_4096:
            key_size = 4096
        else:
            raise ValueError(f"Unsupported asymmetric algorithm: {algorithm}")
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        
        public_key = private_key.public_key()
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def encrypt_with_public_key(self, data: Union[str, bytes], public_key_pem: bytes) -> bytes:
        """Encrypt data with public key.
        
        Args:
            data: Data to encrypt
            public_key_pem: Public key in PEM format
            
        Returns:
            Encrypted data
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library required for asymmetric encryption")
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        public_key = serialization.load_pem_public_key(public_key_pem, backend=default_backend())
        
        ciphertext = public_key.encrypt(
            data,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return ciphertext
    
    def decrypt_with_private_key(self, ciphertext: bytes, private_key_pem: bytes) -> bytes:
        """Decrypt data with private key.
        
        Args:
            ciphertext: Encrypted data
            private_key_pem: Private key in PEM format
            
        Returns:
            Decrypted data
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library required for asymmetric encryption")
        
        private_key = serialization.load_pem_private_key(private_key_pem, password=None, backend=default_backend())
        
        plaintext = private_key.decrypt(
            ciphertext,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        return plaintext
    
    def sign_data(self, data: Union[str, bytes], private_key_pem: bytes) -> bytes:
        """Sign data with private key.
        
        Args:
            data: Data to sign
            private_key_pem: Private key in PEM format
            
        Returns:
            Signature
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library required for signing")
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        private_key = serialization.load_pem_private_key(private_key_pem, password=None, backend=default_backend())
        
        signature = private_key.sign(
            data,
            asym_padding.PSS(
                mgf=asym_padding.MGF1(hashes.SHA256()),
                salt_length=asym_padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return signature
    
    def verify_signature(self, data: Union[str, bytes], signature: bytes, public_key_pem: bytes) -> bool:
        """Verify signature with public key.
        
        Args:
            data: Original data
            signature: Signature to verify
            public_key_pem: Public key in PEM format
            
        Returns:
            True if signature is valid
        """
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("Cryptography library required for signature verification")
        
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        try:
            public_key = serialization.load_pem_public_key(public_key_pem, backend=default_backend())
            
            public_key.verify(
                signature,
                data,
                asym_padding.PSS(
                    mgf=asym_padding.MGF1(hashes.SHA256()),
                    salt_length=asym_padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception:
            return False


class CryptographyManager:
    """Main cryptography manager."""
    
    def __init__(self, key_store: Optional[KeyStore] = None):
        """Initialize cryptography manager.
        
        Args:
            key_store: Key storage backend
        """
        self.key_store = key_store or MemoryKeyStore()
        self.hasher = Hasher()
        self.symmetric = SymmetricEncryption(self.key_store)
        self.asymmetric = AsymmetricEncryption()
        self.logger = logging.getLogger("security.crypto")
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate a secure random token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            URL-safe base64 encoded token
        """
        return secrets.token_urlsafe(length)
    
    def generate_secure_password(self, length: int = 16) -> str:
        """Generate a secure random password.
        
        Args:
            length: Password length
            
        Returns:
            Secure password
        """
        import string
        
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        return password
    
    def constant_time_compare(self, a: Union[str, bytes], b: Union[str, bytes]) -> bool:
        """Constant-time string comparison.
        
        Args:
            a: First string
            b: Second string
            
        Returns:
            True if strings are equal
        """
        if isinstance(a, str):
            a = a.encode('utf-8')
        if isinstance(b, str):
            b = b.encode('utf-8')
        
        return hmac.compare_digest(a, b)
    
    def derive_key_from_password(self, 
                                password: str, 
                                salt: Optional[bytes] = None,
                                kdf: KeyDerivationFunction = KeyDerivationFunction.PBKDF2,
                                iterations: int = 100000) -> Tuple[bytes, bytes]:
        """Derive encryption key from password.
        
        Args:
            password: Password
            salt: Optional salt (generated if not provided)
            kdf: Key derivation function
            iterations: Number of iterations
            
        Returns:
            Tuple of (derived_key, salt)
        """
        if salt is None:
            salt = secrets.token_bytes(16)
        
        if kdf == KeyDerivationFunction.PBKDF2:
            if not CRYPTOGRAPHY_AVAILABLE:
                # Fallback implementation
                import hashlib
                return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations), salt
            
            kdf_instance = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=iterations,
                backend=default_backend()
            )
            key = kdf_instance.derive(password.encode())
            return key, salt
        
        elif kdf == KeyDerivationFunction.SCRYPT:
            if not CRYPTOGRAPHY_AVAILABLE:
                raise RuntimeError("Cryptography library required for Scrypt")
            
            kdf_instance = Scrypt(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                n=2**14,
                r=8,
                p=1,
                backend=default_backend()
            )
            key = kdf_instance.derive(password.encode())
            return key, salt
        
        else:
            raise ValueError(f"Unsupported KDF: {kdf}")
    
    def secure_delete(self, data: Union[str, bytes, bytearray]) -> None:
        """Securely delete sensitive data from memory.
        
        Args:
            data: Data to securely delete
        """
        if isinstance(data, str):
            # Can't securely delete immutable strings in Python
            self.logger.warning("Cannot securely delete immutable string")
            return
        
        if isinstance(data, (bytes, bytearray)):
            # Overwrite with random data
            for i in range(len(data)):
                data[i] = secrets.randbits(8)


# Global cryptography manager instance
_crypto_manager = None


def get_crypto_manager(key_store: Optional[KeyStore] = None) -> CryptographyManager:
    """Get global cryptography manager instance.
    
    Args:
        key_store: Optional key store (only used on first call)
        
    Returns:
        Cryptography manager instance
    """
    global _crypto_manager
    if _crypto_manager is None:
        _crypto_manager = CryptographyManager(key_store)
    return _crypto_manager


def initialize_crypto_manager(key_store: KeyStore) -> CryptographyManager:
    """Initialize global cryptography manager.
    
    Args:
        key_store: Key storage backend
        
    Returns:
        Cryptography manager instance
    """
    global _crypto_manager
    _crypto_manager = CryptographyManager(key_store)
    return _crypto_manager