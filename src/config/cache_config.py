"""
================================================================================
FILE: src/config/cache_config.py
================================================================================

PURPOSE:
    Cache-specific configuration. Defines cache key patterns, TTLs, and
    Redis-specific settings. Centralizes cache logic configuration.

WORKFLOW:
    1. Define cache key patterns (for consistent key generation)
    2. Define cache TTLs (how long to cache different data)
    3. Define cache strategies (when to cache)
    4. Use throughout cache logic for consistency

IMPORTS:
    - None (just constants and config classes)

INPUTS:
    - None (loaded from module)

OUTPUTS:
    - Cache configuration dicts and classes

CACHE CATEGORIES:
    1. Cache Key Patterns
       - Query cache: "{user_id}:query:{query_hash}"
       - Session cache: "{user_id}:{session_id}"
       - Document summary: "{user_id}:summary:{doc_hash}"
    
    2. TTL Configuration (Time-To-Live)
       - Query cache: 1 hour (3600s) - answers change slowly
       - Session cache: 24 hours (86400s) - long-lived sessions
       - Document summary: 7 days (604800s) - docs rarely change
    
    3. Cache Strategy
       - Always cache: queries (high repeated rate)
       - Always cache: summaries (expensive computation)
       - Optional cache: sessions (depends on user preference)
    
    4. Redis Configuration
       - Key prefix: "rag:" (namespace)
       - Eviction policy: "allkeys-lru" (LRU eviction)
       - Max memory: "100mb" (per config)

KEY FACTS:
    - Cache keys are templates (user_id, doc_hash substituted)
    - TTLs vary by data type (frequently accessed = longer TTL)
    - Cache prefix prevents collisions with other services
    - Consistent key generation prevents cache misses

CACHE HIT RATE TARGETS:
    - Query cache: >70% (repeated queries from same user)
    - Session cache: >80% (same session accessed multiple times)
    - Document cache: >90% (docs don't change often)

FUTURE SCOPE (Phase 2+):
    - Add cache warming (pre-populate cache)
    - Add cache invalidation patterns
    - Add per-user cache policies
    - Add cache metrics (hit rate, memory usage)
    - Add distributed cache (multi-region)

TESTING ENVIRONMENT:
    - Use shorter TTLs for faster tests
    - Disable caching for deterministic tests

PRODUCTION DEPLOYMENT:
    - Use appropriate TTLs based on data update frequency
    - Monitor cache hit rates
    - Alert if hit rate drops
    - Adjust TTLs based on metrics
"""

# ================================================================================
# CACHE KEY PATTERNS
# ================================================================================

class CacheKeyPattern:
    """Cache key pattern templates"""
    
    # Query result cache
    QUERY = "rag:user:{user_id}:query:{query_hash}"
    QUERY_SHORT = "u:{user_id}:q:{query_hash}"  # For short keys
    
    # Session cache
    SESSION = "rag:session:{user_id}:{session_id}"
    SESSION_SHORT = "s:{user_id}:{session_id}"
    
    # Document summary cache
    SUMMARY = "rag:summary:{user_id}:{doc_hash}"
    SUMMARY_SHORT = "sum:{user_id}:{doc_hash}"
    
    # Prefix for all cache keys
    PREFIX = "rag:"
    
    @staticmethod
    def query_key(user_id: str, query_hash: str) -> str:
        """Generate query cache key"""
        return CacheKeyPattern.QUERY.format(user_id=user_id, query_hash=query_hash)
    
    @staticmethod
    def session_key(user_id: str, session_id: str) -> str:
        """Generate session cache key"""
        return CacheKeyPattern.SESSION.format(user_id=user_id, session_id=session_id)
    
    @staticmethod
    def summary_key(user_id: str, doc_hash: str) -> str:
        """Generate summary cache key"""
        return CacheKeyPattern.SUMMARY.format(user_id=user_id, doc_hash=doc_hash)


# ================================================================================
# CACHE TTL CONFIGURATION (seconds)
# ================================================================================

class CacheTTL:
    """Cache TTL (Time-To-Live) values"""
    
    # Query cache: 1 hour (answers don't change frequently)
    # High repeated rate from same user = longer TTL
    QUERY_DEFAULT = 3600  # 1 hour
    QUERY_SHORT = 300  # 5 minutes (for testing)
    QUERY_LONG = 86400  # 24 hours
    
    # Session cache: 24 hours (sessions are long-lived)
    # User may access session at different times = longer TTL
    SESSION_DEFAULT = 86400  # 24 hours
    SESSION_SHORT = 1800  # 30 minutes (for testing)
    SESSION_LONG = 604800  # 7 days
    
    # Document summary: 7 days (documents rarely change)
    # High cost to compute, likely accessed multiple times = very long TTL
    SUMMARY_DEFAULT = 604800  # 7 days
    SUMMARY_SHORT = 3600  # 1 hour (for testing)
    SUMMARY_LONG = 2592000  # 30 days
    
    # Cache invalidation: immediate
    # When something changes, remove from cache
    INVALIDATE_IMMEDIATELY = 0


# ================================================================================
# CACHE STRATEGY CONFIGURATION
# ================================================================================

class CacheStrategy:
    """When and how to cache different data"""
    
    # Always cache (high probability of repeated access)
    CACHE_QUERIES = True
    CACHE_SUMMARIES = True
    
    # Optionally cache (depends on configuration)
    CACHE_SESSIONS = True
    
    # Cache configuration
    CACHE_READ_BEFORE_EXECUTE = True  # Check cache before running pipeline
    CACHE_WRITE_AFTER_EXECUTE = True  # Write result to cache after execution
    
    # Cache invalidation
    INVALIDATE_ON_DOC_UPDATE = True  # Remove summary cache when doc updates
    INVALIDATE_ON_CONFIG_CHANGE = True  # Remove all caches on config change


# ================================================================================
# REDIS CONFIGURATION
# ================================================================================

class RedisConfig:
    """Redis-specific configuration"""
    
    # Key namespace/prefix (prevents collisions with other services)
    KEY_PREFIX = "rag:"
    
    # Redis key patterns for scanning
    SCAN_PATTERN_QUERY = "rag:user:*:query:*"
    SCAN_PATTERN_SESSION = "rag:session:*"
    SCAN_PATTERN_SUMMARY = "rag:summary:*"
    
    # Connection settings
    SOCKET_CONNECT_TIMEOUT = 5  # seconds
    SOCKET_KEEPALIVE = True
    SOCKET_KEEPALIVE_INTERVAL = 60  # seconds
    
    # Eviction policy (when Redis is full)
    # allkeys-lru: Remove least recently used key
    EVICTION_POLICY = "allkeys-lru"
    
    # Max memory before eviction kicks in
    MAX_MEMORY_POLICY = "100mb"  # Can be tuned per environment
    
    # Connection pooling
    POOL_MAX_CONNECTIONS = 50
    POOL_MIN_CONNECTIONS = 5


# ================================================================================
# CACHE PERFORMANCE TARGETS
# ================================================================================

class CachePerformance:
    """Cache performance targets and metrics"""
    
    # Cache hit rate targets
    HIT_RATE_QUERY_TARGET = 0.70  # 70% (repeated queries)
    HIT_RATE_SESSION_TARGET = 0.80  # 80% (same session)
    HIT_RATE_SUMMARY_TARGET = 0.90  # 90% (docs don't change)
    
    # Alert thresholds (if lower than this, alert)
    HIT_RATE_QUERY_ALERT = 0.50  # Alert if <50%
    HIT_RATE_SESSION_ALERT = 0.60  # Alert if <60%
    HIT_RATE_SUMMARY_ALERT = 0.75  # Alert if <75%
    
    # Latency targets
    REDIS_HIT_LATENCY_TARGET_MS = 5
    REDIS_MISS_LATENCY_TARGET_MS = 10


# ================================================================================
# CACHE CONFIGURATION CLASS
# ================================================================================

class CacheConfig:
    """Consolidated cache configuration"""
    
    # Key patterns
    keys = CacheKeyPattern()
    
    # TTLs
    ttl = CacheTTL()
    
    # Strategy
    strategy = CacheStrategy()
    
    # Redis
    redis = RedisConfig()
    
    # Performance
    performance = CachePerformance()
    
    # ========================================================================
    # FUTURE EXTENSION (Phase 2+)
    # ========================================================================
    
    # TODO (Phase 2): Add cache warming configuration
    # TODO (Phase 2): Add cache invalidation patterns
    # TODO (Phase 2): Add per-user cache policies
    # TODO (Phase 2): Add distributed cache (multi-region)
    # TODO (Phase 2): Add cache metrics collection


# ================================================================================
# CONSOLIDATE CACHE CONFIG
# ================================================================================

CACHE_CONFIG = CacheConfig()
