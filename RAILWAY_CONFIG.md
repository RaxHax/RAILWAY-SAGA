# Railway Deployment Configuration Guide

This document explains how to configure container stopping, graceful shutdown, and optimize Railway deployments for the Media Search Engine.

## Container Lifecycle Configuration

### Graceful Shutdown

The application is configured for graceful shutdown with a 30-second timeout. This ensures:

1. **Active requests complete**: Running searches finish processing
2. **Resources cleanup**: Models are cleared from memory, connections close
3. **Smooth restarts**: No abrupt terminations

**Configuration** (in `Procfile`):
```
--timeout-graceful-shutdown 30
```

### Keep-Alive Timeout

Set to 5 seconds to prevent hanging connections:
```
--timeout-keep-alive 5
```

## Resource Management

### Model Loading Strategy

The CLIP embedding model loads on first request (lazy loading) to:
- Reduce startup time
- Save memory when not in use
- Allow configuration changes before model initialization

### Shutdown Cleanup

During shutdown, the application:
1. Clears embedding models from memory
2. Empties CUDA cache (if using GPU)
3. Closes database connections
4. Logs all cleanup operations

See `backend/main.py:78-101` for implementation.

## Railway Environment Variables

### Required Configuration

```bash
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key-here  # Use service_role key, NOT anon key!

# Optional: Model Configuration
MODEL_KEY=clip-vit-base-patch32  # Default model

# Optional: Port Configuration
PORT=8080  # Railway sets this automatically
```

### Which Supabase Key to Use?

**Use the `service_role` key** (also called the secret key):

- ✅ **service_role key**: For backend servers (this is what you need!)
  - Full database access
  - Bypasses Row Level Security (RLS)
  - Required for storage operations
  - Safe in server environment (Railway)

- ❌ **anon key**: For client-side apps only
  - Limited by RLS policies
  - Designed for browsers/mobile apps
  - Will cause permission errors in your backend

**Where to find it:**
1. Supabase Dashboard → Settings → API
2. Look for "service_role" under Project API keys
3. Copy the secret key (starts with `eyJ...`)
4. Add to Railway environment variables

### Deployment Settings

Railway automatically sets:
- `RAILWAY_ENVIRONMENT`: Set to production/staging
- `PORT`: Dynamic port assignment
- Ephemeral storage: Logs stored in `/tmp/logs`

## Preventing Unnecessary Restarts

### Health Checks

The application provides health endpoints to prevent premature restarts:

```bash
# Basic health check
GET /api/v1/health

# Detailed health with model status
GET /api/v1/health/detailed
```

Configure Railway health checks:
1. Go to Settings → Deploy
2. Set Health Check Path: `/api/v1/health`
3. Set Initial Delay: 30 seconds (allows model loading)
4. Set Interval: 30 seconds

### Startup Behavior

**Expected startup sequence:**
1. Logging configuration initializes
2. FastAPI app starts
3. Lifespan startup runs
4. Search engine initializes (lazy - only if credentials present)
5. Model loads on first search request

**Startup logs should show:**
```
INFO: Started server process [1]
INFO: Waiting for application startup.
INFO: Initializing Media Search Engine...
WARNING: Supabase credentials not configured... (if not configured)
INFO: Application startup complete.
```

## Optimizing Container Performance

### Memory Management

The CLIP model requires ~500MB-1GB RAM:
- Use at least 1GB RAM on Railway
- Consider 2GB for production workloads
- Model stays in memory after first load

### Startup Time

Typical startup times:
- App startup: ~2 seconds
- First model load: ~10 seconds
- Subsequent requests: <100ms

### Persistent Storage

Railway uses ephemeral storage:
- Logs: `/tmp/logs/` (cleared on restart)
- No persistent file storage
- All media stored in Supabase Storage

## Troubleshooting Container Issues

### Issue: Container keeps restarting

**Symptoms:**
```
Stopping Container
INFO: Shutting down
INFO: Waiting for application shutdown.
```

**Possible causes:**
1. Health check failing (increase timeout)
2. Memory limit exceeded
3. Crash during model loading
4. Port binding issues

**Solutions:**
1. Check Railway logs for errors
2. Verify memory allocation (>1GB)
3. Ensure `PORT` env var is set correctly
4. Check health endpoint returns 200

### Issue: Model loads before logging configures

**Symptoms:**
```
19:46:34 | Loading embedding model...
19:46:45 | File logging active...
```

**This is normal!** The embedding service lazy-loads models:
1. Module imports happen first
2. Logging configures during import
3. Model loads on first use
4. Both timestamps reflect different operations

### Issue: Slow restarts

**Causes:**
- Model reloading on every restart
- No model caching between deployments

**Solutions:**
1. Ensure graceful shutdown (30s timeout configured)
2. Consider reducing deployment frequency
3. Use Railway's "Deploy from scratch" sparingly
4. Keep model in memory after first load

### Issue: Memory errors during shutdown

**Symptoms:**
```
ERROR: Error during resource cleanup...
```

**Solutions:**
1. Increase shutdown timeout if needed
2. Check if GPU cache clearing is failing
3. Verify torch is properly installed

## Advanced Configuration

### Custom Uvicorn Settings

Modify `Procfile` for custom settings:

```bash
# Workers (not recommended for model-heavy apps)
--workers 1

# Limit request size (images/videos)
--limit-max-requests 0

# Logging level
--log-level info

# Access logs (disable for performance)
--no-access-log
```

### Environment-Specific Config

```python
import os

if os.environ.get("RAILWAY_ENVIRONMENT") == "production":
    # Production settings
    LOG_LEVEL = "INFO"
    CACHE_TTL = 300
else:
    # Development settings
    LOG_LEVEL = "DEBUG"
    CACHE_TTL = 60
```

## Monitoring

### Key Metrics to Monitor

1. **Startup Time**: Should be <15 seconds
2. **Memory Usage**: Should be <2GB
3. **Request Latency**:
   - First request: ~10 seconds (model load)
   - Subsequent: <100ms
4. **Restart Frequency**: Should be rare (only on deploys)

### Logging

Logs are available:
- Railway Dashboard: Real-time logs
- File logs: `/tmp/logs/media-search-engine_TIMESTAMP.log`

**Important log patterns:**
- `Initializing Media Search Engine...` - Startup
- `Loading embedding model...` - Model load
- `Shutting down Media Search Engine...` - Graceful shutdown
- `Resource cleanup completed` - Clean shutdown

## Best Practices

1. **Don't restart unnecessarily**: Model reloading takes ~10 seconds
2. **Set proper health checks**: Prevents premature restarts
3. **Monitor memory**: Ensure adequate allocation
4. **Use environment variables**: Keep credentials out of code
5. **Enable graceful shutdown**: Prevents incomplete requests
6. **Check logs regularly**: Identify issues early

## Summary

Your container stopping configuration is now optimized with:

✅ 30-second graceful shutdown timeout
✅ Proper resource cleanup in lifespan handler
✅ Model memory management
✅ Health check endpoints
✅ Logging at all lifecycle stages
✅ Railway-specific optimizations

The "Stopping Container" behavior you're seeing is **normal** when:
- Deploying new code
- Railway restarts for updates
- Manual restarts via dashboard

As long as the shutdown is graceful and the app restarts successfully, this is expected behavior.
