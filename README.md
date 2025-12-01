# 🚂 Railway Deployment Package - READ-ONLY API

This folder contains a **standalone, ready-to-deploy** version of the Media Semantic Search Engine optimized for Railway.

## ⚡ OPTIMIZED FOR LASER-FAST READ PERFORMANCE

This API has been configured as a **READ-ONLY** system with aggressive performance optimizations:
- 🔒 **No upload operations** - Security and performance first
- ⚡ **GZip compression** - 70-90% bandwidth reduction
- 🚀 **Aggressive caching** - Up to 100x faster for cached responses
- 🎯 **Optimized CORS** - Reduced preflight overhead

See `READ_ONLY_OPTIMIZATIONS.md` for complete details.

## 📦 What's Included

This package is completely self-contained:

```
railway/
├── backend/                    # Complete backend application
│   ├── routers/               # API route handlers
│   ├── models/                # Pydantic schemas
│   ├── core/                  # Core utilities
│   ├── main.py                # FastAPI application
│   ├── dependencies.py        # Dependency injection
│   ├── embedding_service.py   # AI model service
│   ├── database_service.py    # Supabase service
│   └── search_engine.py       # Search orchestration
├── config/                    # Configuration
│   ├── settings.py            # Model registry
│   └── user_config.py         # Config management
├── Procfile                   # Railway process definition
├── runtime.txt                # Python version (3.11)
├── requirements.txt           # Optimized dependencies (CPU-only PyTorch)
├── .env.example              # Environment variables template
└── README.md                  # This file
```

## 🚀 Quick Deploy to Railway

### Option 1: Drag & Drop This Folder

1. **Copy this entire `railway/` folder** to a new repository
2. Push to GitHub (or GitLab, etc.)
3. Go to [Railway](https://railway.app)
4. Click "New Project" → "Deploy from GitHub repo"
5. Select your repository
6. Railway will automatically detect and deploy!

### Option 2: Railway CLI

```bash
cd railway/
railway login
railway init
railway up
```

## ⚙️ Configuration

### 1. Set Environment Variables in Railway

Go to your Railway project → Variables, and add:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key-here
MODEL_KEY=clip-vit-base-patch32
VECTOR_DIMENSION=512
```

### 2. Set Up Supabase Database

1. Create a Supabase project at [supabase.com](https://supabase.com)
2. Go to SQL Editor
3. Run the schema (get it from `/api/v1/schema` endpoint after deployment)
4. Create a storage bucket named `media-files`
5. Make the bucket public

## 🎯 Optimizations for Railway

This deployment package is optimized specifically for Railway:

- **CPU-only PyTorch** (~200MB vs ~2GB with GPU support)
- **Streamlined dependencies** - No desktop UI components
- **Auto-configured port** - Uses Railway's `$PORT` environment variable
- **Fast cold starts** - Minimal dependency footprint

## 📊 What Gets Deployed

The Procfile tells Railway to run:

```
web: uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
```

This starts the FastAPI server on Railway's assigned port.

## 🔍 API Endpoints (READ-ONLY)

Once deployed, your API will be available at: `https://your-app.up.railway.app`

### Available Endpoints:
- `GET /` - Health check (shows read-only mode status)
- `GET /docs` - Interactive API documentation
- `GET /api/v1/health` - Detailed health check with optimization info
- `POST /api/v1/search/text` - Search by text (primary use case)
- `GET /api/v1/search/text?q=query` - Search by text (GET method)
- `POST /api/v1/search/image` - Search by image
- `POST /api/v1/search/combined` - Multimodal search (text + image)
- `GET /api/v1/media` - List all media with pagination
- `GET /api/v1/media/{id}` - Get single media item
- `GET /api/v1/stats` - Collection statistics
- `GET /api/v1/models` - Available AI models

### ❌ Removed Endpoints:
- ~~`POST /api/v1/media/upload`~~ - Upload disabled for read-only mode
- ~~`PUT /api/v1/media/{id}/description`~~ - Updates disabled
- ~~`DELETE /api/v1/media/{id}`~~ - Deletes disabled
- ~~`POST /api/v1/admin/*`~~ - Admin operations disabled

## 🐛 Troubleshooting

### Build Fails

- Check that `requirements.txt` uses the CPU-only PyTorch index
- Verify Python 3.11 is specified in `runtime.txt`

### API Returns 503

- Ensure environment variables are set correctly
- Check that Supabase credentials are valid
- Verify the database schema has been applied

### Slow Cold Starts

- This is normal on Railway's free tier
- Consider upgrading to a paid plan for better performance

## 📚 Documentation

- **API Docs:** `https://your-app.up.railway.app/docs`
- **Main Project:** See parent repository for full documentation

## 🔄 Updating

To update your Railway deployment:

1. Make changes to files in this `railway/` folder
2. Commit and push to your repository
3. Railway will automatically redeploy

## 💡 Tips

- Monitor logs in Railway dashboard to debug issues
- Use Railway's metrics to track API usage
- Consider adding custom domain in Railway settings
- Set up Railway's PostgreSQL database for even better performance

## 🆘 Need Help?

- Check Railway's [documentation](https://docs.railway.app)
- Visit the main project repository for issues
- Review API docs at `/docs` endpoint

---

**Ready to deploy?** Just copy this folder to a new repo and connect it to Railway!
