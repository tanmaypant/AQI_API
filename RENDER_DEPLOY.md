# Render Deployment Guide for AQI Anomaly Detection API

## Quick Deploy to Render (5 minutes)

### Step 1: Push Code to GitHub

1. **Create a new repository on GitHub** (if you haven't already)
   - Go to https://github.com/new
   - Name it: `aqi-anomaly-api`
   - Make it Public or Private

2. **Push your code:**
   ```bash
   cd "c:\Users\Tanmay\Desktop\sem 7 project"
   git init
   git add .
   git commit -m "Initial commit - AQI Anomaly Detection API"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/aqi-anomaly-api.git
   git push -u origin main
   ```

### Step 2: Deploy on Render

1. **Go to Render**: https://render.com
2. **Sign up/Login** (can use GitHub account)
3. **Click "New +"** → **"Web Service"**
4. **Connect your GitHub repository**: `aqi-anomaly-api`
5. **Configure the service:**
   - **Name**: `aqi-anomaly-api`
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: (leave blank)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`

6. **Click "Create Web Service"**

### Step 3: Wait for Deployment

- Render will automatically build and deploy your API
- Takes about 2-3 minutes
- You'll get a URL like: `https://aqi-anomaly-api.onrender.com`

### Step 4: Test Your Deployed API

Once deployed, test it:
```bash
curl -X POST "https://YOUR-APP.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"pm25": 180, "pm10": 250, "co": 1.2, "no2": 45, "aqi": 350}'
```

Or visit: `https://YOUR-APP.onrender.com/docs`

---

## Files Created for Deployment

✅ **Procfile** - Tells Render how to start the app
✅ **requirements.txt** - Python dependencies

---

## Important Notes

⚠️ **Free Tier Limitations:**
- App sleeps after 15 minutes of inactivity
- First request after sleep takes ~30 seconds
- 750 hours/month free

💡 **To keep it always active:** Upgrade to paid tier ($7/month)

---

## Troubleshooting

**If deployment fails:**
1. Check build logs in Render dashboard
2. Ensure all files are committed to GitHub
3. Verify `requirements.txt` has all dependencies

**If API doesn't respond:**
- Free tier apps sleep - first request is slow
- Check Render logs for errors

---

## Your API Endpoints (Once Deployed)

- **Docs**: `https://YOUR-APP.onrender.com/docs`
- **Health**: `https://YOUR-APP.onrender.com/health`
- **Predict**: `POST https://YOUR-APP.onrender.com/predict`
- **Model Info**: `https://YOUR-APP.onrender.com/model-info`

Share the `/docs` URL with your friend! 🚀
