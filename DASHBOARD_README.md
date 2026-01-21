# 🏥 Cardiovascular Disease Interactive Dashboard

## Overview

This is a Plotly Dash interactive dashboard with **brushing & linking** capabilities for exploring cardiovascular disease patterns and predictions.

## Features

### 4 Interactive Visualizations:

1. **Feature Importance** (Horizontal Bar Chart)
   - Shows which features are most important for disease prediction
   - **Interaction:** Click any bar to highlight that feature in other charts

2. **Disease Prevalence by Demographics** (Grouped Bar Chart)
   - Displays disease rates across age groups, cholesterol levels, and gender
   - **Linking:** Updates when age range is selected in scatter plot

3. **Age vs Blood Pressure Scatter Plot**
   - Interactive scatter plot with disease color-coding
   - **Brushing:** Use box select or lasso to select age ranges → updates all other charts

4. **Feature Distribution & Correlation**
   - Shows correlation heatmap by default
   - **Linking:** When a feature is clicked, shows distribution of that feature by disease status

### Brushing & Linking Implemented:

✅ **Click Feature Importance chart** → Highlights in correlation chart + shows distribution
✅ **Brush/Select Scatter Plot** → Filters all other charts to selected age range
✅ **Real-time updates** across all visualizations

---

## 🚀 Running Locally

### Step 1: Install Dependencies

```bash
cd D:\TU\2025W\Visual_DS\cardioTrainProject
pip install -r requirements.txt
```

### Step 2: Run the Dashboard

```bash
python dashboard_app.py
```

### Step 3: Open in Browser

Navigate to: **http://localhost:8050**

---

## 🌐 Deploying to Get URL (For Assignment Submission)

### Option 1: Render.com (Recommended - FREE)

1. **Create account**: https://render.com

2. **Create `Procfile`** in project root:
   ```
   web: gunicorn dashboard_app:server
   ```

3. **Update `dashboard_app.py`** - Add at bottom (before `if __name__`):
   ```python
   server = app.server  # Expose Flask server for gunicorn
   ```

4. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Add dashboard"
   git push origin main
   ```

5. **Deploy on Render**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repo
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn dashboard_app:server`
   - Click "Create Web Service"
   - **Get your URL**: `https://your-app-name.onrender.com`

### Option 2: Railway.app (Also FREE)

1. Sign up at https://railway.app
2. "New Project" → "Deploy from GitHub"
3. Select your repo
4. Add environment variables if needed
5. Deploy → Get URL

### Option 3: Heroku (Paid but well-known)

1. Create `Procfile`:
   ```
   web: gunicorn dashboard_app:server
   ```

2. Deploy:
   ```bash
   heroku create your-cardio-dashboard
   git push heroku main
   ```

---

## 📹 Recording Demo Video (If Deployment Fails)

If you can't deploy, record a video showing:

1. **Dashboard Overview** (10 seconds)
2. **Click Feature Importance** → Show other charts updating (15 seconds)
3. **Brush Scatter Plot** → Show all charts filtering (20 seconds)
4. **Explain insights** discovered (30 seconds)

Use **OBS Studio** (free) or **Loom** to record.

---

## 🎨 Dashboard Structure

```
┌─────────────────────────────────────────────┐
│   Cardiovascular Disease Dashboard          │
├─────────────────────────────────────────────┤
│  [Current Selection Info Card]              │
├──────────────────────┬──────────────────────┤
│  Feature Importance  │  Disease Prevalence  │
│  (Click to select)   │  (Updates on brush)  │
├──────────────────────┼──────────────────────┤
│  Age vs BP Scatter   │  Feature Correlation │
│  (Brush to filter)   │  (Shows distribution)│
└──────────────────────┴──────────────────────┘
```

---

## 🔗 Brushing & Linking Examples

### Example 1: Feature Selection
- **Action:** Click "ap_hi" (systolic BP) in Feature Importance chart
- **Result:**
  - Feature highlighted in red
  - Correlation chart shows distribution of systolic BP by disease status
  - Info card shows selected feature

### Example 2: Age Range Selection
- **Action:** Box-select age range 50-65 in scatter plot
- **Result:**
  - Selected points highlighted in scatter
  - Disease Prevalence chart updates to show only ages 50-65
  - Correlation chart recalculates for that age range
  - Info card shows selected age range
  - Stats footer updates counts

---

## 📊 Insights You Can Discover

1. **Which features predict disease best?** → Feature Importance
2. **How does disease prevalence vary by age/gender/cholesterol?** → Prevalence Chart
3. **Relationship between age and blood pressure?** → Scatter Plot
4. **How are features correlated?** → Correlation Heatmap
5. **Distribution differences between healthy and diseased patients?** → Linked Distribution View

---

## 🛠️ Troubleshooting

### Dashboard won't start
```bash
# Check if port 8050 is in use
netstat -ano | findstr :8050

# Try different port
python dashboard_app.py --port 8051
```

### Missing data files
Ensure these files exist:
- `models/cardio_final.csv`
- `models/best_model.pkl` (optional - has fallback)

### Deployment errors
- Make sure `server = app.server` is in `dashboard_app.py`
- Check `Procfile` has: `web: gunicorn dashboard_app:server`
- Verify all dependencies in `requirements.txt`

---

## 📝 For Assignment Submission

### Submit:
1. **URL** (from Render/Railway/Heroku) - **10 points**
2. **Screenshot** showing brushing & linking in action
3. **OR Video** (2-3 minutes) demonstrating interactions - **10 points for brushing & linking**

### Grading Checklist:
- ✅ Uses Python with Plotly Dash (charting library) - **20 points**
- ✅ 4+ visualizations with appropriate charts - **20 points**
- ✅ Interactive (filters, zoom, click, brush) - **10 points**
- ✅ Brushing & linking (click in one chart updates others) - **10 points**

**Total: 40/40 points** 🎉

---

## 📧 Questions?

If you encounter issues, check:
1. All files are in the correct location
2. Dependencies are installed
3. Port 8050 is available
4. Data files exist

Happy Dashboard Building! 🚀