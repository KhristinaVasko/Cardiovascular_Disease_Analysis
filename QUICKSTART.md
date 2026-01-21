# 🚀 Quick Start Guide - Dashboard

## Test Your Dashboard in 3 Steps

### Step 1: Install Dependencies (2 minutes)

Open Command Prompt/Terminal in your project folder:

```bash
cd D:\TU\2025W\Visual_DS\cardioTrainProject
pip install -r requirements.txt
```

### Step 2: Run Dashboard (10 seconds)

```bash
python dashboard_app.py
```

You should see:
```
Dash is running on http://0.0.0.0:8050/
 * Serving Flask app 'dashboard_app'
 * Debug mode: on
```

### Step 3: Open in Browser

Go to: **http://localhost:8050**

---

## 🎯 Test Brushing & Linking

### Test 1: Click Feature Importance
1. Click any bar in the **Feature Importance** chart (top-left)
2. Watch the **Feature Distribution** chart (bottom-right) update
3. The clicked feature turns RED
4. Info card shows selected feature

### Test 2: Brush Scatter Plot
1. In the **Age vs Blood Pressure** chart (bottom-left):
   - Click the "Box Select" tool (top-right of chart)
   - Drag to select a region of points
2. Watch ALL other charts update to show only that age range
3. Info card shows selected age range
4. Stats footer updates with filtered counts

### Test 3: Combine Both
1. Click a feature in Feature Importance
2. Then brush an age range in Scatter Plot
3. See how all 4 charts work together!

---

## ✅ What You Should See

**Dashboard Layout:**
```
┌─────────────────────────────────────────┐
│  🏥 Cardiovascular Disease Dashboard    │
├─────────────────────────────────────────┤
│  [Current Selection Info Card]          │
├──────────────────────┬──────────────────┤
│  Feature Importance  │  Disease         │
│  (horizontal bars)   │  Prevalence      │
│  CLICK ME →          │  (grouped bars)  │
├──────────────────────┼──────────────────┤
│  Age vs BP Scatter   │  Correlation /   │
│  BRUSH ME →          │  Distribution    │
└──────────────────────┴──────────────────┘
│  Stats: Showing X records...            │
└─────────────────────────────────────────┘
```

---

## 🎥 Record Demo Video (If Needed)

If deployment fails, record 2-minute video showing:

1. **Dashboard Overview** (5 sec)
2. **Click Feature → Show all charts update** (30 sec)
3. **Brush Scatter → Show linking** (30 sec)
4. **Combine both interactions** (30 sec)
5. **Explain one insight** (25 sec)

Use **OBS Studio** (free) or **Loom** to screen record.

---

## 🌐 Deploy to Get URL

See `DASHBOARD_README.md` for full deployment instructions.

**Recommended:** Render.com (FREE, 5 minutes)

1. Push to GitHub
2. Connect Render to GitHub repo
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `gunicorn dashboard_app:server`
5. Deploy → Get URL

---

## ❌ Troubleshooting

### Error: "Address already in use"
```bash
# Port 8050 is busy, try different port
python dashboard_app.py --port 8051
```

### Error: "No module named 'dash'"
```bash
pip install dash dash-bootstrap-components plotly
```

### Error: "Cannot find cardio_final.csv"
```bash
# Make sure you ran main.py first to generate the data
python main.py
```

### Dashboard loads but empty charts
- Check if `models/cardio_final.csv` exists
- Make sure you have data in the CSV file

---

## 📧 Need Help?

Check that:
1. ✅ You're in the correct folder (`cardioTrainProject/`)
2. ✅ All dependencies installed (`pip install -r requirements.txt`)
3. ✅ Data file exists (`models/cardio_final.csv`)
4. ✅ Port 8050 is not in use

---

## 🎉 Success Criteria

Your dashboard is ready for submission when:

- ✅ All 4 charts display properly
- ✅ Clicking Feature Importance updates other charts
- ✅ Brushing Scatter Plot filters all views
- ✅ Info card shows current selection
- ✅ Stats footer updates with counts
- ✅ No console errors

**Time to test:** 5 minutes
**Time to deploy:** 10 minutes
**Total:** ~15 minutes to submission-ready dashboard! 🚀