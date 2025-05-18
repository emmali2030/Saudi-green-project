import streamlit as st
import ee
import geemap.foliumap as geemap
import folium
import base64
import warnings
import tempfile
import math
import time 
import streamlit.components.v1 as components
import pandas as pd
import branca.colormap as cm
from folium.plugins import MiniMap
from datetime import datetime, date, timedelta
import plotly.express as px
import os
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["service-account"],
    scopes=["https://www.googleapis.com/auth/earthengine.readonly"]
)

# تهيئة Earth Engine باستخدام بيانات الاعتماد
ee.Initialize(credentials=credentials, project="streamlit-ndvi-project-459419")

# ───────── الأصول الثابتة ─────────
KSA_ASSET = ee.FeatureCollection("projects/streamlit-ndvi-project-459419/assets/SAUDI")
AREAS_ASSET = ee.FeatureCollection("projects/streamlit-ndvi-project-459419/assets/REGIONS")
CITIES_ASSET = ee.FeatureCollection("projects/streamlit-ndvi-project-459419/assets/CITIES")
SOURCE_IDS = {
    "MODIS (500 m / 16 day)": "MODIS/061/MOD13A2",
    "Sentinel-2 (10 m / 5 day)": "COPERNICUS/S2_SR_HARMONIZED",
    "Landsat 8-9 (30 m / 16 day)": "LANDSAT/LC08/C02/T1_L2",
}
ALL_KSA = "المملكة العربية السعودية"
MIN_YEAR = 2020
ksa_geom = ee.FeatureCollection(KSA_ASSET).geometry()

# ───────── وظائف مساعدة ─────────
def b64(fp):
    with open(fp, "rb") as f:
        return base64.b64encode(f.read()).decode()

# مسار اللوجو الجديد
new_logo_path = "assets/KSA.png"
new_logo_base64 = b64(new_logo_path)  # تحويل اللوجو إلى base64

@st.cache_data(show_spinner=False)
def date_range(cid):
    """أقدم/أحدث تاريخ في المصدر، مع احترام حدّ 2020."""
    ic = ee.ImageCollection(cid)
    mn = ic.aggregate_min("system:time_start").getInfo()
    mx = ic.aggregate_max("system:time_start").getInfo()
    earliest = max(date(MIN_YEAR,1,1), datetime.utcfromtimestamp(mn/1000).date())
    latest = datetime.utcfromtimestamp(mx/1000).date()
    return earliest, latest

def ndvi_image(cid, start, end, _geom=None):
    _geom = _geom or ksa_geom  # استخدم المنطقة المختارة، أو المملكة ككل
    coll = (ee.ImageCollection(cid)
            .filterBounds(_geom)
            .filterDate(str(start), str(end)))

    # فحص للأخطاء الخاصة بـ Sentinel-2
    if "COPERNICUS" in cid:  # Sentinel-2
        coll = coll.map(lambda i: i.updateMask(i.select("QA60").Not()))  # إزالة السحب
        ndvi_coll = coll.map(lambda i: i.normalizedDifference(['B8', 'B4']).rename('NDVI'))
        img = ndvi_coll.mean()
        scale = 10
        print("تم تحميل بيانات Sentinel-2 بنجاح")

    # فحص للأخطاء الخاصة بـ Landsat
    elif "LANDSAT" in cid:  # Landsat 8-9
        coll = coll.map(lambda i: i.updateMask(i.select("pixel_qa").eq(0)))  # إزالة السحب
        ndvi_coll = coll.map(lambda i: i.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI'))
        img = ndvi_coll.mean()
        scale = 30
        print("تم تحميل بيانات Landsat بنجاح")

    # فحص لمصادر بيانات أخرى مثل MODIS
    elif "MODIS" in cid:
        img = coll.select('NDVI').map(lambda i: i.multiply(1/10000)).mean()
        scale = 500

    return img.clip(_geom), scale

# --------------------------------

def compute_high_ndvi_ratio(_img, _geom, scale:int, threshold:float=0):
    mask = _img.gt(threshold)
    area_img = mask.multiply(ee.Image.pixelArea()).rename("area").clip(_geom)
    stats = area_img.reduceRegion(
        ee.Reducer.sum(), geometry=_geom, scale=scale,
        maxPixels=1e13, bestEffort=True, tileScale=4
    ).getInfo()
    high_m2 = stats.get("area", 0)
    total_m2 = _geom.area().getInfo()
    return high_m2/total_m2*100

def compute_veg_area_km2(_img, _geom, scale:int, threshold:float=0):
    mask = _img.gt(threshold)
    area_img = mask.multiply(ee.Image.pixelArea()).rename("area").clip(_geom)
    stats = area_img.reduceRegion(
        ee.Reducer.sum(), geometry=_geom, scale=scale,
        maxPixels=1e13, bestEffort=True, tileScale=4
    ).getInfo()
    return stats.get("area", 0) / 1e6  # المساحة بالكيلومتر المربع

@st.cache_data
def get_time_series(cid: str, _geom, scale: int, start: date, end: date, region=None, city=None):
    coll = (ee.ImageCollection(cid)
            .filterBounds(_geom)
            .filterDate(str(start), str(end))
            .select("NDVI"))
    
    def feat(img):
        d = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        m = img.reduceRegion(ee.Reducer.mean(), _geom, scale).get("NDVI")
        return ee.Feature(None, {"date": d, "mean": m})

    feats = coll.map(feat).getInfo()["features"]
    
    # تحويل النتيجة إلى DataFrame
    df = pd.DataFrame({
        "date": [f["properties"]["date"] for f in feats],
        "mean_ndvi": [f["properties"]["mean"] for f in feats]
    })
    
    # إضافة الأعمدة الخاصة بالمدينة والمنطقة إذا تم تحديدهما
    if region:
        df["region"] = region
    if city:
        df["city"] = city

    return df
# --------------------------------
def _b64(fp):
    with open(fp, "rb") as f:
        return base64.b64encode(f.read()).decode()

def compute_ndvi_change(cid: str, start: date, end: date):
    window = 16 if "MOD13A2" in cid else 5
    nd_s, scale = ndvi_image(cid, start, start + timedelta(days=window), focus_geom)
    nd_e, _ = ndvi_image(cid, end, end + timedelta(days=window), focus_geom)
    return nd_e.subtract(nd_s), scale

def get_region_stats(img, scale):
    def compute_stats(feature):
        geom = feature.geometry()
        stats = img.reduceRegion(
            ee.Reducer.mean().combine(ee.Reducer.stdDev(), None, True),
            geom, scale, maxPixels=1e13, tileScale=4
        ).getInfo()
        return feature.set(stats)
    return areas.map(compute_stats).getInfo()["features"]

# ───────── واجهة وتصميم ─────────
custom_css = """
<style>
    .main .block-container {max-width: 95%; padding: 0 1rem;}
    .header-box {
    position: relative;
    height: 300px;
    background-size: cover;
    background-position: center;
    border-radius: 10px;
    margin-bottom: 20px;
}
.header-content {
    position: absolute;
    bottom: 30px;
    left: 50%;
    transform: translateX(-50%);
    text-align: center;
    color: white;
    text-shadow: 2px 2px 8px #000;
}
    .header-box {
        background: url('data:image/gif;base64,{gif}');
        background-size: cover;
        background-position: center;
        padding: 60px 20px 30px;
        margin-bottom: 10px;
        border-radius: 8px;
        text-align: center;
        position: relative;
        width: 100%;
    }
    .header-box img.logo {
        position: absolute;
        top: 20px;
        right: 40px;
        height: 130px;
    }
    .header-box h1 {
        font-family: 'Andalus';
        font-size: 55px;
        font-weight: bold;
        color: black;
        text-shadow: 2px 2px 8px white;
        margin: 0;
    }
    .header-box h1 a {display: none;}
    .subtitle {
        display: inline-block;
        background: #235d1e;
        color: #fff;
        padding: 10px 20px;
        border-radius: 6px;
        font-size: 22px;
        font-family: 'Andalus';
        margin-top: 10px;
    }
    .footer-note {
        font-family: 'Arabic Typesetting';
        font-size: 24px;
        text-align: center;
        margin: 10px 0 40px;
    }
    label {direction: rtl; display: block; text-align: right;}
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
    }
    .metric-title {
        font-size: 16px;
        font-weight: bold;
        color: #2e7d32;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1b5e20;
    }
    .section-title {
        font-size: 20px;
        font-weight: bold;
        color: #1b5e20;
        margin-bottom: 15px;
        padding-bottom: 5px;
        border-bottom: 2px solid #2e7d32;
        direction: rtl; /* ← من اليمين للشمال */        
        text-align: right;  
    }
    .map-container {
        height: 600px;
        margin-bottom: 20px;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .st-emotion-cache-1v0mbdj img {
        max-width: 100%;
        border-radius: 8px;
    }
    .st-emotion-cache-1629p8f h1 {
        margin-top: 0;
    }

/* Slider إصلاح جذري لتلوين الـ */
[data-testid="stSlider"] .stSlider > div > div:first-child {
    background: #1b5e20 !important; /* أخضر غامق */
    height: 12px !important;
    border-radius: 10px !important;
}

[data-testid="stSlider"] .stSlider > div > div:last-child {
    background: #a5d6a7 !important; /* أخضر فاتح */
    height: 12px !important;
    border-radius: 10px !important;
}

[data-testid="stSlider"] .handle {
    background-color: #1b5e20 !important;
    border: 2px solid white !important;
    width: 22px !important;
    height: 22px !important;
    border-radius: 50% !important;
    box-shadow: 0 0 5px rgba(0,0,0,0.2) !important;
}
.leaflet-control-splitmap {
    right: 20px !important;
    left: auto !important;
}
.leaflet-control-splitmap:after {
    content: "↔";
    font-size: 24px;
    color: #fff;
    text-shadow: 0 0 5px #000;
}
.leaflet-control-layers label {
    font-family: 'Arabic Typesetting' !important;
    font-size: 16px !important;
    direction: rtl !important;
    margin-right: 10px !important;
}
#iframe#swipe {
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
    .new-logo {
        position: absolute;
        top: 10px;  /* تحديد المسافة من الأعلى */
        left: 20px;  /* تحديد المسافة من اليسار */
        height: 60px;  /* تقليل حجم اللوجو */
        filter: drop-shadow(3px 3px 5px rgba(0,0,0,0.3));  /* إضافة الظل */
        border-radius: 8px;
    }
</style>
"""

st.set_page_config("السعودية الخضراء", layout="wide", page_icon="🌿",)
logo = _b64(r"assets\LOGO.png")
gif_path = r"assets\ndvi_header_banner.gif"
gif_bytes = open(gif_path, "rb").read()
gif_data_url = f"data:image/gif;base64,{base64.b64encode(gif_bytes).decode()}"
st.markdown(custom_css + f"""
<div class="header-box" style="background-image: url('{gif_data_url}');">
    <img class="logo" src="data:image/png;base64,{logo}" alt="Logo"/>
    <img class="new-logo" src="data:image/png;base64,{new_logo_base64}" alt="New Logo"/>
    <div class="header-content">
        <h1>السعودية الخضراء بعيون الأقمار الاصطناعية</h1>
        <div class="subtitle">🌱 منصة تفاعلية لمراقبة خُضرة المملكة.. دعمًا لمستقبل السعودية الخضراء</div>
    </div>
</div>
<div class="footer-note">
  هذا المشروع يأتي في إطار دعم مستهدفات <strong>مبادرة السعودية الخضراء</strong>
  عبر خريطة تفاعلية تترصد وتُحلل التغيرات المكانية والزمانية في الغطاء النباتي داخل المملكة العربية السعودية باستخدام مؤشرات الاستشعار عن بعد متعددة المصادر، لتحقيق مراقبة مؤشرات الاستدامة البيئية بكفاءة عالية
</div>
""", unsafe_allow_html=True)


# ───────── تحميل أشكال المملكة والمناطق والـمدن ─────────
ksa, areas, cities = (
    ee.FeatureCollection(KSA_ASSET),
    ee.FeatureCollection(AREAS_ASSET),
    ee.FeatureCollection(CITIES_ASSET),
)

# ───────── عناصر الفلترة ─────────
regions = [ALL_KSA] + sorted(areas.aggregate_array("PROV_NAME_").distinct().getInfo())
c5, c4, c3, c2, c1 = st.columns(5)
with c1:
    with st.container():
        st.markdown(
            "<div style='text-align:right; direction:rtl; font-size:20px; font-weight:bold; color:#000000; margin-bottom:5px; margin-top:-15px;'>المنطقة :</div>",
            unsafe_allow_html=True
        )
        region = st.selectbox("", regions, index=0, label_visibility="collapsed")

with c2:
    city_list = ([] if region == ALL_KSA else
                 cities.filter(ee.Filter.eq("PROV_NAME_", region))
                       .aggregate_array("Gov_name").distinct().sort().getInfo())
    with st.container():
        st.markdown(
            "<div style='text-align:right; direction:rtl; font-size:20px; font-weight:bold; color:#000000; margin-bottom:5px; margin-top:-15px;'>المدينة :</div>",
            unsafe_allow_html=True
        )
        city = st.selectbox("", [""] + city_list, index=0, label_visibility="collapsed")

with c3:
    with st.container():
        st.markdown(
            "<div style='text-align:right; direction:rtl; font-size:20px; font-weight:bold; color:#000000; margin-bottom:5px; margin-top:-15px;'>مصدر البيانات :</div>",
            unsafe_allow_html=True
        )
        src_name = st.selectbox("", list(SOURCE_IDS), index=0, label_visibility="collapsed")

cid = SOURCE_IDS[src_name]
earliest, latest = date_range(cid)

default_start = max(date(2023, 1, 1), earliest)
default_end = min(date(2023, 12, 31), latest)

with c4:
    with st.container():
        st.markdown(
            "<div style='text-align:right; direction:rtl; font-size:20px; font-weight:bold; color:#000000; margin-bottom:5px; margin-top:-15px;'>من :</div>",
            unsafe_allow_html=True
        )
        start = st.date_input("", default_start, earliest, latest, label_visibility="collapsed")

with c5:
    with st.container():
        st.markdown(
            "<div style='text-align:right; direction:rtl; font-size:20px; font-weight:bold; color:#000000; margin-bottom:5px; margin-top:-15px;'>إلى :</div>",
            unsafe_allow_html=True
        )
        end = st.date_input("", default_end, start, latest, label_visibility="collapsed")


start, end = max(start, earliest), min(end, latest)

# ✳️ تتبع التغييرات
current_filters = (region, city, src_name, start, end)

if "last_filters" not in st.session_state:
    st.session_state["last_filters"] = current_filters
    st.session_state["reload_trigger"] = True
elif st.session_state["last_filters"] != current_filters:
    st.session_state["last_filters"] = current_filters
    st.session_state["reload_trigger"] = True
else:
    st.session_state["reload_trigger"] = False

# ───────── NDVI Image ─────────
# أول حاجة: حدد focus_geom
if city:
    focus_fc = cities.filter(ee.Filter.eq("Gov_name", city))
    zoom = 9
elif region != ALL_KSA:
    focus_fc = areas.filter(ee.Filter.eq("PROV_NAME_", region))
    zoom = 7
else:
    focus_fc = ksa
    zoom = 5

focus_geom = focus_fc.geometry()
threshold = st.session_state.get("threshold", 0.1)  # قيمة افتراضية لو مش متحددة لسه

loading_msg = st.empty()  # نحجز مكان لرسالة النجاح بعد التحميل

if st.session_state["reload_trigger"]:
    with st.spinner("⏳ جاري تحميل الطبقات .. شكراً لانتظارك"):
        # حسابات NDVI وكل البيانات المطلوبة
        ndvi_img, src_scale = ndvi_image(cid, start, end, focus_geom)
        high_pct = compute_high_ndvi_ratio(ndvi_img, focus_geom, src_scale, threshold)
        veg_area = compute_veg_area_km2(ndvi_img, focus_geom, src_scale, threshold)
        df_ts = get_time_series(cid, focus_geom, src_scale, start, end)

    # ← نخرج من الـ spinner الأول، ثم نعرض رسالة النجاح بوضوح
    loading_msg.success("🎉 تم تحميل الطبقات بنجاح!")
    time.sleep(3)  # نسيبها 3 ثواني عشان المستخدم يشوفها
    loading_msg.empty()

else:
    # تحميل البيانات من الكاش
    ndvi_img, src_scale = ndvi_image(cid, start, end, focus_geom)
    high_pct = compute_high_ndvi_ratio(ndvi_img, focus_geom, src_scale, threshold)
    veg_area = compute_veg_area_km2(ndvi_img, focus_geom, src_scale, threshold)
    df_ts = get_time_series(cid, focus_geom, src_scale, start, end)



# تدرج لوني ديناميكي حسب المصدر
if "MODIS" in cid:
    vis = {
        'min': 0,
        'max': 0.8,
        'palette': ['#F5F5DC', "#206B02", "#062E02"],
        'opacity': 0.9
    }
else:
    vis = {
        'min': 0,
        'max': 0.6,
        'palette': ['#F5F5DC', "#206B02", "#062E02"],
        'opacity': 0.9
    }



# ───────── حساب المقاييس ─────────
high_pct = compute_high_ndvi_ratio(ndvi_img, focus_geom, src_scale, threshold)
veg_area = compute_veg_area_km2(ndvi_img, focus_geom, src_scale, threshold)
df_ts = get_time_series(cid, focus_geom, src_scale, start, end)

# ───────── إنشاء الخريطة الأساسية ─────────
m = geemap.Map(draw_control=False, measure_control=False,
               toolbar_control=False, fullscreen_control=True)
m.options.update({"maxBounds": [[15, 34], [32.5, 56.5]], "minZoom": 4.3})
m.setOptions("HYBRID")

# 🟢 ضبط نطاق التركيز
bounds = focus_geom.bounds().getInfo()['coordinates'][0]
m.fit_bounds([[bounds[0][1], bounds[0][0]], [bounds[2][1], bounds[2][0]]])

# 🟢 أضف طبقة NDVI
m.addLayer(ndvi_img, vis, "NDVI", True)

# ← إضافة مؤشر اتجاه الشمال على الخريطة الأساسية
north_icon_path = r"assets\NORTH.png"
encoded_arrow = _b64(north_icon_path)

north_html = f'''
<div style="
    position: absolute;
    top: 20px;
    right: 20px;
    z-index: 1000;
    width: 90px;
    height: 90px;">
    <img src="data:image/png;base64,{encoded_arrow}"
         style="
            width: 100%;
            height: 100%;
            object-fit: contain;
            transform: rotate(0deg);
            filter: drop-shadow(2px 2px 4px rgb(255,255,255));" />
</div>
'''

m.get_root().html.add_child(folium.Element(north_html))

if zoom != 4.3:
    m.addLayer(focus_fc.style(color="yellow", width=2, fillColor="00000000"), {}, "")
    m.addLayer(KSA_ASSET.style(color="darkgreen", width=1, fillColor="00000000"),{}, "حدود المملكة")

mini = MiniMap(toggle_display=True, minimized=True, position="bottomleft")
mini.add_to(m)
folium.Rectangle([[15, 34], [32.5, 56.5]],
                  fill=False, color="red", weight=2).add_to(mini)
cm.LinearColormap(["beige", "#aaffaa", "green"], vmin=-1, vmax=1).add_to(m)

# ───────── إنشاء خريطة التغيرات ─────────
with st.spinner("⏳ جاري حساب التغير في NDVI ..."):
    change_img, ch_scale = compute_ndvi_change(cid, start, end)
    vis_ch = {
    "min": -1,
    "max": 1,
    'palette': ['#F5F5DC', "#C5D69D", "#042401"],  # أحمر غامق → أصفر فاتح → أخضر غامق
    'Opacity': 1.0,
    "scale": ch_scale
    }


m_change = geemap.Map(draw_control=False, measure_control=False, toolbar_control=False, fullscreen_control=False)
m_change.options.update({"maxBounds": [[15, 34], [32.5, 56.5]], "minZoom": 6})
m_change.setOptions("SATELLITE")
m_change.fit_bounds([[bounds[0][1], bounds[0][0]], [bounds[2][1], bounds[2][0]]])
m_change.addLayer(change_img.clip(focus_geom), vis_ch, "ΔNDVI", True)
m_change.addLayer(focus_fc.style(color="black", width=2, fillColor="00000000"), {}, "حدود المنطقة")

# ───────── عرض Streamlit ─────────
map_col, mid_col, right_col = st.columns([2, 2, 2])  # تم تعديل نسب الأعمدة هنا

with map_col:
    st.markdown('<div class="section-title">🗺 الخريطة الأساسية (NDVI)</div>', unsafe_allow_html=True)
    m.to_streamlit(height=540)

    # ───────── تفسير المؤشر ─────────
    with st.expander('ℹ️ تـفسير مؤشر الغطاء النباتى'):
        st.markdown("""
        <div style="font-size: 18px; font-weight: bold; text-align: right; line-height: 1.6; direction: rtl; padding: 10px;">
            <p><strong>مؤشر NDVI (Normalized Difference Vegetation Index)</strong> هو مقياس لصحة النباتات وكثافتها:</p>
            <p><strong>قيم من -1 إلى 0:</strong> مناطق غير نباتية (ماء، صحراء)</p>
            <p><strong>قيم من 0 إلى 0.3:</strong> نباتات ضعيفة أو متفرقة (لون بيج مثل <span style="color:#F5F5DC; font-weight: bold;">هـذا</span>)</p>
            <p><strong>قيم من 0.3 إلى 0.6:</strong> نباتات متوسطة الكثافة (لون أخضر فاتح مثل <span style="color:#206B02; font-weight: bold;">هـذا</span>)</p>
            <p><strong>قيم أعلى من 0.6:</strong> نباتات كثيفة وصحية (لون أخضر غامق مثل <span style="color:#062E02; font-weight: bold;">هـذا</span>)</p>
            <p><strong>تفسير التغيرات:</strong></p>
            <p><span style="color:#F5F5DC; font-weight: bold;">اللون البيـج</span>: انخفاض في الغطاء النباتي</p>
            <p><span style="color:#206B02; font-weight: bold;">اللون الأخـضر الفاتح</span>: تغير طفيف</p>
            <p><span style="color:#062E02; font-weight: bold;">اللون الأخـضر الغامق</span>: زيادة في الغطاء النباتي</p>
        </div>
        """, unsafe_allow_html=True)

    # تصدير البيانات بناءً على الفلاتر
    with st.container():
        if st.button("📥 تصدير البيانات"):
            # تصدير البيانات المفلترة باستخدام الفلاتر التي تم تحديدها
            filtered_df = df_ts[df_ts['date'].between(str(start), str(end))]
            
            # إذا كان هناك مدينة أو منطقة تم تحديدها، نفلتر بناءً عليها
            if 'region' in filtered_df.columns and region != ALL_KSA:
                filtered_df = filtered_df[filtered_df['region'] == region]
            
            if 'city' in filtered_df.columns and city:
                filtered_df = filtered_df[filtered_df['city'] == city]

            # استخدام العتبة المحددة (threshold) لتصفية البيانات إذا لزم الأمر
            filtered_df = filtered_df[filtered_df['mean_ndvi'] > threshold]

            # قسمة القيم على 10000 لتصحيح الأرقام (إذا كانت كبيرة جدًا)
            filtered_df['mean_ndvi'] = filtered_df['mean_ndvi'] / 10000  # قسم القيم على 10000
            
            # تحميل البيانات
            csv = filtered_df.to_csv(index=False).encode('utf-8')

            # عرض زر لتحميل البيانات
            st.download_button(
                label="تنزيل ملف CSV",
                data=csv,
                file_name="filtered_ndvi_data.csv",
                mime="text/csv"
            )

with mid_col:
    st.markdown("""
        <div style="text-align: right; font-size: 20px;
                    font-weight: bold; color: #1b5e20; margin-bottom: 0px;">
            <span style="font-size: 20px;">: </span> حساب <span style="color:#1b5e20">عتبة الخضرة</span> 🖉
        </div>
    """, unsafe_allow_html=True)

    # 🟢 تخزين قيمة العتبة داخل session_state
    st.session_state["threshold"] = st.slider(
        "", min_value=0.0, max_value=1.0, value=0.1, step=0.05, label_visibility="collapsed"
    )

    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">قيمة العتبة المستخدمة</div>
        <div class="metric-value">{st.session_state['threshold']}</div>
    </div>
    """, unsafe_allow_html=True)

    # 🗓️ حساب مدة الفترة الزمنية بوحدات مختلفة
    days_diff = (end - start).days + 1
    weeks_diff = math.floor(days_diff / 7)
    months_diff = math.floor(days_diff / 30.44)  # متوسط عدد أيام الشهر

    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">مدة الفترة المختارة</div>
            <div class="metric-value" style="direction: rtl;">
                🗓️ {days_diff} يوم<br>
                📅 {weeks_diff} أسبوع<br>
                📆 {months_diff} شهر
            </div>
        </div>
    """, unsafe_allow_html=True)


    st.markdown('<div class="section-title">📊 إحصائيات الخضرة</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">نسبة الخضرة</div>
        <div class="metric-value" style="direction: rtl;">                
        <div class="metric-value">{high_pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # تنسيق الرقم مع فواصل الآلاف
    formatted_area = "{:,.2f}".format(veg_area)

    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-title">مساحة الخضرة</div>
            <div class="metric-value" style="direction: rtl;">
                {formatted_area} كم²
            </div>
        </div>
    """, unsafe_allow_html=True)


    st.markdown('<div class="section-title">📈 تطور المؤشر</div>', unsafe_allow_html=True)
        # تعديل "تطور المؤشر" باستخدام Plotly
    df_ts['date'] = pd.to_datetime(df_ts['date'])  # تأكد من أن التاريخ هو نوع `datetime`

        # فلترة البيانات بناءً على العتبة التي يحددها المستخدم
    filtered_df = df_ts[df_ts['mean_ndvi'] > st.session_state['threshold']]

    
    # تصغير الأرقام عن طريق تقسيمها على 10000 إذا كانت كبيرة
    filtered_df['mean_ndvi'] = filtered_df['mean_ndvi'] / 10000  # قسم القيم على 10000 لتقليل الأرقام

    # إنشاء الرسم البياني باستخدام Plotly
    fig = px.line(filtered_df, x='date', y='mean_ndvi')

    # تخصيص الشكل (تصميم احترافي)
    fig.update_layout(
        xaxis_title="التــاريـخ",
        yaxis_title="مـؤشـر الـغطاء الـنباتى",
        font=dict(family="Arial, sans-serif", size=12, color="black"),
        title_font=dict(size=18, family="Verdana, sans-serif", color='rgb(26, 13, 171)'),  # العنوان الرئيسي
        xaxis=dict(
            showgrid=True, 
            gridcolor='lightgray',
            title_font=dict(size=16, color="black")  # تخصيص حجم الخط ولونه لمحور x
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='lightgray',
            title_font=dict(size=16, color="black")  # تخصيص حجم الخط ولونه لمحور y
        ),
        title='',  # هنا حذفنا العنوان
        plot_bgcolor='#F8F9FA',  # إضافة خلفية فاتحة للرسم البياني
        paper_bgcolor='#F8F9FA',  # إضافة خلفية فاتحة للورقة (المساحة حول الرسم البياني)
        margin=dict(t=30, b=30, l=30, r=30),  # تعديل الهوامش (اختياري)
        showlegend=False  # إخفاء الأسطورة إذا كانت غير ضرورية        
    )

    fig.update_traces(line=dict(color='green', width=3))

    # عرض الرسم البياني التفاعلي في Streamlit
    st.plotly_chart(fig, use_container_width=True)


# ───────── داخل قسم with right_col: ─────────
with right_col:
    st.markdown('<div class="section-title">🕓 خريطة التغيرات (تغير الغطاء النباتي عبر الزمن)</div>', unsafe_allow_html=True)
    
    with st.spinner("⏳ جاري إعداد خريطة المقارنة..."):
        # حساب الفترات الزمنية
# أول 16 يوم من الفترة
        ndvi_start, scale_start = ndvi_image(cid, start, start + timedelta(days=16), focus_geom)

# آخر 16 يوم من الفترة
        ndvi_end, scale_end = ndvi_image(cid, end - timedelta(days=16), end, focus_geom)

    
        # احسب الإحداثيات المناسبة من focus_geom
        bounds = focus_geom.bounds().getInfo()["coordinates"][0]
        southwest = [bounds[0][1], bounds[0][0]]
        northeast = [bounds[2][1], bounds[2][0]]

        # أنشئ الخريطة بناءً على المنطقة المحددة
        dual_map = folium.plugins.DualMap(
            tiles=None,
            control_scale=True,
            draw_control=False, measure_control=False,
            toolbar_control=False, fullscreen_control=True
        )

        # Add Esri Satellite basemap to both sides
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False,
            control=False,
            draw_control=False, measure_control=False,
            toolbar_control=False, fullscreen_control=True           
        ).add_to(dual_map.m1)

        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri Satellite',
            overlay=False,
            control=False,
            draw_control=False, measure_control=False,
            toolbar_control=False, fullscreen_control=True            
        ).add_to(dual_map.m2)
        dual_map.fit_bounds([southwest, northeast])

    
        # توليد روابط الصور
        vis_params = {
            'min': -1,
            'max': 1,
            'palette': ['#F5F5DC', "#C5D69D", "#042401"],
            'dimensions': 1024,
            'format': 'png',
            'Opacity': 1.0
        }
        url_start = ndvi_start.getThumbUrl(vis_params)
        url_end = ndvi_end.getThumbUrl(vis_params)

        # ↓↓↓↓↓ هذا الجزء بالذات لازم يدخل جوة
        focus_fc_geojson = focus_fc.getInfo()
        folium.GeoJson(
            data=focus_fc_geojson,
            name='الحدود',
            style_function=lambda x: {
                'color': '#FFFF0080',  # أصفر شفاف
                'weight': 2,
                'fillColor': 'transparent',
                'fillOpacity': 0
            }
        ).add_to(dual_map)
    
        folium.raster_layers.ImageOverlay(
            image=url_start,
            bounds=[southwest, northeast],
            name='الفترة الأولى',
            opacity=1.0,
            draw_control=False, measure_control=False,
            toolbar_control=False, fullscreen_control=True
        ).add_to(dual_map.m1)

        folium.raster_layers.ImageOverlay(
            image=url_end,
            bounds=[southwest, northeast],
            name='الفترة الثانية',
            opacity=1.0,
            draw_control=False, measure_control=False,
            toolbar_control=False, fullscreen_control=True
        ).add_to(dual_map.m2)

    # ✨ نصوص داخل الخريطتين
    # ✅ بداية الفترة داخل الخريطة الأولى
        folium.Marker(
            location=[northeast[0] + 6.0, northeast[0] + 13.0],
            icon=folium.DivIcon(html=f'''
            <div style="font-size: 13px; font-weight: bold;
                    background-color: #2e7d32; color: white;
                    padding: 5px 10px; border-radius: 6px;
                    display: inline-block; line-height: 1.2;
                    box-shadow: 1px 1px 6px rgba(0,0,0,0.3);">
                بداية الفترة - {start.strftime('%Y-%m-%d')}
            </div>
        ''')
    ).add_to(dual_map.m1)

    # ✅ نهاية الفترة داخل الخريطة الثانية
        folium.Marker(
            location=[northeast[0] + 6.0, northeast[0] + 13.0],
            icon=folium.DivIcon(html=f'''
            <div style="font-size: 13px; font-weight: bold;
                    background-color: #b71c1c; color: white;
                    padding: 5px 10px; border-radius: 6px;
                    display: inline-block; line-height: 1.2;
                    box-shadow: 1px 1px 6px rgba(0,0,0,0.3);">
            نهاية الفترة - {end.strftime('%Y-%m-%d')}
            </div>
        ''')
    ).add_to(dual_map.m2)


        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            dual_map.save(f.name)# تقسيم العرض إلى 3 أعمدة: بداية ← الخريطة ← نهاية
        components.html(open(f.name, 'r', encoding='utf-8').read(), height=360)


    st.markdown('<div class="section-title">🎥 الخريطة المتحركة</div>', unsafe_allow_html=True)

    # مسار مجلد الـ GIF المحلي
    gif_folder_path = r"assets\GIF"

    # تحديد ملفات الـ GIF الخاصة بكل منطقة
    gif_files = {
        "الرياض": os.path.join(gif_folder_path, "الرياض_final.gif"),
        "مكة المكرمة": os.path.join(gif_folder_path, "مكة المكرمة_final.gif"),
        "عسير": os.path.join(gif_folder_path, "عسير_final.gif"),
        "تبوك": os.path.join(gif_folder_path, "تبوك_final.gif"),
        "المدينة": os.path.join(gif_folder_path, "المدينة_final.gif"),
        "السعودية كاملة": os.path.join(gif_folder_path, "السعودية_final.gif"),
        "المنطقة الشرقية": os.path.join(gif_folder_path, "المنطقة الشرقية_final.gif"),
        "الحدود الشمالية": os.path.join(gif_folder_path, "الحدود الشمالية_final.gif"),
        "القصيم": os.path.join(gif_folder_path, "القصيم_final.gif"),        
        "حائل": os.path.join(gif_folder_path, "حائل_final.gif"),
        "الباحة": os.path.join(gif_folder_path, "الباحة_final.gif"),
        "الجوف": os.path.join(gif_folder_path, "الجوف_final.gif"),
        "جازان": os.path.join(gif_folder_path, "جازان_final.gif"),
        "نجران": os.path.join(gif_folder_path, "نجران_final.gif"),
    }

    # تحديد الـ GIF الذي سيتم اختياره بناءً على المنطقة
    selected_gif_path = gif_files.get(region, os.path.join(gif_folder_path, "السعودية_final.gif"))

    # التأكد من أن الملف موجود
    if os.path.exists(selected_gif_path):
        gif_base64 = _b64(selected_gif_path)  # تحويل الـ GIF إلى base64
        st.image(f"data:image/gif;base64,{gif_base64}", use_container_width=True)
    else:
        st.error("الملف غير موجود في المسار المحدد!")


# ───────── إضافة التفاصيل في أسفل الصفحة ─────────
st.markdown("""
<div style="text-align: center; font-size: 20px; margin-top: 50px; font-weight: bold; color: #000000; direction: rtl; line-height: 0.7;">
    <p>اعداد طالبات الدكتوراة :<span style="font-weight: normal;"> ايمان القحطاني - نهى الحمراني - نوره اليوسف </span></p>
    <p>اشراف : <span style="font-weight: normal;"></span></p>
    <p><span style="font-weight: normal;"> أ.د. مفرح القرادي </span></p>

</div>
""", unsafe_allow_html=True)
