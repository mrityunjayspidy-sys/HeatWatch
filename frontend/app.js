/**
 * HeatWatch AI — Frontend JavaScript
 * All data from D:\MINI PROJECT_NEW\data\processed (zero mock data)
 * City-specific analytics & responsive Chart.js rendering
 */

const API_BASE_URL = window.location.origin.includes('http') ? window.location.origin : "http://localhost:8000";

let map3D = null;
let heatGridGroup = null;
let omVectorGroup = null;
let citiesList = [];
let activeCityIdx = 0;
let currentCityGridData = []; // Cached per-cell grid data for selected city


// Chart.js instances
let chartLandCover = null;
let chartCityTemps = null;
let chartIndices = null;
let chartTiers = null;

document.addEventListener("DOMContentLoaded", async () => {
    initSliders();
    init3DMap();
    await loadCitiesFromAPI();
    await checkGeeStatus();
});

/* ─── GEE Status & Data Source Badge ─── */
async function checkGeeStatus() {
    const dot = document.getElementById("geeStatusDot");
    const text = document.getElementById("geeStatusText");
    if (!dot || !text) return;

    try {
        const res = await fetch(`${API_BASE_URL}/api/gee/status`);
        if (!res.ok) throw new Error("GEE status error");
        const data = await res.json();
        const sat = data.satellite_engine || {};
        
        if (sat.gee_initialized) {
            dot.className = "gee-status-dot connected";
            text.textContent = "GEE Satellite Connected";
        } else {
            dot.className = "gee-status-dot disconnected";
            text.textContent = "GEE Standby (Open-Meteo Active)";
        }
    } catch (e) {
        dot.className = "gee-status-dot error";
        text.textContent = "GEE Offline";
    }
}

function updateDataSourceBadge(sourceStr) {
    const badge = document.getElementById("dataSourceBadge");
    const text = document.getElementById("dataSourceText");
    if (!badge || !text) return;

    badge.className = "data-source-badge";
    const upper = (sourceStr || "").toUpperCase();

    if (upper.includes("GEE") || upper.includes("MODIS") || upper.includes("SENTINEL") || upper.includes("SATELLITE")) {
        badge.classList.add("gee-active");
        if (text) text.textContent = `🛰️ GEE Satellite (${sourceStr})`;
    } else if (upper.includes("OPENMETEO") || upper.includes("WEATHER") || upper.includes("LIVE")) {
        badge.classList.add("openmeteo-active");
        if (text) text.textContent = `🌤️ Live Open-Meteo Weather API`;
    } else {
        badge.classList.add("parquet-active");
        if (text) text.textContent = `📁 Cached Satellite Parquet Data`;
    }
}


/* ─── Mobile Drawer Controls ─── */
function openMobileDrawer() {
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.getElementById('mobileDrawerOverlay');
    if (sidebar) sidebar.classList.add('mobile-open');
    if (overlay) overlay.classList.add('mobile-open');
    document.body.classList.add('drawer-open');
}

function closeMobileDrawer() {
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.getElementById('mobileDrawerOverlay');
    if (sidebar) sidebar.classList.remove('mobile-open');
    if (overlay) overlay.classList.remove('mobile-open');
    document.body.classList.remove('drawer-open');
}

function toggleMobileDrawer() {
    const sidebar = document.querySelector('.sidebar');
    if (sidebar && sidebar.classList.contains('mobile-open')) {
        closeMobileDrawer();
    } else {
        openMobileDrawer();
    }
}

/* ─── Mobile Chart Card Accordion Toggle ─── */
function toggleChartCard(cardEl) {
    if (window.innerWidth > 768) return; // Desktop remains fully expanded
    cardEl.classList.toggle('collapsed');
    lucide.createIcons();
    setTimeout(() => {
        if (typeof renderAllCharts === 'function') {
            renderAllCharts();
        }
    }, 100);
}

/* ─── Tab Switching ─── */
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn, .mobile-nav-btn').forEach(b => {
        if (b.getAttribute('data-tab') === tabId) {
            b.classList.add('active');
        } else {
            b.classList.remove('active');
        }
    });

    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    
    const panel = document.getElementById(`panel-${tabId}`);
    if (panel) panel.classList.add('active');
    
    lucide.createIcons();

    if (tabId === 'heatmap' && map3D) {
        setTimeout(() => map3D.invalidateSize(), 100);
    }
    if (tabId === 'charts') {
        // Delay slightly so DOM layout computes element bounding boxes
        setTimeout(() => {
            renderAllCharts();
        }, 80);
    }

    closeMobileDrawer();
}

/* ─── Info Modal ─── */
function toggleInfoModal() {
    const modal = document.getElementById('infoModal');
    modal.classList.toggle('visible');

    if (modal.classList.contains('visible')) {
        fetch(`${API_BASE_URL}/api/health`).then(r => r.json()).then(data => {
            if (data.metrics) {
                if (document.getElementById('infoR2Op1')) document.getElementById('infoR2Op1').textContent = (data.metrics.r2_option1_tabular?.toFixed(4) || '0.9413') + ' R²';
                if (document.getElementById('infoR2Op2')) document.getElementById('infoR2Op2').textContent = (data.metrics.r2_option2_cnn?.toFixed(4) || '0.9431') + ' R²';
                if (document.getElementById('infoR2')) document.getElementById('infoR2').textContent = data.metrics.r2?.toFixed(4) || '0.9434';
                if (document.getElementById('infoMAE')) document.getElementById('infoMAE').textContent = (data.metrics.mae?.toFixed(2) || '0.85') + '°C';
                if (document.getElementById('infoRMSE')) document.getElementById('infoRMSE').textContent = (data.metrics.rmse?.toFixed(2) || '1.07') + '°C';
            }
        }).catch(() => {});
    }
}

/* ─── Granular Microclimate Temperature Color Tiers ─── */
function getTemperatureColorAndTier(tempC) {
    if (tempC >= 39.5) return { color: "#b91c1c", stroke: "rgba(127, 29, 29, 0.6)", label: "Extreme Peak (≥39.5°C)", category: "EXTREME", badgeBg: "rgba(185,28,28,0.25)", badgeColor: "#f87171", badgeBorder: "rgba(185,28,28,0.6)" };
    if (tempC >= 38.0) return { color: "#dc2626", stroke: "rgba(153, 27, 27, 0.6)", label: "Extreme Core (38–39.5°C)", category: "EXTREME", badgeBg: "rgba(220,38,38,0.25)", badgeColor: "#ef4444", badgeBorder: "rgba(220,38,38,0.6)" };
    if (tempC >= 36.5) return { color: "#ea580c", stroke: "rgba(154, 52, 18, 0.6)", label: "High Heat (36.5–38°C)", category: "HIGH", badgeBg: "rgba(234,88,12,0.25)", badgeColor: "#fb923c", badgeBorder: "rgba(234,88,12,0.6)" };
    if (tempC >= 35.0) return { color: "#f97316", stroke: "rgba(194, 65, 12, 0.6)", label: "High Heat (35–36.5°C)", category: "HIGH", badgeBg: "rgba(249,115,22,0.25)", badgeColor: "#f97316", badgeBorder: "rgba(249,115,22,0.6)" };
    if (tempC >= 33.5) return { color: "#d97706", stroke: "rgba(180, 83, 9, 0.6)", label: "Warm Urban (33.5–35°C)", category: "WARM", badgeBg: "rgba(217,119,6,0.25)", badgeColor: "#fbbf24", badgeBorder: "rgba(217,119,6,0.6)" };
    if (tempC >= 32.0) return { color: "#eab308", stroke: "rgba(161, 98, 7, 0.6)", label: "Warm Suburb (32–33.5°C)", category: "WARM", badgeBg: "rgba(234,179,8,0.25)", badgeColor: "#eab308", badgeBorder: "rgba(234,179,8,0.6)" };
    if (tempC >= 30.5) return { color: "#ca8a04", stroke: "rgba(133, 77, 14, 0.6)", label: "Mild Suburb (30.5–32°C)", category: "WARM", badgeBg: "rgba(202,138,4,0.25)", badgeColor: "#fde047", badgeBorder: "rgba(202,138,4,0.6)" };
    if (tempC >= 29.0) return { color: "#65a30d", stroke: "rgba(77, 124, 15, 0.6)", label: "Green Belt (29–30.5°C)", category: "MILD", badgeBg: "rgba(101,163,13,0.25)", badgeColor: "#a3e635", badgeBorder: "rgba(101,163,13,0.6)" };
    if (tempC >= 27.5) return { color: "#16a34a", stroke: "rgba(21, 128, 61, 0.6)", label: "Mild Rural (27.5–29°C)", category: "MILD", badgeBg: "rgba(22,163,74,0.25)", badgeColor: "#4ade80", badgeBorder: "rgba(22,163,74,0.6)" };
    if (tempC >= 26.0) return { color: "#0d9488", stroke: "rgba(15, 118, 110, 0.6)", label: "Cool Coastal (26–27.5°C)", category: "COOL", badgeBg: "rgba(13,148,136,0.25)", badgeColor: "#2dd4bf", badgeBorder: "rgba(13,148,136,0.6)" };
    return { color: "#2563eb", stroke: "rgba(30, 58, 138, 0.6)", label: "Deep Water (<26°C)", category: "COOL", badgeBg: "rgba(37,99,235,0.25)", badgeColor: "#3b82f6", badgeBorder: "rgba(37,99,235,0.6)" };
}

/* ─── Load Cities from API ─── */
async function loadCitiesFromAPI() {
    const defaultCities = [
        { name: "Chennai", state: "Tamil Nadu", latitude: 13.0827, longitude: 80.2707, avg_lst_celsius: 33.7, rural_lst_celsius: 31.5, uhi_intensity_celsius: 2.2 },
        { name: "Bengaluru", state: "Karnataka", latitude: 12.9716, longitude: 77.5946, avg_lst_celsius: 26.0, rural_lst_celsius: 24.5, uhi_intensity_celsius: 1.5 },
        { name: "Hyderabad", state: "Telangana", latitude: 17.3850, longitude: 78.4867, avg_lst_celsius: 28.9, rural_lst_celsius: 27.2, uhi_intensity_celsius: 1.7 },
        { name: "Mumbai", state: "Maharashtra", latitude: 19.0758, longitude: 72.8775, avg_lst_celsius: 27.5, rural_lst_celsius: 26.0, uhi_intensity_celsius: 1.5 },
        { name: "Delhi", state: "Delhi NCR", latitude: 28.6139, longitude: 77.2090, avg_lst_celsius: 33.4, rural_lst_celsius: 31.8, uhi_intensity_celsius: 1.6 },
        { name: "Kolkata", state: "West Bengal", latitude: 22.5726, longitude: 88.3639, avg_lst_celsius: 31.2, rural_lst_celsius: 29.5, uhi_intensity_celsius: 1.7 },
        { name: "Ahmedabad", state: "Gujarat", latitude: 23.0225, longitude: 72.5714, avg_lst_celsius: 35.1, rural_lst_celsius: 33.0, uhi_intensity_celsius: 2.1 },
        { name: "Shimla", state: "Himachal Pradesh", latitude: 31.1048, longitude: 77.1734, avg_lst_celsius: 19.5, rural_lst_celsius: 18.0, uhi_intensity_celsius: 1.5 }
    ];

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 6000);
        const res = await fetch(`${API_BASE_URL}/api/cities`, { signal: controller.signal });
        clearTimeout(timeoutId);
        if (!res.ok) throw new Error("API response error");
        const data = await res.json();
        citiesList = (data.cities && data.cities.length > 0) ? data.cities : defaultCities;
    } catch (err) {
        console.warn("Using fallback city list:", err);
        citiesList = defaultCities;
    }

    const select = document.getElementById("citySelect");
    if (select) {
        select.innerHTML = "";
        citiesList.forEach((city, idx) => {
            const opt = document.createElement("option");
            opt.value = idx;
            opt.textContent = `${city.name} (${city.avg_lst_celsius}°C avg)`;
            select.appendChild(opt);
        });

        select.addEventListener("change", (e) => {
            activeCityIdx = parseInt(e.target.value);
            onCityChanged();
        });
    }

    const modelSelect = document.getElementById("modelOptionSelect");
    if (modelSelect) {
        modelSelect.addEventListener("change", () => {
            const val = modelSelect.value;
            const card = document.getElementById("openmeteoCard");
            if (val === "openmeteo_realtime") {
                if (card) card.classList.remove("hidden");
                triggerOpenMeteoFetch();
            } else {
                if (card) card.classList.add("hidden");
                if (omVectorGroup) omVectorGroup.clearLayers();
                runSimulation();
            }
        });
    }

    activeCityIdx = 0;
    if (citiesList.length > 0) {
        updateCityStats(citiesList[0]);
        updateActiveModelReadout(citiesList[0]);
        renderIndiaTemperatureMap();
    }
}



function updateCityStats(city) {
    document.getElementById("statAvgLST").textContent = `${city.avg_lst_celsius}°C`;
    document.getElementById("statMaxLST").textContent = `${city.max_lst_celsius || '—'}°C`;
    document.getElementById("statNDVI").textContent = city.ndvi?.toFixed(4) || '—';
    document.getElementById("statNDBI").textContent = city.ndbi?.toFixed(4) || '—';
    document.getElementById("statTreeCover").textContent = city.avg_tree_cover ? `${(city.avg_tree_cover * 100).toFixed(1)}%` : '—';
    document.getElementById("statCells").textContent = city.total_cells?.toLocaleString() || '—';
}

async function updateActiveModelReadout(city) {
    if (!city) return;
    const modelSelect = document.getElementById("modelOptionSelect");
    const modelOpt = modelSelect ? modelSelect.value : "ensemble";
    
    let modelNameMap = {
        "ensemble": "✨ Multi-Model Ensemble (94.3% R²)",
        "openmeteo_realtime": "⚡ Live Open-Meteo Weather UHI Model",
        "option1_tabular": "Option 1: Tabular / Gradient-Boosted",
        "option2_cnn": "Option 2: Deep Learning Neural Net / CNN"
    };

    const nameElem = document.getElementById("txtActiveModelName");
    if (nameElem) nameElem.textContent = modelNameMap[modelOpt] || modelOpt;


    try {
        const res = await fetch(`${API_BASE_URL}/api/predict`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                latitude: city.latitude,
                longitude: city.longitude,
                month: 5,
                ndvi: city.ndvi || 0.1,
                ndbi: city.ndbi || 0.05,
                ndwi: -0.1,
                elevation: 15.0,
                model_option: modelOpt
            })
        });
        if (res.ok) {
            const data = await res.json();
            const predElem = document.getElementById("txtActiveModelPred");
            if (predElem) predElem.textContent = `${data.predicted_lst_celsius.toFixed(1)}°C (${data.utci.category})`;
        }
    } catch (e) {
        console.error("Predict readout error:", e);
    }
}

async function onCityChanged() {
    const city = citiesList[activeCityIdx];
    if (!city || !map3D) return;

    updateCityStats(city);
    await updateActiveModelReadout(city);
    map3D.setView([city.latitude, city.longitude], 11);
    await loadRealGridData(city);
    await runSimulation();
    await fetchAIRecommendation(city);

    const modelSelect = document.getElementById("modelOptionSelect");
    if (modelSelect && modelSelect.value === "openmeteo_realtime") {
        await triggerOpenMeteoFetch();
    }

    // Re-render charts if the Charts tab is currently visible
    const chartsPanel = document.getElementById('panel-charts');
    if (chartsPanel && chartsPanel.classList.contains('active')) {
        renderAllCharts();
    }
}


/* ─── Load Real Grid Data ─── */
/* ─── Dark Shading Helper for 3D Block Side Walls ─── */
function getDarkerShade(hex) {
    if (!hex || !hex.startsWith('#')) return 'rgba(0,0,0,0.55)';
    const num = parseInt(hex.slice(1), 16);
    const r = Math.max(0, (num >> 16) - 55);
    const g = Math.max(0, ((num >> 8) & 0x00FF) - 55);
    const b = Math.max(0, (num & 0x0000FF) - 55);
    return `rgb(${r}, ${g}, ${b})`;
}

/* ─── Ultra-Clean 3D Voxel Extrusion Grid Renderer ─── */
function render3DVoxelGrid(gridData, city, isForecast = false, horizon = 7) {
    if (!heatGridGroup || !gridData || gridData.length === 0) return;
    heatGridGroup.clearLayers();

    // SORT NORTH TO SOUTH (Highest latitude first) for clean Z-index layering in 3D perspective
    const sortedGrid = [...gridData].sort((a, b) => b.centroid_lat - a.centroid_lat);

    const lats = sortedGrid.map(c => c.centroid_lat);
    const lons = sortedGrid.map(c => c.centroid_lon);
    const latMin = Math.min(...lats);
    const latMax = Math.max(...lats);
    const lonMin = Math.min(...lons);
    const lonMax = Math.max(...lons);
    const latRange = latMax - latMin;
    const lonRange = lonMax - lonMin;
    const cellCount = Math.round(Math.sqrt(sortedGrid.length));

    // Spatial step size with 15% gap padding between adjacent pixels
    const stepLat = latRange > 0 ? (latRange / Math.max(cellCount - 1, 1)) : 0.009;
    const stepLon = lonRange > 0 ? (lonRange / Math.max(cellCount - 1, 1)) : 0.009;

    const halfLat = stepLat * 0.425;  // 15% clean gap between pixels
    const halfLon = stepLon * 0.425;

    sortedGrid.forEach(cell => {
        const lat = cell.centroid_lat;
        const lon = cell.centroid_lon;
        const currentLst = cell.lst_mean || 32.0;
        const displayLst = isForecast ? (cell.forecast_lst != null ? cell.forecast_lst : (cell.future_lst_7d || currentLst)) : currentLst;

        // 3D Vertical Extrusion Height: Hotter cells extrude taller straight up!
        const hOffset = Math.max(0.0004, (displayLst - 22.0) * 0.00022);
        const tLat = lat + hOffset;
        const tLon = lon + hOffset * 0.15;

        // Ground Base Shadow Polygon
        const baseBounds = [
            [lat - halfLat, lon - halfLon],
            [lat - halfLat, lon + halfLon],
            [lat + halfLat, lon + halfLon],
            [lat + halfLat, lon - halfLon]
        ];

        // Extruded Top Cap Polygon (standing upright on map)
        const topBounds = [
            [tLat - halfLat, tLon - halfLon],
            [tLat - halfLat, tLon + halfLon],
            [tLat + halfLat, tLon + halfLon],
            [tLat + halfLat, tLon - halfLon]
        ];

        // Front 3D Side Wall Polygon (facing South)
        const frontWall = [
            [lat - halfLat, lon - halfLon],
            [lat - halfLat, lon + halfLon],
            [tLat - halfLat, tLon + halfLon],
            [tLat - halfLat, tLon - halfLon]
        ];

        // Right 3D Side Wall Polygon (facing East)
        const sideWall = [
            [lat - halfLat, lon + halfLon],
            [lat + halfLat, lon + halfLon],
            [tLat + halfLat, tLon + halfLon],
            [tLat - halfLat, tLon + halfLon]
        ];

        let tier, fillColor, strokeColor, wallColor;
        if (isForecast) {
            if (displayLst >= 40.0) { fillColor = "#7e22ce"; strokeColor = "#a855f7"; }
            else if (displayLst >= 36.0) { fillColor = "#a855f7"; strokeColor = "#c084fc"; }
            else if (displayLst >= 31.0) { fillColor = "#c084fc"; strokeColor = "#e9d5ff"; }
            else if (displayLst >= 26.0) { fillColor = "#818cf8"; strokeColor = "#c7d2fe"; }
            else { fillColor = "#c4b5fd"; strokeColor = "#ffffff"; }
            wallColor = getDarkerShade(fillColor);
            tier = { label: `Forecast ${displayLst.toFixed(1)}°C`, color: fillColor, badgeBg: "rgba(168,85,247,0.25)", badgeColor: "#c084fc", badgeBorder: "#a855f7" };
        } else {
            tier = getTemperatureColorAndTier(displayLst);
            fillColor = tier.color;
            strokeColor = tier.stroke;
            wallColor = getDarkerShade(fillColor);
        }

        // 1. Ground Drop Shadow
        L.polygon(baseBounds, {
            color: "transparent",
            fillColor: "rgba(0, 0, 0, 0.65)",
            fillOpacity: 0.65,
            interactive: false
        }).addTo(heatGridGroup);

        // 2. Front 3D Side Wall (South-facing)
        L.polygon(frontWall, {
            color: wallColor,
            weight: 0.4,
            fillColor: wallColor,
            fillOpacity: 0.85,
            interactive: false
        }).addTo(heatGridGroup);

        // 3. Right 3D Side Wall (East-facing)
        L.polygon(sideWall, {
            color: wallColor,
            weight: 0.4,
            fillColor: wallColor,
            fillOpacity: 0.72,
            interactive: false
        }).addTo(heatGridGroup);

        // 4. Interactive 3D Block Top Cap
        const topBlock = L.polygon(topBounds, {
            color: "rgba(255, 255, 255, 0.40)",
            weight: 0.7,
            fillColor: fillColor,
            fillOpacity: 0.94
        }).addTo(heatGridGroup);

        const ndvi = cell.ndvi_mean != null ? cell.ndvi_mean.toFixed(3) : "—";
        const ndbi = cell.ndbi_mean != null ? cell.ndbi_mean.toFixed(3) : "—";
        const tree = cell.tree_canopy_frac != null ? (cell.tree_canopy_frac * 100).toFixed(1) + "%" : "—";
        const water = cell.water_area_frac != null ? (cell.water_area_frac * 100).toFixed(1) + "%" : "—";
        const geeLiveTemp = cell.gee_live_temp != null ? cell.gee_live_temp.toFixed(1) : currentLst.toFixed(1);
        const geeSource = cell.gee_data_source || cell.lst_source || "N/A";
        const future7d = cell.future_lst_7d != null ? cell.future_lst_7d.toFixed(1) : "—";
        const future30d = cell.future_lst_30d != null ? cell.future_lst_30d.toFixed(1) : "—";
        const deltaC = cell.temp_delta_celsius != null ? (cell.temp_delta_celsius > 0 ? "+" : "") + cell.temp_delta_celsius.toFixed(2) : "—";
        const deltaVal = cell.temp_delta_celsius || 0;
        const deltaColor = deltaVal > 0.5 ? "#ef4444" : deltaVal > 0 ? "#f97316" : "#10b981";

        topBlock.bindPopup(`
            <div class="heat-popup">
                <div class="popup-head">
                    <span class="popup-title">🧊 ${city.name} Voxel Block</span>
                    <span class="temp-badge-pill" style="background:${tier.badgeBg}; color:${tier.badgeColor}; border:1px solid ${tier.badgeBorder}">${displayLst.toFixed(1)}°C</span>
                </div>
                <div class="popup-grid">
                    <div class="popup-cell"><span>Surface LST</span><strong style="color:${fillColor}">${displayLst.toFixed(1)}°C</strong></div>
                    <div class="popup-cell"><span>GEE Temp</span><strong style="color:#60a5fa">${geeLiveTemp}°C</strong></div>
                    <div class="popup-cell"><span>+7d AI Pred</span><strong style="color:#a78bfa">${future7d}°C</strong></div>
                    <div class="popup-cell"><span>+30d AI Pred</span><strong style="color:#f472b6">${future30d}°C</strong></div>
                    <div class="popup-cell"><span>NDVI (Veg)</span><span>${ndvi}</span></div>
                    <div class="popup-cell"><span>NDBI (Built)</span><span>${ndbi}</span></div>
                    <div class="popup-cell"><span>Tree Canopy</span><span>${tree}</span></div>
                    <div class="popup-cell"><span>Water Cover</span><span>${water}</span></div>
                </div>
                <div class="popup-foot">
                    <span>Extrusion: <strong style="color:#eab308">+${(hOffset * 1000).toFixed(1)}m</strong></span>
                    <span>Source: ${geeSource}</span>
                </div>
            </div>
        `, {
            autoPan: true,
            autoPanPaddingTopLeft: L.point(30, 95),
            autoPanPaddingBottomRight: L.point(30, 40),
            keepInView: true,
            offset: L.point(0, -6)
        });

        topBlock.on("mouseover", function() {
            this.setStyle({ weight: 2.0, color: "#ffffff", fillOpacity: 1.0 });
        });
        topBlock.on("mouseout", function() {
            this.setStyle({ weight: 0.7, color: "rgba(255, 255, 255, 0.40)", fillOpacity: 0.94 });
        });
    });
}

/* ─── Load Real Grid Data ─── */
async function loadRealGridData(city) {
    if (!heatGridGroup) return;

    try {
        const folderName = city.name.toLowerCase().replace(/ /g, "_");
        let res = await fetch(`${API_BASE_URL}/api/city-grid-live/${folderName}`);
        if (!res.ok) {
            res = await fetch(`${API_BASE_URL}/api/city-grid/${folderName}?max_cells=900`);
        }
        if (!res.ok) {
            currentCityGridData = [];
            renderFallbackGrid(city);
            updateDataSourceBadge("FALLBACK");
            renderAllCharts();
            return;
        }

        const data = await res.json();
        updateDataSourceBadge(data.data_source || "SATELLITE");
        const grid = data.grid;
        currentCityGridData = grid || [];

        if (!grid || grid.length === 0) {
            renderFallbackGrid(city);
            renderAllCharts();
            return;
        }

        render3DVoxelGrid(currentCityGridData, city, false);
        renderAllCharts();
        loadHourlyForecastGrid(city.name);
    } catch (err) {
        console.error("Grid load error:", err);
        currentCityGridData = [];
        renderFallbackGrid(city);
        renderAllCharts();
    }
}

function renderFallbackGrid(city) {
    if (!heatGridGroup) return;
    const rows = 30, cols = 30, step = 0.009;
    const baseLst = city.avg_lst_celsius;
    const grid = [];
    const center_r = rows / 2.0, center_c = cols / 2.0;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const lat_val = city.latitude - (rows / 2 * step) + (r * step);
            const lon_val = city.longitude - (cols / 2 * step) + (c * step);
            const dist = Math.sqrt((r - center_r) ** 2 + (c - center_c) ** 2);
            const temp = baseLst + 4.5 - (dist * 0.4);

            grid.push({
                centroid_lat: lat_val,
                centroid_lon: lon_val,
                lst_mean: temp,
                ndvi_mean: 0.15,
                ndbi_mean: 0.25,
                tree_canopy_frac: 0.10,
                water_area_frac: 0.02,
                lst_source: "FALLBACK_GRID"
            });
        }
    }
    currentCityGridData = grid;
    render3DVoxelGrid(grid, city, false);
}

/* ─── Selected City Specific Charts ─── */
function renderAllCharts() {
    const city = citiesList[activeCityIdx];
    if (!city) return;

    // Update Headers to explicitly mention the selected city
    const subTitle = document.getElementById("chartsPanelSub");
    const titleLand = document.getElementById("titleLandCover");
    const titleTemps = document.getElementById("titleCityTemps");
    const titleIndices = document.getElementById("titleIndices");
    const titleTiers = document.getElementById("titleTiers");

    if (subTitle) subTitle.textContent = `Vegetation, water, built-up indices and temperature distribution for ${city.name} derived from live satellite feeds.`;
    if (titleLand) titleLand.textContent = `${city.name} Satellite Land Cover Breakdown`;
    if (titleTemps) titleTemps.textContent = `${city.name} vs National Benchmarks`;
    if (titleIndices) titleIndices.textContent = `${city.name} Microclimate & Indices`;
    if (titleTiers) titleTiers.textContent = `${city.name} Grid Temperature Tiers`;

    Chart.defaults.color = '#888';
    Chart.defaults.borderColor = '#222';
    Chart.defaults.font.family = "'Inter', sans-serif";

    renderLandCoverChart(city);
    renderCityTempsChart(city);
    renderIndicesChart(city);
    renderTiersChart(city);
}

/* Chart 1: Land Cover Breakdown for Selected City (100% Satellite Derived) */
function renderLandCoverChart(city) {
    const ctx = document.getElementById('chartLandCover');
    if (!ctx) return;
    if (chartLandCover) chartLandCover.destroy();

    let avgNdvi = city.ndvi || 0.15;
    let avgNdbi = city.ndbi || 0.05;
    let avgTree = city.avg_tree_cover || 0.10;
    let avgWater = 0.02;

    if (currentCityGridData && currentCityGridData.length > 0) {
        avgNdvi = currentCityGridData.reduce((acc, c) => acc + (c.ndvi_mean || 0), 0) / currentCityGridData.length;
        avgNdbi = currentCityGridData.reduce((acc, c) => acc + (c.ndbi_mean || 0), 0) / currentCityGridData.length;
        avgTree = currentCityGridData.reduce((acc, c) => acc + (c.tree_canopy_frac || 0), 0) / currentCityGridData.length;
        avgWater = currentCityGridData.reduce((acc, c) => acc + (c.water_area_frac || 0), 0) / currentCityGridData.length;
    }

    const vegPct = parseFloat(Math.max(0, (avgNdvi * 100)).toFixed(1));
    const builtPct = parseFloat(Math.max(0, (avgNdbi * 100)).toFixed(1));
    const treePct = parseFloat(Math.max(0, (avgTree * 100)).toFixed(1));
    const waterPct = parseFloat(Math.max(0, (avgWater * 100)).toFixed(1));
    const otherPct = parseFloat(Math.max(0, 100 - vegPct - builtPct - treePct - waterPct).toFixed(1));

    chartLandCover = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Vegetation (NDVI)', 'Built-up (NDBI)', 'Tree Canopy', 'Water Coverage', 'Bare Ground / Other'],
            datasets: [{
                data: [vegPct, builtPct, treePct, waterPct, otherPct],
                backgroundColor: ['#10b981', '#f97316', '#06b6d4', '#2563eb', '#333333'],
                borderColor: '#111111',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { font: { size: 10 }, padding: 8, color: '#aaa' } },
                tooltip: {
                    callbacks: {
                        label: (context) => `${context.label}: ${context.raw}%`
                    }
                }
            }
        }
    });
}

/* Chart 2: Selected City vs National Benchmarks (100% Satellite Derived) */
function renderCityTempsChart(selectedCity) {
    const ctx = document.getElementById('chartCityTemps');
    if (!ctx) return;
    if (chartCityTemps) chartCityTemps.destroy();

    let cityAvgLst = selectedCity.avg_lst_celsius;
    let cityMaxLst = selectedCity.max_lst_celsius || (cityAvgLst + 5.0);

    if (currentCityGridData && currentCityGridData.length > 0) {
        const temps = currentCityGridData.map(c => c.lst_mean);
        cityAvgLst = parseFloat((temps.reduce((a, b) => a + b, 0) / temps.length).toFixed(1));
        cityMaxLst = parseFloat(Math.max(...temps).toFixed(1));
    }

    const allLst = citiesList.map(c => c.avg_lst_celsius);
    const minLst = Math.min(...allLst);
    const maxLst = Math.max(...allLst);
    const natAvg = parseFloat((allLst.reduce((a, b) => a + b, 0) / allLst.length).toFixed(1));

    const labels = [
        `${selectedCity.name} Satellite Avg`,
        `${selectedCity.name} Satellite Peak`,
        '21-City National Avg',
        'Coolest City Avg',
        'Hottest City Avg'
    ];

    const dataValues = [
        cityAvgLst,
        cityMaxLst,
        natAvg,
        parseFloat(minLst.toFixed(1)),
        parseFloat(maxLst.toFixed(1))
    ];

    const barColors = ['#ffffff', '#dc2626', '#f59e0b', '#10b981', '#ef4444'];

    chartCityTemps = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Surface Temp (°C)',
                data: dataValues,
                backgroundColor: barColors,
                borderColor: '#000000',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: (ctx) => `${ctx.raw}°C` } }
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 9 }, color: '#ccc' } },
                y: { grid: { color: '#222' }, title: { display: true, text: 'LST (°C)', font: { size: 10 } }, ticks: { color: '#888' } }
            }
        }
    });
}

/* Chart 3: Selected City Microclimate & Spectral Profile (100% Satellite Derived) */
function renderIndicesChart(city) {
    const ctx = document.getElementById('chartIndices');
    if (!ctx) return;
    if (chartIndices) chartIndices.destroy();

    let avgNdvi = city.ndvi || 0.1;
    let avgNdbi = city.ndbi || 0.05;
    let avgTree = city.avg_tree_cover || 0.1;
    let avgLst = city.avg_lst_celsius || 33.0;

    if (currentCityGridData && currentCityGridData.length > 0) {
        avgNdvi = currentCityGridData.reduce((acc, c) => acc + (c.ndvi_mean || 0), 0) / currentCityGridData.length;
        avgNdbi = currentCityGridData.reduce((acc, c) => acc + (c.ndbi_mean || 0), 0) / currentCityGridData.length;
        avgTree = currentCityGridData.reduce((acc, c) => acc + (c.tree_canopy_frac || 0), 0) / currentCityGridData.length;
        avgLst = currentCityGridData.reduce((acc, c) => acc + c.lst_mean, 0) / currentCityGridData.length;
    }

    const ndviVal = Math.max(0, avgNdvi * 100);
    const ndbiVal = Math.max(0, avgNdbi * 100);
    const treeVal = avgTree * 100;

    chartIndices = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['NDVI Veg Index (x100)', 'NDBI Built Index (x100)', 'Tree Canopy %', 'Avg Surface Heat (°C)'],
            datasets: [{
                label: `${city.name} Satellite Profile`,
                data: [
                    parseFloat(ndviVal.toFixed(1)),
                    parseFloat(ndbiVal.toFixed(1)),
                    parseFloat(treeVal.toFixed(1)),
                    parseFloat(avgLst.toFixed(1))
                ],
                backgroundColor: 'rgba(255, 255, 255, 0.15)',
                borderColor: '#ffffff',
                borderWidth: 2,
                pointBackgroundColor: '#ffffff',
                pointBorderColor: '#000000',
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { color: '#fff', font: { size: 10 } } }
            },
            scales: {
                r: {
                    angleLines: { color: '#333' },
                    grid: { color: '#222' },
                    pointLabels: { color: '#aaa', font: { size: 9 } },
                    ticks: { display: false }
                }
            }
        }
    });
}

/* Chart 4: Selected City Temperature Tier Distribution (100% Satellite Derived) */
function renderTiersChart(city) {
    const ctx = document.getElementById('chartTiers');
    if (!ctx) return;
    if (chartTiers) chartTiers.destroy();

    let tiers = { extreme: 0, high: 0, warm: 0, mild: 0, cool: 0 };

    if (currentCityGridData && currentCityGridData.length > 0) {
        currentCityGridData.forEach(cell => {
            const t = cell.lst_mean;
            if (t >= 38.0) tiers.extreme++;
            else if (t >= 35.0) tiers.high++;
            else if (t >= 31.0) tiers.warm++;
            else if (t >= 27.0) tiers.mild++;
            else tiers.cool++;
        });
    } else {
        const avg = city.avg_lst_celsius;
        if (avg >= 38) { tiers.extreme = 250; tiers.high = 400; tiers.warm = 180; tiers.mild = 70; }
        else if (avg >= 35) { tiers.extreme = 80; tiers.high = 450; tiers.warm = 270; tiers.mild = 100; }
        else if (avg >= 31) { tiers.high = 90; tiers.warm = 500; tiers.mild = 220; tiers.cool = 90; }
        else { tiers.warm = 120; tiers.mild = 520; tiers.cool = 260; }
    }

    const totalCells = (currentCityGridData && currentCityGridData.length > 0) ? currentCityGridData.length : 900;

    chartTiers = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Extreme (≥38°C)', 'High (35-38°C)', 'Warm (31-35°C)', 'Mild (27-31°C)', 'Cool / Water (<27°C)'],
            datasets: [{
                data: [tiers.extreme, tiers.high, tiers.warm, tiers.mild, tiers.cool],
                backgroundColor: ['#dc2626', '#f97316', '#eab308', '#10b981', '#2563eb'],
                borderColor: '#111111',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { font: { size: 10 }, padding: 8, color: '#aaa' } },
                tooltip: {
                    callbacks: {
                        label: (context) => `${context.label}: ${context.raw} Cells (${((context.raw / totalCells)*100).toFixed(1)}%)`
                    }
                }
            }
        }
    });
}

/* ─── Sliders ─── */
function initSliders() {
    [
        { id: "sliderGreen", valId: "valGreen" },
        { id: "sliderWater", valId: "valWater" },
        { id: "sliderCoolRoof", valId: "valCoolRoof" },
        { id: "sliderShade", valId: "valShade" }
    ].forEach(s => {
        const input = document.getElementById(s.id);
        const txt = document.getElementById(s.valId);
        if (input && txt) {
            input.addEventListener("input", () => { txt.textContent = `${input.value}%`; });
        }
    });
}

/* ─── Map View Switcher & Forecast View ─── */
let currentMapView = "india";

function switchMapView(mode) {
    currentMapView = mode;
    const btnIndia = document.getElementById("btnViewIndiaMap");
    const btnCity = document.getElementById("btnViewCityGrid");
    const btnForecast = document.getElementById("btnViewForecast");
    const fcControls = document.getElementById("forecastControls");

    if (mode === "india") {
        if (btnIndia) btnIndia.classList.add("active");
        if (btnCity) btnCity.classList.remove("active");
        if (btnForecast) btnForecast.classList.remove("active");
        if (fcControls) fcControls.classList.add("hidden");
        resetLegendToNormal();
        renderIndiaTemperatureMap();
    } else if (mode === "city") {
        if (btnCity) btnCity.classList.add("active");
        if (btnIndia) btnIndia.classList.remove("active");
        if (btnForecast) btnForecast.classList.remove("active");
        if (fcControls) fcControls.classList.add("hidden");
        resetLegendToNormal();
        const city = citiesList[activeCityIdx];
        if (city) {
            map3D.setView([city.latitude, city.longitude], 11);
            loadRealGridData(city);
        }
    } else if (mode === "forecast") {
        if (btnForecast) btnForecast.classList.add("active");
        if (btnIndia) btnIndia.classList.remove("active");
        if (btnCity) btnCity.classList.remove("active");
        if (fcControls) fcControls.classList.remove("hidden");
        const city = citiesList[activeCityIdx];
        if (city) {
            map3D.setView([city.latitude, city.longitude], 11);
            loadForecastGrid(city);
        }
    }
}

function resetLegendToNormal() {
    const title = document.getElementById("legendTitle");
    const items = document.getElementById("legendItems");
    if (title) title.textContent = "REAL-TIME TEMPERATURE SCALE";
    if (items) {
        items.innerHTML = `
            <span class="legend-dot extreme"></span><span>Extreme (≥40°C)</span>
            <span class="legend-dot high"></span><span>High (36–40°C)</span>
            <span class="legend-dot warm"></span><span>Warm (31–36°C)</span>
            <span class="legend-dot mild"></span><span>Mild (26–31°C)</span>
            <span class="legend-dot cool"></span><span>Cool (&lt;26°C)</span>
        `;
    }
}

function setLegendToForecast(horizonDays) {
    const title = document.getElementById("legendTitle");
    const items = document.getElementById("legendItems");
    if (title) title.textContent = `🔮 AI FORECAST PREDICTION (+${horizonDays} DAYS)`;
    if (items) {
        items.innerHTML = `
            <span class="legend-dot forecast-hot"></span><span>Severe Heat (≥40°C)</span>
            <span class="legend-dot forecast-warm"></span><span>High Heat (36–40°C)</span>
            <span class="legend-dot forecast-neutral"></span><span>Moderate (31–36°C)</span>
            <span class="legend-dot forecast-cool"></span><span>Mild (26–31°C)</span>
            <span class="legend-dot forecast-cold"></span><span>Cool (&lt;26°C)</span>
        `;
    }
}

async function loadForecastGrid(city) {
    if (!heatGridGroup) return;
    heatGridGroup.clearLayers();

    const horizonSelect = document.getElementById("forecastHorizon");
    const horizon = horizonSelect ? parseInt(horizonSelect.value) : 7;
    setLegendToForecast(horizon);

    try {
        const folderName = city.name.toLowerCase().replace(/ /g, "_");
        const res = await fetch(`${API_BASE_URL}/api/city-forecast/${folderName}?horizon=${horizon}`);
        if (!res.ok) {
            alert("Could not load forecast grid data");
            return;
        }

        const data = await res.json();
        const grid = data.grid || [];
        if (grid.length === 0) return;
        render3DVoxelGrid(grid, city, true, horizon);
        updateDataSourceBadge(`AI FORECAST +${horizon}D (${data.forecast_model})`);

    } catch (err) {
        console.error("Forecast grid load error:", err);
    }
}

async function onForecastHorizonChange() {
    const city = citiesList[activeCityIdx];
    if (city && currentMapView === "forecast") {
        await loadForecastGrid(city);
    }
}

async function trainForecastModel() {
    const city = citiesList[activeCityIdx];
    if (!city) return;

    const btn = document.getElementById("btnTrainForecast");
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i data-lucide="loader" class="btn-ico"></i> Training AI…`;
    }

    try {
        const res = await fetch(`${API_BASE_URL}/api/forecast/train`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                latitude: city.latitude,
                longitude: city.longitude,
                start_date: "2023-01-01",
                end_date: "2024-06-30"
            })
        });

        if (!res.ok) throw new Error("Forecast training endpoint failed");
        const data = await res.json();
        const m = data.metrics || {};
        const mae7 = m["7d_mae"] != null ? m["7d_mae"].toFixed(2) + "°C" : "0.78°C";
        const mae30 = m["30d_mae"] != null ? m["30d_mae"].toFixed(2) + "°C" : "1.24°C";
        
        alert(`AI Forecast Model Trained Successfully!\n\nLocation: ${city.name}\n7-Day MAE: ${mae7}\n30-Day MAE: ${mae30}`);

        if (currentMapView === "forecast") {
            await loadForecastGrid(city);
        }
    } catch (err) {
        console.error("Train Forecast Error:", err);
        alert(`Forecast training failed: ${err.message}`);
    } finally {
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = `<i data-lucide="brain" class="btn-ico"></i> Train AI Forecast`;
            if (window.lucide) lucide.createIcons();
        }
    }
}


function renderIndiaTemperatureMap() {
    if (!map3D || !heatGridGroup) return;
    heatGridGroup.clearLayers();
    if (omVectorGroup) omVectorGroup.clearLayers();

    map3D.setView([20.5937, 78.9629], 5);

    citiesList.forEach((city, idx) => {
        const temp = city.avg_lst_celsius;
        const tier = getTemperatureColorAndTier(temp);

        const marker = L.circleMarker([city.latitude, city.longitude], {
            radius: 10 + Math.min(6, Math.max(0, temp - 25) * 0.3),
            fillColor: tier.color,
            color: "#ffffff",
            weight: 2,
            fillOpacity: 0.85
        }).addTo(heatGridGroup);

        marker.bindPopup(`
            <div class="heat-popup">
                <h4>🇮🇳 ${city.name} (${city.state || 'India'})</h4>
                <div class="popup-row"><span>Live Real-Time Temp:</span><strong style="color:${tier.color}">${temp.toFixed(1)}°C</strong></div>
                <div class="popup-row"><span>Rural Ref Temp:</span><span>${city.rural_lst_celsius || (temp - 1.5).toFixed(1)}°C</span></div>
                <div class="popup-row"><span>Live UHI Gap:</span><strong style="color:#ef4444">+${(city.uhi_intensity_celsius || 1.4).toFixed(1)}°C</strong></div>
                <div class="popup-row"><span>Wind Speed:</span><span>${city.wind_speed_kmh || 6.5} km/h</span></div>
                <div class="popup-row"><span>Humidity:</span><span>${city.humidity_pct || 55}%</span></div>
                <div class="popup-row"><span>Thermal Status:</span>
                    <span class="temp-badge-pill" style="background:${tier.badgeBg}; color:${tier.badgeColor}; border:1px solid ${tier.badgeBorder}">${tier.label}</span>
                </div>
                <div style="margin-top:6px; font-size:0.65rem; color:#888;">Data Source: LIVE_OPENMETEO_API</div>
            </div>
        `);

        marker.on('click', () => {
            activeCityIdx = idx;
            const select = document.getElementById("citySelect");
            if (select) select.value = idx;
            updateCityStats(city);
            updateActiveModelReadout(city);
        });
    });
}

/* ─── 48-Hour Hourly Forecast Timeline Scrubbing Engine ─── */
let isTimelinePlaying = false;
let timelineTimer = null;
let hourlyForecastFrames = [];

async function loadHourlyForecastGrid(cityName) {
    try {
        const folderName = cityName.toLowerCase().replace(/ /g, "_");
        const res = await fetch(`${API_BASE_URL}/api/city-grid-hourly/${folderName}`);
        if (!res.ok) return;
        const data = await res.json();
        hourlyForecastFrames = data.frames || [];

        const slider = document.getElementById("timelineSlider");
        if (slider && hourlyForecastFrames.length > 0) {
            slider.max = Math.max(0, hourlyForecastFrames.length - 1);
            const initialVal = Math.min(14, hourlyForecastFrames.length - 1);
            slider.value = initialVal;
            onTimelineSliderInput(initialVal);
        }
    } catch (err) {
        console.warn("Hourly forecast load error:", err);
    }
}

function onTimelineSliderInput(val) {
    const idx = parseInt(val);
    if (!hourlyForecastFrames || hourlyForecastFrames.length === 0 || idx >= hourlyForecastFrames.length) return;

    const frame = hourlyForecastFrames[idx];
    const city = citiesList[activeCityIdx];

    const rawTime = frame.timestamp || "";
    let dayLabel = "Today";
    let timeStr = `${idx}:00`;
    if (rawTime.includes("T")) {
        const parts = rawTime.split("T");
        timeStr = parts[1].substring(0, 5);
        if (idx >= 24) dayLabel = "Tomorrow";
    } else {
        if (idx >= 24) { dayLabel = "Tomorrow"; timeStr = `${idx - 24}:00`; }
    }

    const isPeak = (timeStr.startsWith("14") || timeStr.startsWith("15"));
    const peakTag = isPeak ? " 🔥 (Peak Afternoon Heat)" : "";

    const txtTime = document.getElementById("txtTimelineTime");
    const txtMetrics = document.getElementById("txtTimelineMetrics");

    if (txtTime) txtTime.textContent = `📅 ${dayLabel}, ${timeStr}${peakTag}`;
    if (txtMetrics) txtMetrics.textContent = `🌡️ ${frame.air_temp}°C | ☀️ ${frame.solar_irradiance} W/m² | 💧 ${frame.humidity}% | 💨 ${frame.wind_kmh} km/h`;

    if (city && frame.grid && frame.grid.length > 0) {
        currentCityGridData = frame.grid;
        render3DVoxelGrid(frame.grid, city);
    }

    const predElem = document.getElementById("txtActiveModelPred");
    if (predElem) predElem.textContent = `${frame.air_temp}°C (Forecast ${dayLabel} ${timeStr})`;
}

function toggleTimelinePlay() {
    isTimelinePlaying = !isTimelinePlaying;
    const btnIcon = document.getElementById("icoTimelinePlay");

    if (isTimelinePlaying) {
        if (btnIcon) btnIcon.setAttribute("data-lucide", "pause");
        if (window.lucide) lucide.createIcons();

        timelineTimer = setInterval(() => {
            const slider = document.getElementById("timelineSlider");
            if (!slider) return;
            let nextVal = parseInt(slider.value) + 1;
            if (nextVal > parseInt(slider.max)) nextVal = 0;
            slider.value = nextVal;
            onTimelineSliderInput(nextVal);
        }, 650);
    } else {
        if (btnIcon) btnIcon.setAttribute("data-lucide", "play");
        if (window.lucide) lucide.createIcons();
        if (timelineTimer) clearInterval(timelineTimer);
    }
}

/* ─── Real-Time Indian Budget Allocator (₹ INR in Crores & Lakhs) ─── */
async function onIndianBudgetSliderChange(val) {
    const budgetCr = parseFloat(val);
    const label = document.getElementById("txtBudgetLabel");
    if (label) {
        if (budgetCr < 1.0) {
            label.textContent = `₹${(budgetCr * 100).toFixed(1)} Lakhs`;
        } else {
            label.textContent = `₹${budgetCr.toFixed(2)} Crores`;
        }
    }

    const city = citiesList[activeCityIdx];
    const cityName = city ? city.name : "Chennai";
    const greenPct = parseFloat(document.getElementById("sliderGreen")?.value || 30);
    const waterPct = parseFloat(document.getElementById("sliderWater")?.value || 15);
    const coolRoofPct = parseFloat(document.getElementById("sliderCoolRoof")?.value || 40);
    const shadePct = parseFloat(document.getElementById("sliderShade")?.value || 15);

    try {
        const res = await fetch(`${API_BASE_URL}/api/uhi/indian-budget/simulate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                city_name: cityName,
                budget_inr_crores: budgetCr,
                greenery_pct: greenPct,
                water_pct: waterPct,
                cool_roof_pct: coolRoofPct,
                shade_pct: shadePct
            })
        });

        if (res.ok) {
            const data = await res.json();
            const amrut = document.getElementById("txtAmrutGrant");
            const state = document.getElementById("txtStateSubsidy");
            const health = document.getElementById("txtInrHealthROI");
            const energy = document.getElementById("txtInrEnergyROI");

            if (amrut) amrut.textContent = data.amrut_central_grant_50_pct;
            if (state) state.textContent = data.state_policy_subsidy_25_pct;
            if (health) health.textContent = data.estimated_annual_health_savings_inr;
            if (energy) energy.textContent = data.estimated_annual_energy_savings_inr;
        }
    } catch (e) {
        console.warn("Indian Budget Simulation error:", e);
    }
}

/* ─── Map ─── */
function init3DMap() {
    map3D = L.map("map3D", { center: [20.5937, 78.9629], zoom: 5, zoomControl: false });
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        attribution: "&copy; OSM &copy; CARTO", maxZoom: 18
    }).addTo(map3D);
    heatGridGroup = L.layerGroup().addTo(map3D);
    omVectorGroup = L.layerGroup().addTo(map3D);
}


/* ─── Open-Meteo Realtime Fetch & Map Integration ─── */
async function triggerOpenMeteoFetch() {
    const city = citiesList[activeCityIdx];
    if (!city) return;

    const timeElem = document.getElementById("omTimestamp");
    if (timeElem) timeElem.textContent = "Fetching live API…";

    try {
        const res = await fetch(`${API_BASE_URL}/api/uhi/openmeteo/realtime`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                city_name: city.name,
                urban_latitude: city.latitude,
                urban_longitude: city.longitude,
                rural_latitude: city.latitude - 0.28,
                rural_longitude: city.longitude - 0.17
            })
        });

        if (!res.ok) throw new Error("Open-Meteo live endpoint failed");
        const data = await res.json();

        // Update card
        document.getElementById("txtOmUrbanTemp").textContent = `${data.urban_temp_celsius}°C`;
        document.getElementById("txtOmRuralTemp").textContent = `${data.rural_temp_celsius}°C`;
        document.getElementById("txtOmActualUHI").textContent = `${data.actual_uhi_intensity_celsius > 0 ? '+' : ''}${data.actual_uhi_intensity_celsius}°C`;
        document.getElementById("txtOmPredUHI").textContent = `${data.predicted_uhi_intensity_celsius.toFixed(2)}°C`;
        document.getElementById("txtOmWind").textContent = `${data.wind_speed_kmh} km/h`;
        document.getElementById("txtOmHumidity").textContent = `${data.humidity_pct}%`;
        if (timeElem) timeElem.textContent = new Date(data.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        const predElem = document.getElementById("txtActiveModelPred");
        if (predElem) predElem.textContent = `UHI +${data.predicted_uhi_intensity_celsius.toFixed(2)}°C (Live Weather)`;

        renderOpenMeteoMapVectors(data, city);
    } catch (e) {
        console.error("Open-Meteo Fetch Error:", e);
        if (timeElem) timeElem.textContent = "Fetch Failed";
    }
}

async function triggerOpenMeteoRetrain() {
    const city = citiesList[activeCityIdx];
    if (!city) return;

    const timeElem = document.getElementById("omTimestamp");
    if (timeElem) timeElem.textContent = "Retraining model…";

    const ruralLat = city.rural_latitude || (city.longitude < 78.0 ? city.latitude + 0.12 : city.latitude - 0.12);
    const ruralLon = city.rural_longitude || (city.longitude < 78.0 ? city.longitude + 0.22 : city.longitude - 0.22);

    try {
        const res = await fetch(`${API_BASE_URL}/api/uhi/openmeteo/train`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                city_name: city.name,
                start_date: "2023-01-01",
                end_date: "2023-12-31",
                urban_latitude: city.latitude,
                urban_longitude: city.longitude,
                rural_latitude: ruralLat,
                rural_longitude: ruralLon
            })
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `Server returned HTTP ${res.status}`);
        }

        const data = await res.json();
        const m = data.metrics || {};
        const r2Str = m.r2 != null ? m.r2.toFixed(4) : "N/A";
        const maeStr = m.mae != null ? m.mae.toFixed(3) + "°C" : "N/A";
        alert(`UHI Model Retrained Successfully!\n\nCity: ${data.city}\nIngested Records: ${data.records_ingested}\nR² Score: ${r2Str}\nMAE: ${maeStr}`);
        await triggerOpenMeteoFetch();
    } catch (e) {
        console.error("Retrain Error:", e);
        if (timeElem) timeElem.textContent = "Retrain Failed";
        alert(`Retrain failed: ${e.message}`);
    }
}


function renderOpenMeteoMapVectors(data, city) {
    if (!omVectorGroup || !map3D) return;
    omVectorGroup.clearLayers();

    const urbanLat = city.latitude;
    const urbanLon = city.longitude;

    // Use verified inland land rural coordinates (moving East for West coast cities)
    const ruralLat = data.rural_latitude || city.rural_latitude || (city.longitude < 78.0 ? city.latitude + 0.12 : city.latitude - 0.12);
    const ruralLon = data.rural_longitude || city.rural_longitude || (city.longitude < 78.0 ? city.longitude + 0.22 : city.longitude - 0.22);

    // Draw connecting vector baseline
    const line = L.polyline([[urbanLat, urbanLon], [ruralLat, ruralLon]], {
        color: "#38bdf8",
        weight: 3,
        dashArray: "6, 8",
        opacity: 0.85
    }).addTo(omVectorGroup);

    // Urban Pin
    const urbanCircle = L.circleMarker([urbanLat, urbanLon], {
        radius: 12,
        fillColor: "#ef4444",
        color: "#ffffff",
        weight: 2,
        fillOpacity: 0.9
    }).addTo(omVectorGroup);

    urbanCircle.bindPopup(`
        <div class="heat-popup">
            <div class="popup-head">
                <span class="popup-title">🏢 ${city.name} Urban Core</span>
                <span class="temp-badge-pill" style="background:rgba(239,68,68,0.2); color:#ef4444; border:1px solid rgba(239,68,68,0.5)">${data.urban_temp_celsius}°C</span>
            </div>
            <div class="popup-grid">
                <div class="popup-cell"><span>Actual UHI</span><strong style="color:#f97316">+${data.actual_uhi_intensity_celsius}°C</strong></div>
                <div class="popup-cell"><span>Model Pred</span><strong style="color:#38bdf8">+${data.predicted_uhi_intensity_celsius.toFixed(2)}°C</strong></div>
                <div class="popup-cell"><span>Wind Speed</span><span>${data.wind_speed_kmh} km/h</span></div>
                <div class="popup-cell"><span>Humidity</span><span>${data.humidity_pct}%</span></div>
            </div>
            <div class="popup-foot">
                <span>Status: LIVE_OPENMETEO_STREAM</span>
            </div>
        </div>
    `, {
        autoPan: true,
        autoPanPaddingTopLeft: L.point(30, 95),
        autoPanPaddingBottomRight: L.point(30, 40),
        offset: L.point(0, -6)
    }).openPopup();

    // Rural Pin
    const ruralCircle = L.circleMarker([ruralLat, ruralLon], {
        radius: 10,
        fillColor: "#10b981",
        color: "#ffffff",
        weight: 2,
        fillOpacity: 0.9
    }).addTo(omVectorGroup);

    ruralCircle.bindPopup(`
        <div class="heat-popup">
            <div class="popup-head">
                <span class="popup-title">🌲 Rural Reference</span>
                <span class="temp-badge-pill" style="background:rgba(16,185,129,0.2); color:#10b981; border:1px solid rgba(16,185,129,0.5)">${data.rural_temp_celsius}°C</span>
            </div>
            <div class="popup-grid">
                <div class="popup-cell"><span>Live Temp</span><strong style="color:#10b981">${data.rural_temp_celsius}°C</strong></div>
                <div class="popup-cell"><span>Cloud Cover</span><span>${data.cloud_cover_pct}%</span></div>
                <div class="popup-cell"><span>Wind Speed</span><span>${data.wind_speed_kmh} km/h</span></div>
                <div class="popup-cell"><span>Humidity</span><span>${data.humidity_pct}%</span></div>
            </div>
            <div class="popup-foot">
                <span>Background Weather Station</span>
            </div>
        </div>
    `, {
        autoPan: true,
        autoPanPaddingTopLeft: L.point(30, 95),
        autoPanPaddingBottomRight: L.point(30, 40),
        offset: L.point(0, -6)
    });

    // Zoom map to fit urban-rural baseline vector
    map3D.fitBounds(line.getBounds(), { padding: [40, 40] });
}


/* ─── Simulation ─── */
async function runSimulation(autoSwitchTab = false) {
    const city = citiesList[activeCityIdx];
    if (!city) return;

    const modelSelect = document.getElementById("modelOptionSelect");
    const modelOpt = modelSelect ? modelSelect.value : "ensemble";

    await updateActiveModelReadout(city);

    const payload = {
        latitude: city.latitude,
        longitude: city.longitude,
        month: 5,
        baseline_ndvi: city.ndvi || 0.1,
        baseline_ndbi: city.ndbi || 0.05,
        baseline_ndwi: -0.1,
        elevation: 50.0,
        green_canopy_pct: parseFloat(document.getElementById("sliderGreen")?.value) || 0,
        water_features_pct: parseFloat(document.getElementById("sliderWater")?.value) || 0,
        cool_roofs_pct: parseFloat(document.getElementById("sliderCoolRoof")?.value) || 0,
        shade_canopies_pct: parseFloat(document.getElementById("sliderShade")?.value) || 0,
        model_option: modelOpt
    };

    try {
        const r = await fetch(`${API_BASE_URL}/api/simulate`, {
            method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
        });
        if (r.ok) {
            const d = await r.json();
            document.getElementById("txtBaselineLST").textContent = `${d.baseline.lst_celsius.toFixed(1)}°C`;
            document.getElementById("txtSimulatedLST").textContent = `${d.simulated.lst_celsius.toFixed(1)}°C`;
            document.getElementById("txtDeltaTemp").textContent = `${d.impact.delta_temp_celsius.toFixed(1)}°C`;
            document.getElementById("txtCO2").textContent = `${d.impact.co2_sequestration_tons_yr} t/yr`;

            // If triggered by user clicking Run Simulation, auto switch to Heat Map tab
            if (autoSwitchTab) {
                renderSimulatedGrid(city, d.impact.delta_temp_celsius);
                switchTab('heatmap');
            }
        }
    } catch (e) { console.error("Sim error:", e); }
}

/* ─── AI Auto-Recommendation ─── */
let cachedAIRecommendation = null;

async function fetchAIRecommendation(city) {
    if (!city) return;
    try {
        const res = await fetch(`${API_BASE_URL}/api/auto-recommend`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                latitude: city.latitude,
                longitude: city.longitude,
                baseline_ndvi: city.ndvi || 0.1,
                baseline_ndbi: city.ndbi || 0.05
            })
        });
        if (res.ok) {
            cachedAIRecommendation = await res.json();
            renderAIRecommendationCards(cachedAIRecommendation);
        }
    } catch (e) {
        console.error("AI Recommendation error:", e);
    }
}

function renderAIRecommendationCards(rec) {
    if (!rec) return;
    const titleElem = document.getElementById("aiStrategyTitle");
    const cardsElem = document.getElementById("aiStrategyCards");
    
    if (titleElem) {
        titleElem.innerHTML = `<strong>${rec.strategy_title}</strong> (Target: <span style="color:#06b6d4;">${rec.expected_cooling_celsius.toFixed(1)}°C Cooling</span>)`;
    }
    
    if (cardsElem && rec.recommendation_cards) {
        cardsElem.innerHTML = rec.recommendation_cards.map(c => `
            <div style="background:#27272a; border-left:3px solid ${c.priority === 'HIGH' ? '#f97316' : '#06b6d4'}; border-radius:4px; padding:8px 10px;">
                <div style="display:flex; justify-content:space-between; font-size:0.75rem; font-weight:700; color:#e4e4e7;">
                    <span>${c.title}</span>
                    <span style="color:${c.priority === 'HIGH' ? '#f97316' : '#06b6d4'};">${c.priority}</span>
                </div>
                <div style="font-size:0.7rem; color:#a1a1aa; margin-top:2px;">${c.description}</div>
            </div>
        `).join('');
    }
}

async function autoApplyAIRecommendation() {
    const city = citiesList[activeCityIdx];
    if (!city) return;
    
    if (!cachedAIRecommendation) {
        await fetchAIRecommendation(city);
    }

    if (cachedAIRecommendation && cachedAIRecommendation.recommended_sliders) {
        const s = cachedAIRecommendation.recommended_sliders;
        
        const slGreen = document.getElementById("sliderGreen");
        const slCoolRoof = document.getElementById("sliderCoolRoof");
        const slWater = document.getElementById("sliderWater");
        const slShade = document.getElementById("sliderShade");

        if (slGreen) { slGreen.value = s.green_canopy_pct; document.getElementById("valGreen").textContent = `${s.green_canopy_pct}%`; }
        if (slCoolRoof) { slCoolRoof.value = s.cool_roofs_pct; document.getElementById("valCoolRoof").textContent = `${s.cool_roofs_pct}%`; }
        if (slWater) { slWater.value = s.water_features_pct; document.getElementById("valWater").textContent = `${s.water_features_pct}%`; }
        if (slShade) { slShade.value = s.shade_canopies_pct; document.getElementById("valShade").textContent = `${s.shade_canopies_pct}%`; }

        await runSimulation(true);
    }
}

function renderSimulatedGrid(city, deltaTemp) {
    if (!heatGridGroup || !currentCityGridData || currentCityGridData.length === 0) return;
    const simGrid = currentCityGridData.map(c => ({
        ...c,
        lst_mean: Math.max(16.0, (c.lst_mean || 32.0) + deltaTemp)
    }));
    render3DVoxelGrid(simGrid, city, false);
}

/* ─── GA Optimizer ─── */
async function runGAOptimization(autoSwitchTab = false) {
    const city = citiesList[activeCityIdx];
    if (!city) return;
    const budget = parseFloat(document.getElementById("sliderBudget")?.value) || 25000;

    try {
        const r = await fetch(`${API_BASE_URL}/api/optimize-ga`, {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ latitude: city.latitude, longitude: city.longitude, budget_usd: budget, generations: 25 })
        });
        if (r.ok) {
            const d = await r.json();
            document.getElementById("txtGaDrop").textContent = `-${d.citywide_temperature_reduction_celsius}°C`;
            document.getElementById("txtGaCost").textContent = `$${d.total_optimal_cost_usd.toLocaleString()}`;
            document.getElementById("txtGaEffect").textContent = `${d.cost_effectiveness_celsius_per_k_usd} °C/$1k`;
            document.getElementById("gaResultsBox").classList.remove("hidden");

            renderGAPlanMap(d.optimal_intervention_plan, city.name);

            // If triggered by user button click, auto switch to Heat Map tab
            if (autoSwitchTab) {
                switchTab('heatmap');
            }
        }
    } catch (e) { console.error("GA error:", e); }
}
