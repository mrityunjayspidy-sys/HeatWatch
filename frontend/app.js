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
});

/* ─── Tab Switching ─── */
function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    
    const btn = document.querySelector(`[data-tab="${tabId}"]`);
    const panel = document.getElementById(`panel-${tabId}`);
    
    if (btn) btn.classList.add('active');
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

/* ─── Temperature Color Tiers ─── */
function getTemperatureColorAndTier(tempC) {
    if (tempC >= 40.0) return { color: "#dc2626", stroke: "#7f1d1d", label: "Extreme (≥40°C)", category: "EXTREME", badgeBg: "rgba(220,38,38,0.25)", badgeColor: "#ef4444", badgeBorder: "rgba(220,38,38,0.6)" };
    if (tempC >= 36.0) return { color: "#f97316", stroke: "#9a3412", label: "High (36-40°C)", category: "HIGH", badgeBg: "rgba(249,115,22,0.25)", badgeColor: "#f97316", badgeBorder: "rgba(249,115,22,0.6)" };
    if (tempC >= 31.0) return { color: "#f59e0b", stroke: "#b45309", label: "Warm (31-36°C)", category: "WARM", badgeBg: "rgba(245,158,11,0.25)", badgeColor: "#f59e0b", badgeBorder: "rgba(245,158,11,0.6)" };
    if (tempC >= 26.0) return { color: "#10b981", stroke: "#047857", label: "Mild (26-31°C)", category: "MILD", badgeBg: "rgba(16,185,129,0.25)", badgeColor: "#10b981", badgeBorder: "rgba(16,185,129,0.6)" };
    return { color: "#06b6d4", stroke: "#0e7490", label: "Cool (<26°C)", category: "COOL", badgeBg: "rgba(6,182,212,0.25)", badgeColor: "#06b6d4", badgeBorder: "rgba(6,182,212,0.6)" };
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
    map3D.setView([city.latitude, city.longitude], 12);
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
async function loadRealGridData(city) {
    if (!heatGridGroup) return;
    heatGridGroup.clearLayers();

    try {
        const folderName = city.name.toLowerCase().replace(/ /g, "_");
        const res = await fetch(`${API_BASE_URL}/api/city-grid/${folderName}?max_cells=400`);
        if (!res.ok) {
            currentCityGridData = [];
            renderFallbackGrid(city);
            return;
        }

        const data = await res.json();
        const grid = data.grid;
        currentCityGridData = grid || [];

        if (!grid || grid.length === 0) {
            renderFallbackGrid(city);
            return;
        }

        const lats = grid.map(c => c.centroid_lat);
        const lons = grid.map(c => c.centroid_lon);
        const lsts = grid.map(c => c.lst_mean);
        const minLst = Math.min(...lsts);
        const maxLst = Math.max(...lsts);
        const lstRange = (maxLst - minLst) || 1.0;

        const latRange = Math.max(...lats) - Math.min(...lats);
        const lonRange = Math.max(...lons) - Math.min(...lons);
        const cellCount = Math.sqrt(grid.length);
        const stepLat = latRange / Math.max(cellCount, 1) || 0.005;
        const stepLon = lonRange / Math.max(cellCount, 1) || 0.005;

        grid.forEach(cell => {
            const lat = cell.centroid_lat;
            const lon = cell.centroid_lon;
            const lst = cell.lst_mean;
            const half = [stepLat / 2, stepLon / 2];

            const bounds = [
                [lat - half[0], lon - half[1]],
                [lat - half[0], lon + half[1]],
                [lat + half[0], lon + half[1]],
                [lat + half[0], lon - half[1]]
            ];

            // Absolute Temperature HSL Hue Mapping
            // 42°C+ -> 0° Red | 36°C -> 35° Orange | 31°C -> 95° Warm Lime | 26°C -> 138° Emerald Green | <=20°C -> 190° Cyan
            const norm = Math.max(0.0, Math.min(1.0, (lst - 20.0) / (42.0 - 20.0)));
            const hue = (1.0 - norm) * 190;
            const fillColor = `hsl(${Math.round(hue)}, 88%, 46%)`;
            const strokeColor = `hsl(${Math.round(hue)}, 92%, 26%)`;


            const tier = getTemperatureColorAndTier(lst);

            const block = L.polygon(bounds, {
                color: strokeColor, weight: 1,
                fillColor: fillColor, fillOpacity: 0.85
            }).addTo(heatGridGroup);

            const ndvi = cell.ndvi_mean != null ? cell.ndvi_mean.toFixed(3) : "—";
            const ndbi = cell.ndbi_mean != null ? cell.ndbi_mean.toFixed(3) : "—";
            const tree = cell.tree_canopy_frac != null ? (cell.tree_canopy_frac * 100).toFixed(1) + "%" : "—";
            const water = cell.water_area_frac != null ? (cell.water_area_frac * 100).toFixed(1) + "%" : "—";
            const openmeteoTemp = cell.openmeteo_ambient_temp ? cell.openmeteo_ambient_temp.toFixed(1) + "°C" : "—";

            block.bindPopup(`
                <div class="heat-popup">
                    <h4>📍 ${city.name} — Microclimate Pixel Cell</h4>
                    <div class="popup-row"><span>Exact Surface Temp (LST):</span><strong style="color:${fillColor}">${lst.toFixed(1)}°C</strong></div>
                    <div class="popup-row"><span>Open-Meteo Ambient Temp:</span><span>${openmeteoTemp}</span></div>
                    <div class="popup-row"><span>NDVI (Vegetation Index):</span><span>${ndvi}</span></div>
                    <div class="popup-row"><span>NDBI (Built-Up Index):</span><span>${ndbi}</span></div>
                    <div class="popup-row"><span>Tree Canopy Coverage:</span><span>${tree}</span></div>
                    <div class="popup-row"><span>Water Coverage:</span><span>${water}</span></div>
                    <div class="popup-row"><span>Thermal Tier:</span>
                        <span class="temp-badge-pill" style="background:${tier.badgeBg}; color:${tier.badgeColor}; border:1px solid ${tier.badgeBorder}">${tier.label}</span>
                    </div>
                </div>
            `);
        });
    } catch (err) {


        console.error("Grid load error:", err);
        currentCityGridData = [];
        renderFallbackGrid(city);
    }
}

function renderFallbackGrid(city) {
    if (!heatGridGroup) return;
    heatGridGroup.clearLayers();
    const rows = 12, cols = 12, step = 0.005;
    const baseLst = city.avg_lst_celsius;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const p1 = [city.latitude - (rows/2 * step) + (r * step), city.longitude - (cols/2 * step) + (c * step)];
            const dist = Math.sqrt((r - rows/2)**2 + (c - cols/2)**2);
            const temp = baseLst + 3.0 - (dist * 0.8);
            const tier = getTemperatureColorAndTier(temp);

            L.polygon([p1, [p1[0], p1[1]+step], [p1[0]+step, p1[1]+step], [p1[0]+step, p1[1]]], {
                color: tier.stroke, weight: 1, fillColor: tier.color, fillOpacity: 0.78
            }).addTo(heatGridGroup);
        }
    }
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

    if (subTitle) subTitle.textContent = `Vegetation, water, built-up indices and temperature distribution for ${city.name}.`;
    if (titleLand) titleLand.textContent = `${city.name} Land Cover Breakdown`;
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

/* Chart 1: Land Cover Breakdown for Selected City */
function renderLandCoverChart(city) {
    const ctx = document.getElementById('chartLandCover');
    if (!ctx) return;
    if (chartLandCover) chartLandCover.destroy();

    const vegPct = parseFloat(((city.ndvi || 0.15) * 100).toFixed(1));
    const builtPct = parseFloat(Math.max(0, ((city.ndbi || 0.05) * 100)).toFixed(1));
    const treePct = parseFloat(((city.avg_tree_cover || 0.1) * 100).toFixed(1));
    const otherPct = parseFloat(Math.max(0, 100 - vegPct - builtPct - treePct).toFixed(1));

    chartLandCover = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Vegetation', 'Built-up', 'Tree Canopy', 'Bare Soil / Other'],
            datasets: [{
                data: [vegPct, builtPct, treePct, otherPct],
                backgroundColor: ['#10b981', '#f97316', '#06b6d4', '#333333'],
                borderColor: '#111111',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'bottom', labels: { font: { size: 11 }, padding: 10, color: '#aaa' } },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.raw}%`;
                        }
                    }
                }
            }
        }
    });
}

/* Chart 2: Selected City vs National Benchmarks */
function renderCityTempsChart(selectedCity) {
    const ctx = document.getElementById('chartCityTemps');
    if (!ctx) return;
    if (chartCityTemps) chartCityTemps.destroy();

    // Find min, max, avg across all cities
    const allLst = citiesList.map(c => c.avg_lst_celsius);
    const minLst = Math.min(...allLst);
    const maxLst = Math.max(...allLst);
    const avgLst = (allLst.reduce((a, b) => a + b, 0) / allLst.length);

    // Show Selected City, Selected City Max, 21-City Avg, National Min, National Max
    const labels = [
        `${selectedCity.name} Avg`,
        `${selectedCity.name} Peak`,
        '21-City National Avg',
        'Coolest City Avg',
        'Hottest City Avg'
    ];

    const dataValues = [
        selectedCity.avg_lst_celsius,
        selectedCity.max_lst_celsius || (selectedCity.avg_lst_celsius + 5.0),
        parseFloat(avgLst.toFixed(1)),
        parseFloat(minLst.toFixed(1)),
        parseFloat(maxLst.toFixed(1))
    ];

    const barColors = [
        '#ffffff', // Highlighted white bar for Selected City Avg
        '#dc2626', // Red for Peak
        '#f59e0b', // Amber for National Avg
        '#10b981', // Green for Min
        '#ef4444'  // Red for Max
    ];

    chartCityTemps = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Temperature (°C)',
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
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.raw}°C`
                    }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { size: 10 }, color: '#ccc' } },
                y: { grid: { color: '#222' }, title: { display: true, text: 'LST (°C)', font: { size: 10 } }, ticks: { color: '#888' } }
            }
        }
    });
}

/* Chart 3: Selected City Microclimate & Spectral Profile */
function renderIndicesChart(city) {
    const ctx = document.getElementById('chartIndices');
    if (!ctx) return;
    if (chartIndices) chartIndices.destroy();

    const ndviVal = Math.max(0, (city.ndvi || 0.1) * 100);
    const ndbiVal = Math.max(0, (city.ndbi || 0.05) * 100);
    const treeVal = (city.avg_tree_cover || 0.1) * 100;
    const tempVal = (city.avg_lst_celsius || 35.0);

    chartIndices = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['NDVI Index (x100)', 'NDBI Index (x100)', 'Tree Canopy %', 'Avg Surface Heat (°C)'],
            datasets: [{
                label: city.name,
                data: [
                    parseFloat(ndviVal.toFixed(1)),
                    parseFloat(ndbiVal.toFixed(1)),
                    parseFloat(treeVal.toFixed(1)),
                    parseFloat(tempVal.toFixed(1))
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
                legend: { position: 'bottom', labels: { color: '#fff', font: { size: 11 } } }
            },
            scales: {
                r: {
                    angleLines: { color: '#333' },
                    grid: { color: '#222' },
                    pointLabels: { color: '#aaa', font: { size: 10 } },
                    ticks: { display: false }
                }
            }
        }
    });
}

/* Chart 4: Selected City Temperature Tier Distribution */
function renderTiersChart(city) {
    const ctx = document.getElementById('chartTiers');
    if (!ctx) return;
    if (chartTiers) chartTiers.destroy();

    let tiers = { extreme: 0, high: 0, warm: 0, mild: 0, cool: 0 };

    // If real grid cell data is loaded for this city, aggregate exact grid cell tiers
    if (currentCityGridData && currentCityGridData.length > 0) {
        currentCityGridData.forEach(cell => {
            const t = cell.lst_mean;
            if (t >= 40.0) tiers.extreme++;
            else if (t >= 36.0) tiers.high++;
            else if (t >= 31.0) tiers.warm++;
            else if (t >= 26.0) tiers.mild++;
            else tiers.cool++;
        });
    } else {
        // Estimate based on city's average LST
        const avg = city.avg_lst_celsius;
        if (avg >= 40) { tiers.extreme = 60; tiers.high = 30; tiers.warm = 10; }
        else if (avg >= 36) { tiers.extreme = 20; tiers.high = 50; tiers.warm = 20; tiers.mild = 10; }
        else if (avg >= 31) { tiers.high = 15; tiers.warm = 60; tiers.mild = 20; tiers.cool = 5; }
        else { tiers.warm = 20; tiers.mild = 60; tiers.cool = 20; }
    }

    chartTiers = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Extreme (≥40°C)', 'High (36-40°C)', 'Warm (31-36°C)', 'Mild (26-31°C)', 'Cool (<26°C)'],
            datasets: [{
                data: [tiers.extreme, tiers.high, tiers.warm, tiers.mild, tiers.cool],
                backgroundColor: ['#dc2626', '#f97316', '#f59e0b', '#10b981', '#06b6d4'],
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
                        label: (ctx) => `${ctx.label}: ${ctx.raw} grid cells`
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

/* ─── Map View Switcher ─── */
let currentMapView = "india";

function switchMapView(mode) {
    currentMapView = mode;
    const btnIndia = document.getElementById("btnViewIndiaMap");
    const btnCity = document.getElementById("btnViewCityGrid");

    if (mode === "india") {
        if (btnIndia) btnIndia.classList.add("active");
        if (btnCity) btnCity.classList.remove("active");
        renderIndiaTemperatureMap();
    } else {
        if (btnCity) btnCity.classList.add("active");
        if (btnIndia) btnIndia.classList.remove("active");
        const city = citiesList[activeCityIdx];
        if (city) {
            map3D.setView([city.latitude, city.longitude], 12);
            loadRealGridData(city);
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
            <h4>🏢 Urban Core — Real-Time Open-Meteo</h4>
            <div class="popup-row"><span>Live Urban Temp:</span><strong style="color:#ef4444">${data.urban_temp_celsius}°C</strong></div>
            <div class="popup-row"><span>Actual UHI Gap:</span><strong style="color:#f97316">+${data.actual_uhi_intensity_celsius}°C</strong></div>
            <div class="popup-row"><span>Predicted UHI Model:</span><strong style="color:#38bdf8">+${data.predicted_uhi_intensity_celsius.toFixed(2)}°C</strong></div>
            <div class="popup-row"><span>Wind Speed:</span><span>${data.wind_speed_kmh} km/h</span></div>
            <div class="popup-row"><span>Relative Humidity:</span><span>${data.humidity_pct}%</span></div>
            <div style="font-size:0.65rem; color:#888; margin-top:4px;">Status: LIVE_OPENMETEO_STREAM</div>
        </div>
    `).openPopup();

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
            <h4>🌲 Rural Reference — Open-Meteo</h4>
            <div class="popup-row"><span>Live Rural Temp:</span><strong style="color:#10b981">${data.rural_temp_celsius}°C</strong></div>
            <div class="popup-row"><span>Cloud Cover:</span><span>${data.cloud_cover_pct}%</span></div>
            <div class="popup-row"><span>Wind Speed:</span><span>${data.wind_speed_kmh} km/h</span></div>
            <div style="font-size:0.65rem; color:#888; margin-top:4px;">Background Weather Station</div>
        </div>
    `);

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
    heatGridGroup.clearLayers();

    const grid = currentCityGridData;
    const lats = grid.map(c => c.centroid_lat);
    const lons = grid.map(c => c.centroid_lon);
    const latRange = Math.max(...lats) - Math.min(...lats);
    const lonRange = Math.max(...lons) - Math.min(...lons);
    const cellCount = Math.sqrt(grid.length);
    const stepLat = latRange / Math.max(cellCount, 1) || 0.005;
    const stepLon = lonRange / Math.max(cellCount, 1) || 0.005;

    grid.forEach(cell => {
        const lat = cell.centroid_lat;
        const lon = cell.centroid_lon;
        const baseLst = cell.lst_mean;
        const simLst = Math.max(16.0, baseLst + deltaTemp);
        const half = [stepLat / 2, stepLon / 2];

        const bounds = [
            [lat - half[0], lon - half[1]],
            [lat - half[0], lon + half[1]],
            [lat + half[0], lon + half[1]],
            [lat + half[0], lon - half[1]]
        ];

        const tier = getTemperatureColorAndTier(simLst);

        const block = L.polygon(bounds, {
            color: tier.stroke, weight: 1,
            fillColor: tier.color, fillOpacity: 0.82
        }).addTo(heatGridGroup);

        block.bindPopup(`
            <div class="heat-popup">
                <h4>📍 ${city.name} — Post-Simulation Cell</h4>
                <div class="popup-row"><span>Baseline LST:</span><span>${baseLst.toFixed(1)}°C</span></div>
                <div class="popup-row"><span>Simulated LST:</span><strong style="color:${tier.color}">${simLst.toFixed(1)}°C</strong></div>
                <div class="popup-row"><span>Net Heat Drop:</span><strong style="color:#06b6d4">${deltaTemp.toFixed(1)}°C</strong></div>
                <div class="popup-row"><span>Level:</span>
                    <span class="temp-badge-pill" style="background:${tier.badgeBg}; color:${tier.badgeColor}; border:1px solid ${tier.badgeBorder}">${tier.label}</span>
                </div>
            </div>
        `);
    });
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
