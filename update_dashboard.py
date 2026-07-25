import os
import time
import glob
import json
import io
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

SAVE_DIR = "NSE_Cash_Bhavcopies_12M"
OUTPUT_HTML = "index.html"

# Common Headers for NSE Requests
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/"
}

# ==========================================
# STEP 1: DOWNLOAD MISSING CASH BHAVCOPIES
# ==========================================
def download_bhavcopies():
    print("=== STEP 1: DOWNLOADING NSE CASH BHAVCOPIES ===")
    os.makedirs(SAVE_DIR, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    print("Initializing session and fetching cookies from NSE...")
    try:
        session.get("https://www.nseindia.com", timeout=10)
        print("Session established successfully.\n")
    except Exception as e:
        print(f"Warning: Could not fetch initial cookies: {e}\n")

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365)
    business_days = pd.date_range(start=start_date, end=end_date, freq='B')

    print(f"Checking data for {len(business_days)} potential trading days...")
    successful_downloads = 0
    already_existed = 0

    for date in business_days:
        date_str = date.strftime('%d%m%Y')
        filename = f"sec_bhavdata_full_{date_str}.csv"
        file_path = os.path.join(SAVE_DIR, filename)

        # Skip if file already downloaded
        if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
            already_existed += 1
            continue

        url = f"https://archives.nseindia.com/products/content/{filename}"

        try:
            response = session.get(url, timeout=10)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"[SUCCESS] Downloaded: {filename}")
                successful_downloads += 1
            elif response.status_code == 404:
                print(f"[SKIPPED] Market Holiday / No file: {date.strftime('%Y-%m-%d')}")
            elif response.status_code == 403:
                print(f"[BLOCKED] 403 Forbidden on {date.strftime('%Y-%m-%d')}. Rate limit reached.")
                break
            else:
                print(f"[ERROR] Failed {date.strftime('%Y-%m-%d')} - Status Code: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Exception on {date.strftime('%Y-%m-%d')}: {e}")

        time.sleep(1) # Friendly sleep to avoid rate limits

    print(f"Download Summary: {already_existed} existing, {successful_downloads} newly downloaded.\n")

# ==========================================
# STEP 2: FETCH NIFTY 500 LIST & SECTORS
# ==========================================
def fetch_nifty500():
    print("=== STEP 2: FETCHING NIFTY 500 LIST & SECTORS ===")
    nifty500_url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    sector_map = {}
    nifty500_symbols = set()

    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        session.get("https://www.nseindia.com", timeout=10)
        response = session.get(nifty500_url, timeout=10)

        if response.status_code == 200:
            df_n500 = pd.read_csv(io.StringIO(response.text))
            df_n500['Symbol'] = df_n500['Symbol'].astype(str).str.strip()
            df_n500['Industry'] = df_n500['Industry'].astype(str).str.strip()
            sector_map = dict(zip(df_n500['Symbol'], df_n500['Industry']))
            nifty500_symbols = set(df_n500['Symbol'])
            print(f"Successfully fetched {len(nifty500_symbols)} Nifty 500 stocks.\n")
        else:
            print(f"Warning: Failed to fetch Nifty 500 list (Status {response.status_code}).\n")
    except Exception as e:
        print(f"Warning: Exception fetching Nifty 500: {e}\n")

    return sector_map, nifty500_symbols

# ==========================================
# STEP 3: PROCESS DATA & BUILD DASHBOARD
# ==========================================
def build_dashboard(sector_map, nifty500_symbols):
    print("=== STEP 3: PROCESSING DATA & BUILDING INDEX.HTML ===")
    csv_files = glob.glob(os.path.join(SAVE_DIR, "sec_bhavdata_full_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {SAVE_DIR}.")

    df_list = []
    for f in csv_files:
        try:
            temp_df = pd.read_csv(f)
            temp_df.columns = temp_df.columns.str.strip()
            if 'SYMBOL' in temp_df.columns: temp_df['SYMBOL'] = temp_df['SYMBOL'].astype(str).str.strip()
            if 'SERIES' in temp_df.columns: temp_df['SERIES'] = temp_df['SERIES'].astype(str).str.strip()
            df_list.append(temp_df)
        except Exception:
            continue

    df = pd.concat(df_list, ignore_index=True)

    numeric_cols = ['OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'TTL_TRD_QNTY', 'DELIV_QTY', 'DELIV_PER']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('-', '0').str.strip(), errors='coerce').fillna(0)

    df['DATE1'] = pd.to_datetime(df['DATE1'].astype(str).str.strip(), format='%d-%b-%Y', errors='coerce')
    df = df.dropna(subset=['DATE1']).sort_values('DATE1')

    df['SECTOR'] = df['SYMBOL'].map(sector_map).fillna('Other')
    df['IS_NIFTY500'] = df['SYMBOL'].isin(nifty500_symbols)

    max_date = df['DATE1'].max()
    last_7d = max_date - timedelta(days=7)
    last_30d = max_date - timedelta(days=30)

    df_7d = df[df['DATE1'] >= last_7d]
    df_30d = df[df['DATE1'] >= last_30d]

    screener = df_30d.groupby(['SYMBOL', 'SERIES']).agg(
        latest_close=('CLOSE_PRICE', 'last'),
        avg_deliv_qty_30d=('DELIV_QTY', 'mean'),
        avg_deliv_pct_30d=('DELIV_PER', 'mean'),
        SECTOR=('SECTOR', 'last'),
        IS_NIFTY500=('IS_NIFTY500', 'last')
    ).reset_index()

    screener_7d = df_7d.groupby(['SYMBOL', 'SERIES']).agg(
        avg_deliv_qty_7d=('DELIV_QTY', 'mean'),
        avg_deliv_pct_7d=('DELIV_PER', 'mean')
    ).reset_index()

    screener = pd.merge(screener, screener_7d, on=['SYMBOL', 'SERIES'], how='left').fillna(0)
    screener = screener.round({
        'latest_close': 2, 'avg_deliv_pct_30d': 2, 'avg_deliv_pct_7d': 2, 
        'avg_deliv_qty_30d': 0, 'avg_deliv_qty_7d': 0
    })

    series_list = sorted(df['SERIES'].unique().tolist())
    sector_list = sorted([s for s in screener['SECTOR'].unique() if s != 'Other'])
    screener_json = screener.to_dict(orient='records')

    chart_data = {}
    for (symbol, series), group in df.groupby(['SYMBOL', 'SERIES']):
        if series not in chart_data:
            chart_data[series] = {}
        chart_data[series][symbol] = {
            'd': group['DATE1'].dt.strftime('%Y-%m-%d').tolist(),
            'c': group['CLOSE_PRICE'].tolist(),
            'dq': group['DELIV_QTY'].tolist()
        }

    html_content = f"""<!DOCTYPE html>
<html lang="en" data-bs-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NSE Pro Analytics Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css" rel="stylesheet">
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ transition: background-color 0.3s, color 0.3s; font-size: 0.9rem; }}
        .card {{ box-shadow: 0 2px 5px rgba(0,0,0,0.1); border: none; margin-bottom: 20px; transition: 0.3s; }}
        [data-bs-theme="dark"] .card {{ background-color: #2b3035; box-shadow: 0 2px 5px rgba(0,0,0,0.5); }}
        .metric-value {{ font-size: 1.2rem; font-weight: bold; color: #0d6efd; }}
        [data-bs-theme="dark"] .metric-value {{ color: #6ea8fe; }}
        .metric-label {{ font-size: 0.75rem; color: #6c757d; text-transform: uppercase; }}
        #screenerTable tbody tr:hover {{ cursor: pointer; }}
        .nav-tabs .nav-link {{ cursor: pointer; color: #495057; border-bottom: 2px solid transparent; }}
        .nav-tabs .nav-link.active {{ font-weight: bold; color: #0d6efd !important; border-bottom: 2px solid #0d6efd; background: transparent; }}
        [data-bs-theme="dark"] .nav-tabs .nav-link {{ color: #adb5bd; }}
        [data-bs-theme="dark"] .nav-tabs .nav-link.active {{ color: #6ea8fe !important; border-bottom: 2px solid #6ea8fe; }}
        .correlation-positive {{ color: #198754; font-weight: bold; }}
        .correlation-negative {{ color: #dc3545; font-weight: bold; }}
        [data-bs-theme="dark"] .correlation-positive {{ color: #75b798; }}
        [data-bs-theme="dark"] .correlation-negative {{ color: #ea868f; }}
        .z-badge {{ font-size: 0.85rem; padding: 4px 8px; border-radius: 4px; font-weight: 600; display: inline-block; width: 60px; text-align: center; }}
        .z-hot {{ background-color: rgba(25, 135, 84, 0.2); color: #198754; border: 1px solid rgba(25, 135, 84, 0.5); }}
        .z-cold {{ background-color: rgba(220, 53, 69, 0.2); color: #dc3545; border: 1px solid rgba(220, 53, 69, 0.5); }}
        [data-bs-theme="dark"] .z-hot {{ color: #75b798; border-color: #75b798; }}
        [data-bs-theme="dark"] .z-cold {{ color: #ea868f; border-color: #ea868f; }}
    </style>
</head>
<body>

<nav class="navbar navbar-expand-lg bg-body-tertiary mb-4">
  <div class="container-fluid">
    <a class="navbar-brand fw-bold" href="#">📉 NSE Analytics Engine</a>
    <div class="d-flex align-items-center">
        <label class="form-check-label me-2" for="themeToggle">Dark Mode</label>
        <div class="form-check form-switch"><input class="form-check-input" type="checkbox" id="themeToggle"></div>
    </div>
  </div>
</nav>

<div class="container-fluid">
    <div class="row">
        <div class="col-lg-9">
            <ul class="nav nav-tabs mb-3" id="myTab">
              <li class="nav-item"><a class="nav-link active" data-mode="all">🌐 All Market</a></li>
              <li class="nav-item"><a class="nav-link" data-mode="nifty500">★ Nifty 500</a></li>
            </ul>

            <div class="row text-center mb-3">
                <div class="col"><div class="card p-2"><div class="metric-label">Close</div><div class="metric-value" id="m_close">₹0.00</div></div></div>
                <div class="col"><div class="card p-2"><div class="metric-label">Avg Deliv Qty (7D)</div><div class="metric-value" id="m_dq7">0</div></div></div>
                <div class="col"><div class="card p-2"><div class="metric-label">Avg Deliv % (7D)</div><div class="metric-value" id="m_d7">0.0%</div></div></div>
                <div class="col"><div class="card p-2"><div class="metric-label">Z-Score (Price)</div><div class="metric-value" id="m_zp">-</div></div></div>
                <div class="col"><div class="card p-2"><div class="metric-label">Z-Score (Delivery)</div><div class="metric-value" id="m_zdq">-</div></div></div>
            </div>

            <div class="card p-3"><div id="plotlyChart" style="height: 400px; width: 100%;"></div></div>
            
            <div class="card p-3">
                <h5 class="border-bottom pb-2 mb-3">Master Screener & Filters</h5>
                
                <div class="row g-2 mb-2">
                    <div class="col-md-3">
                        <label class="form-label fw-bold small mb-1">Series</label>
                        <select id="seriesFilter" class="form-select form-select-sm">
                            {"".join([f'<option value="{s}">{s}</option>' for s in series_list])}
                        </select>
                    </div>
                    <div class="col-md-3">
                        <label class="form-label fw-bold small mb-1">Sector</label>
                        <select id="sectorFilter" class="form-select form-select-sm">
                            <option value="">All Sectors</option>
                            {"".join([f'<option value="{s}">{s}</option>' for s in sector_list])}
                        </select>
                    </div>
                    <div class="col-md-3">
                        <label class="form-label fw-bold small mb-1 text-danger">Z-Score Lookback</label>
                        <select id="zLookback" class="form-select form-select-sm border-danger">
                            <option value="3">3 Months (63 Days)</option>
                            <option value="6" selected>6 Months (126 Days)</option>
                            <option value="9">9 Months (189 Days)</option>
                            <option value="12">12 Months (252 Days)</option>
                        </select>
                    </div>
                    <div class="col-md-3 d-flex align-items-end">
                        <button id="resetFilters" class="btn btn-sm btn-outline-secondary w-100">Reset All Filters</button>
                    </div>
                </div>

                <div class="row g-2 mb-4 pb-3 border-bottom">
                    <div class="col-md-3">
                        <label class="form-label fw-bold small mb-1">Min Deliv Qty(7D)</label>
                        <input type="number" id="f_dq7" class="form-control form-control-sm" placeholder="e.g. 50000">
                    </div>
                    <div class="col-md-3">
                        <label class="form-label fw-bold small mb-1">Min Deliv Qty(30D)</label>
                        <input type="number" id="f_dq30" class="form-control form-control-sm" placeholder="e.g. 50000">
                    </div>
                    <div class="col-md-3">
                        <label class="form-label fw-bold small mb-1">Min Deliv %(7D)</label>
                        <input type="number" id="f_dp7" class="form-control form-control-sm" placeholder="e.g. 60">
                    </div>
                    <div class="col-md-3">
                        <label class="form-label fw-bold small mb-1">Min Deliv %(30D)</label>
                        <input type="number" id="f_dp30" class="form-control form-control-sm" placeholder="e.g. 50">
                    </div>
                </div>

                <div class="table-responsive">
                    <table id="screenerTable" class="table table-hover table-sm align-middle" style="width:100%">
                        <thead class="table-dark">
                            <tr>
                                <th>Symbol</th>
                                <th>Sector</th>
                                <th>Close (₹)</th>
                                <th>Avg D.Qty(7D)</th>
                                <th>Avg D.%(7D)</th>
                                <th>Avg D.Qty(30D)</th>
                                <th>Avg D.%(30D)</th>
                                <th title="Statistical deviation of Price">Z-Price</th>
                                <th title="Statistical deviation of Delivery Qty">Z-Deliv</th>
                            </tr>
                        </thead>
                        <tbody></tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="col-lg-3">
            <div class="card p-3 mb-3 sticky-top" style="top: 20px;">
                <div class="border-bottom fw-bold pb-2 mb-3">Search & Select Stock</div>
                <select id="stockFilter" class="form-select mb-4"></select>
                
                <div class="border-bottom fw-bold pb-2 mb-2 mt-2">📊 Cash Correlation</div>
                <p class="small text-muted mb-3">Does Delivery Qty drive Price Return?</p>
                <ul class="list-group list-group-flush small">
                  <li class="list-group-item d-flex justify-content-between px-0 bg-transparent">Last 30 Days<span id="corr_30" class="badge bg-light text-dark">-</span></li>
                  <li class="list-group-item d-flex justify-content-between px-0 bg-transparent">Last 90 Days<span id="corr_90" class="badge bg-light text-dark">-</span></li>
                  <li class="list-group-item d-flex justify-content-between px-0 bg-transparent">Full History<span id="corr_1y" class="badge bg-light text-dark">-</span></li>
                </ul>
                
                <div class="alert alert-secondary mt-3 small p-2">
                    <strong>Z-Score Guide:</strong><br>
                    Calculates how far the current value is from the moving average.<br>
                    <span class="text-success fw-bold">> 2.0</span> = Extremely High Anomaly<br>
                    <span class="text-danger fw-bold">< -2.0</span> = Extremely Low Anomaly
                </div>

                <div class="border-bottom fw-bold pb-2 mb-2 mt-4">Stock Profile</div>
                <div id="profile_sector" class="fw-bold mb-1">-</div>
                <div class="small">Nifty 500: <span id="profile_nifty" class="fw-bold">-</span></div>
            </div>
        </div>
    </div>
</div>

<script>
    const screenerData = {json.dumps(screener_json)};
    const chartData = {json.dumps(chart_data)};
    
    let dataTable;
    let currentTabMode = 'all';

    function getPearson(x, y) {{
        let n = Math.min(x.length, y.length);
        if (n < 2) return 0;
        let sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0, sum_y2 = 0;
        for (let i = 0; i < n; i++) {{ sum_x += x[i]; sum_y += y[i]; sum_xy += (x[i] * y[i]); sum_x2 += (x[i] * x[i]); sum_y2 += (y[i] * y[i]); }}
        let step4 = Math.sqrt(((n * sum_x2) - (sum_x * sum_x)) * ((n * sum_y2) - (sum_y * sum_y)));
        return step4 === 0 ? 0 : ((n * sum_xy) - (sum_x * sum_y)) / step4;
    }}
    
    function calcCorr(data, days) {{
        if (!data || data.c.length < 5) return null;
        let d = Math.min(days, data.c.length - 1);
        let start = data.c.length - d - 1;
        let closes = data.c.slice(start), delivs = data.dq.slice(start), returns = [];
        for (let i = 1; i < closes.length; i++) {{ returns.push(closes[i-1] > 0 ? ((closes[i] - closes[i-1]) / closes[i-1]) * 100 : 0); }}
        return getPearson(delivs.slice(1), returns);
    }}

    function calcZ(arr) {{
        if (!arr || arr.length < 2) return 0;
        let sum = 0; for(let i=0; i<arr.length; i++) sum += arr[i];
        let mean = sum / arr.length;
        let sumSq = 0; for(let i=0; i<arr.length; i++) sumSq += Math.pow(arr[i] - mean, 2);
        let std = Math.sqrt(sumSq / (arr.length - 1));
        if (std === 0) return 0;
        return (arr[arr.length - 1] - mean) / std;
    }}

    function calculateAllZScores() {{
        const months = parseInt($('#zLookback').val()) || 6;
        const lookbackDays = months * 21;
        
        screenerData.forEach(row => {{
            const series = row.SERIES;
            const sym = row.SYMBOL;
            if (chartData[series] && chartData[series][sym]) {{
                const d = chartData[series][sym];
                const c_window = d.c.slice(-lookbackDays);
                const dq_window = d.dq.slice(-lookbackDays);
                row.z_p = calcZ(c_window);
                row.z_dq = calcZ(dq_window);
            }} else {{
                row.z_p = 0; row.z_dq = 0;
            }}
        }});
        
        if (dataTable) {{ dataTable.rows().invalidate().draw(false); }}
    }}

    function formatZScore(val) {{
        if (val === null || val === undefined) return '-';
        let num = val.toFixed(2);
        if (val >= 2.0) return `<span class="z-badge z-hot">${{num}}</span>`;
        if (val <= -2.0) return `<span class="z-badge z-cold">${{num}}</span>`;
        return `<span class="z-badge bg-light text-dark border">${{num}}</span>`;
    }}
    
    function formatCorr(v) {{
        if (v === null) return "N/A";
        let n = v.toFixed(2);
        return v > 0.2 ? `<span class="correlation-positive">${{n}}</span>` : (v < -0.2 ? `<span class="correlation-negative">${{n}}</span>` : `<span>${{n}}</span>`);
    }}

    $.fn.dataTable.ext.search.push(function(settings, data, dataIndex, rowData) {{
        if (currentTabMode === 'nifty500' && !rowData.IS_NIFTY500) return false;
        
        const seriesF = $('#seriesFilter').val();
        if (seriesF && rowData.SERIES !== seriesF) return false;
        const sec = $('#sectorFilter').val();
        if (sec && rowData.SECTOR !== sec) return false;

        const min_dq7 = parseFloat($('#f_dq7').val()) || 0;
        const min_dq30 = parseFloat($('#f_dq30').val()) || 0;
        const min_dp7 = parseFloat($('#f_dp7').val()) || 0;
        const min_dp30 = parseFloat($('#f_dp30').val()) || 0;

        if (rowData.avg_deliv_qty_7d < min_dq7) return false;
        if (rowData.avg_deliv_qty_30d < min_dq30) return false;
        if (rowData.avg_deliv_pct_7d < min_dp7) return false;
        if (rowData.avg_deliv_pct_30d < min_dp30) return false;

        return true;
    }});

    function initTable() {{
        dataTable = $('#screenerTable').DataTable({{
            data: screenerData,
            columns: [
                {{ data: 'SYMBOL' }},
                {{ data: 'SECTOR' }},
                {{ data: 'latest_close', render: $.fn.dataTable.render.number(',', '.', 2) }},
                {{ data: 'avg_deliv_qty_7d', render: $.fn.dataTable.render.number(',', '.', 0) }},
                {{ data: 'avg_deliv_pct_7d', render: $.fn.dataTable.render.number(',', '.', 2, '', '%') }},
                {{ data: 'avg_deliv_qty_30d', render: $.fn.dataTable.render.number(',', '.', 0) }},
                {{ data: 'avg_deliv_pct_30d', render: $.fn.dataTable.render.number(',', '.', 2, '', '%') }},
                {{ data: 'z_p', render: function(d) {{ return formatZScore(d); }} }},
                {{ data: 'z_dq', render: function(d) {{ return formatZScore(d); }} }}
            ],
            pageLength: 10,
            order: [[8, 'desc']],
            dom: '<"row"<"col-sm-12 col-md-6"l><"col-sm-12 col-md-6"f>>rt<"row"<"col-sm-12 col-md-5"i><"col-sm-12 col-md-7"p>>'
        }});
        
        $('#screenerTable tbody').on('click', 'tr', function () {{
            let r = dataTable.row(this).data();
            if(r) {{ 
                setTimeout(() => {{ $('#stockFilter').val(r.SYMBOL).trigger('change'); }}, 50); 
                window.scrollTo({{top:0, behavior:'smooth'}}); 
            }}
        }});
    }}

    function populateDropdown() {{
        const sel = $('#stockFilter'); const prev = sel.val(); sel.empty();
        let syms = [...new Set(dataTable.rows({{search:'applied'}}).data().toArray().map(r => r.SYMBOL))].sort();
        if (syms.length === 0) sel.append(new Option("No match", ""));
        else {{ syms.forEach(s => sel.append(new Option(s, s))); if(syms.includes(prev)) sel.val(prev); }}
        sel.trigger('change');
    }}

    function updateDashboard() {{
        const series = $('#seriesFilter').val(), sym = $('#stockFilter').val();
        if (!series || !sym || !chartData[series] || !chartData[series][sym]) return;
        
        const data = chartData[series][sym];
        const r = screenerData.find(x => x.SYMBOL === sym && x.SERIES === series);
        
        if (r) {{
            $('#m_close').text('₹' + r.latest_close.toFixed(2));
            $('#m_dq7').text(r.avg_deliv_qty_7d.toLocaleString('en-IN', {{maximumFractionDigits:0}}));
            $('#m_d7').text(r.avg_deliv_pct_7d.toFixed(2) + '%');
            $('#m_zp').html(formatZScore(r.z_p));
            $('#m_zdq').html(formatZScore(r.z_dq));
            
            $('#profile_sector').text(r.SECTOR);
            $('#profile_nifty').html(r.IS_NIFTY500 ? '<span class="text-success fw-bold">Yes</span>' : '<span class="text-danger fw-bold">No</span>');
        }}
        
        $('#corr_30').html(formatCorr(calcCorr(data, 21))); 
        $('#corr_90').html(formatCorr(calcCorr(data, 63))); 
        $('#corr_1y').html(formatCorr(calcCorr(data, 252)));

        const isDark = document.documentElement.getAttribute('data-bs-theme') === 'dark';
        const tColor = isDark ? '#f8f9fa' : '#212529', gColor = isDark ? '#495057' : '#dee2e6';

        Plotly.newPlot('plotlyChart', [
            {{ x: data.d, y: data.c, type: 'scatter', mode: 'lines', name: 'Close Price', line: {{color: '#FF8C00', width: 2.5}}, yaxis: 'y1' }},
            {{ x: data.d, y: data.dq, type: 'bar', name: 'Delivery Qty', marker: {{color: 'rgba(46,139,87,0.6)'}}, yaxis: 'y2' }}
        ], {{
            title: {{ text: `${{sym}} - Price & Accumulation Trends`, font: {{color: tColor}} }}, 
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', margin: {{r:10, t:40, b:30, l:50}},
            hovermode: 'x unified', font: {{color: tColor}}, showlegend: false,
            xaxis: {{ showgrid: true, gridcolor: gColor }},
            yaxis: {{ title: 'Close (₹)', domain: [0.3, 1], gridcolor: gColor }},
            yaxis2: {{ title: 'Deliv Qty', domain: [0, 0.25], showgrid: false }}
        }}, {{responsive: true}});
    }}

    $(document).ready(function() {{
        calculateAllZScores();
        initTable();
        
        $('.nav-link').on('click', function (e) {{
            e.preventDefault();
            $('.nav-link').removeClass('active'); $(this).addClass('active');
            currentTabMode = $(this).data('mode');
            dataTable.draw(); 
            populateDropdown();
        }});

        $('#f_dq7, #f_dq30, #f_dp7, #f_dp30, #sectorFilter, #seriesFilter').on('input change', function() {{ 
            dataTable.draw(); populateDropdown(); 
        }});
        
        $('#zLookback').on('change', function() {{
            calculateAllZScores();
            populateDropdown();
        }});

        $('#resetFilters').click(function() {{ 
            $('#f_dq7, #f_dq30, #f_dp7, #f_dp30, #sectorFilter').val(''); 
            $('#zLookback').val('6'); calculateAllZScores();
            dataTable.draw(); populateDropdown(); 
        }});
        
        $('#stockFilter').on('change', updateDashboard);
        $('#themeToggle').on('change', function() {{ 
            document.documentElement.setAttribute('data-bs-theme', this.checked ? 'dark' : 'light'); 
            updateDashboard(); 
        }});
        
        if ($('#seriesFilter option[value="EQ"]').length > 0) $('#seriesFilter').val('EQ');
        dataTable.draw(); 
        populateDropdown();
    }});
</script>
</body>
</html>
"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as file:
        file.write(html_content)

    print(f"Successfully generated {OUTPUT_HTML}!\n")

# ==========================================
# MAIN EXECUTION FLOW
# ==========================================
if __name__ == "__main__":
    download_bhavcopies()
    sector_map, nifty500_symbols = fetch_nifty500()
    build_dashboard(sector_map, nifty500_symbols)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
