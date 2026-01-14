# 部署.py
import os
import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ====== 环境变量/默认路径 ======
MODEL_SCRIPT = os.getenv("MODEL_SCRIPT", r"D:\PythonFile\JCAI\data_ak\模型.py")
MODEL_PATH = os.getenv("FINAL_MODEL_PATH", "")
DATA_PATH = os.getenv("DATA_PATH", r"D:\PythonFile\JCAI\data_ak\outputs\dataquant_features.csv")
CURRENT_DATA_PATH = DATA_PATH  # 当前正在使用的数据源（默认=DATA_PATH）
DEFAULT_TOPK = int(os.getenv("TOPK", "10"))

# ====== 报告图路径（你给的那些 saved: ...）======
REPORT_DIR = os.getenv("REPORT_DIR", r"D:\PythonFile\JCAI\data_ak\report_outputs")
REPORT_FILES = [
    "equity_curve_gross_vs_net.png",
    "turnover_series.png",
    "leverage_series.png",
    "drawdown_net.png",
    "rolling_sharpe_60d.png",
    "rolling_sharpe_126d.png",
    "rolling_sharpe_252d.png",
]


def load_logic(py_path: str):
    p = Path(py_path)
    if not p.exists():
        raise FileNotFoundError(f"MODEL_SCRIPT not found: {py_path}")
    spec = importlib.util.spec_from_file_location("logic", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


logic = load_logic(MODEL_SCRIPT)
app = FastAPI(title="Predict API", version="1.0")


# ====== 缓存（用于 /plots 出图） ======
LAST_ALL_DF: Optional[pd.DataFrame] = None
LAST_TOP_DF: Optional[pd.DataFrame] = None
LAST_METRIC: Optional[str] = None
LAST_REQ_TOPK: int = DEFAULT_TOPK  # 记录用户输入的 topk（用于自适应展示）


# ====== UI 页面（多选ticker + 上传） ======
UI_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Predict API - UI</title>
  <style>
    :root{
      --bg:#0b1220;
      --txt:#e6eefc;
      --muted:#a9b7d0;
      --line:rgba(255,255,255,.08);
      --accent:#60a5fa;
      --accent2:#34d399;
      --warn:#fbbf24;
      --danger:#fb7185;
      --shadow: 0 10px 30px rgba(0,0,0,.35);
      --radius: 16px;
    }
    *{ box-sizing:border-box; }
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "PingFang SC","Microsoft YaHei",sans-serif;
      background: radial-gradient(1000px 500px at 20% 10%, rgba(96,165,250,.25), transparent 60%),
                  radial-gradient(900px 500px at 90% 40%, rgba(52,211,153,.18), transparent 55%),
                  var(--bg);
      color:var(--txt);
    }
    .wrap{ max-width:1180px; margin:0 auto; padding:20px; }
    .topbar{ display:flex; gap:14px; align-items:center; justify-content:space-between; margin-bottom:16px; }
    .title{ display:flex; align-items:center; gap:12px; }
    .logo{
      width:42px;height:42px;border-radius:14px;
      background: linear-gradient(135deg, rgba(96,165,250,.9), rgba(52,211,153,.8));
      box-shadow: var(--shadow);
    }
    h2{ margin:0; font-size:20px; letter-spacing:.2px; }
    .sub{ margin-top:2px; font-size:12px; color:var(--muted); }
    .badge{
      display:inline-flex; align-items:center; gap:8px;
      padding:8px 12px; border:1px solid var(--line); border-radius:999px;
      background: rgba(255,255,255,.03); box-shadow: 0 6px 18px rgba(0,0,0,.25);
      font-size:12px; color:var(--muted); white-space:nowrap;
    }
    .dot{ width:9px;height:9px;border-radius:50%; background:var(--warn);
      box-shadow: 0 0 0 3px rgba(251,191,36,.15); }
    .dot.ok{ background:var(--accent2); box-shadow: 0 0 0 3px rgba(52,211,153,.15); }
    .dot.bad{ background:var(--danger); box-shadow: 0 0 0 3px rgba(251,113,133,.15); }

    .grid{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }
    @media(max-width:980px){ .grid{ grid-template-columns:1fr; } }

    .card{
      background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
      border:1px solid var(--line); border-radius:var(--radius); box-shadow:var(--shadow); overflow:hidden;
    }
    .card .hd{
      padding:14px 16px; border-bottom:1px solid var(--line); background: rgba(255,255,255,.02);
      display:flex; align-items:center; justify-content:space-between; gap:12px;
    }
    .card .hd h3{ margin:0; font-size:14px; letter-spacing:.2px; color:var(--txt); }
    .card .bd{ padding:16px; }
    .hint{ color:var(--muted); font-size:12px; line-height:1.45; }
    
    .plotControls{
      margin-top:10px;
      padding:10px 12px;
      border:1px solid var(--line);
      border-radius:14px;
      background: rgba(0,0,0,.14);
    }
    .plotControls .row{
      display:grid;
      grid-template-columns: 120px 1fr 70px;
      gap:10px;
      align-items:center;
      margin:8px 0;
    }
    .plotControls .row label{
      margin:0;
      font-size:12px;
      color:var(--muted);
    }
    .plotControls .val{
      text-align:right;
      font-size:12px;
      color:var(--txt);
    }

    label{ font-size:12px; color:var(--muted); display:block; margin-bottom:6px; }
    select, input, textarea{
      width:100%; padding:10px 10px; border-radius:12px; border:1px solid var(--line);
      background: rgba(0,0,0,.22); color:var(--txt); outline:none; transition:.15s ease;
    }
    select:focus, input:focus, textarea:focus{
      border-color: rgba(96,165,250,.55); box-shadow: 0 0 0 4px rgba(96,165,250,.15);
    }
    textarea{
      height:240px; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size:12px; line-height:1.4; margin-top:10px;
    }

    .controls{ display:grid; grid-template-columns: 1.2fr 1fr 1fr .7fr; gap:10px; margin-top:10px; }
    @media(max-width:980px){ .controls{ grid-template-columns: 1fr 1fr; } }

    .controls2{ display:grid; grid-template-columns: .8fr .8fr; gap:10px; margin-top:10px; }

    .btns{ display:flex; gap:10px; flex-wrap:wrap; margin-top:12px; }
    .btn{
      border:1px solid var(--line); background: rgba(255,255,255,.04); color:var(--txt);
      padding:10px 12px; border-radius:12px; cursor:pointer; transition:.15s ease; font-size:13px;
    }
    .btn:hover{ transform: translateY(-1px); background: rgba(255,255,255,.06); }
    .btn.primary{
      border:none; background: linear-gradient(135deg, rgba(96,165,250,.95), rgba(52,211,153,.85));
      color:#04101a; font-weight:700;
    }
    .btn.ghost{ background: transparent; }

    .kv{
      display:flex; justify-content:space-between; align-items:center; padding:10px 12px;
      border:1px solid var(--line); border-radius:12px; background: rgba(0,0,0,.18);
      margin:8px 0; gap:12px;
    }
    .kv b{ font-size:12px; color:var(--muted); font-weight:600; min-width:70px; }
    .kv span{ font-size:13px; color:var(--txt); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:72%; text-align:right; }

    .footer{ margin-top:14px; font-size:12px; color:var(--muted); display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; }
    code{ color:#c7d2fe; }
    .note{
      margin-top:10px; padding:10px 12px; border:1px dashed rgba(255,255,255,.14);
      border-radius:12px; background: rgba(0,0,0,.14);
    }
    a{ color:#8ab4ff; }

    /* 报告图：两列卡片 */
    .plotsGrid{
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:12px;
      margin-top:10px;
    }
    @media(max-width:980px){ .plotsGrid{ grid-template-columns:1fr; } }

    .plotCard{
      border:1px solid var(--line);
      border-radius:14px;
      background: rgba(0,0,0,.16);
      overflow:hidden;
      box-shadow: 0 10px 26px rgba(0,0,0,.22);
    }
    .plotCard .cap{
      padding:10px 12px;
      font-size:12px;
      color:var(--muted);
      border-bottom:1px solid var(--line);
      background: rgba(255,255,255,.02);
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:10px;
    }
    
    .zoomBtn{
      border:1px solid var(--line);
      background: rgba(255,255,255,.06);
      color:var(--txt);
      padding:6px 10px;
      border-radius:10px;
      cursor:pointer;
      font-size:12px;
    }
    .zoomBtn:hover{ transform: translateY(-1px); background: rgba(255,255,255,.10); }
    
    /* 图片放大弹窗 */
    .imgModal{
      position:fixed;
      inset:0;
      background: rgba(0,0,0,.72);
      display:none;
      align-items:center;
      justify-content:center;
      z-index:9999;
      padding:18px;
    }
    .imgModal.show{ display:flex; }
    .imgModal .inner{
      max-width: 96vw;
      max-height: 92vh;
      border:1px solid var(--line);
      border-radius:14px;
      overflow:hidden;
      background: rgba(15,23,42,.95);
      box-shadow: var(--shadow);
    }
    .imgModal .bar{
      display:flex;
      justify-content:space-between;
      align-items:center;
      padding:10px 12px;
      border-bottom:1px solid var(--line);
      color:var(--muted);
      font-size:12px;
    }
    .imgModal .closeBtn{
      border:1px solid var(--line);
      background: rgba(255,255,255,.06);
      color:var(--txt);
      padding:6px 10px;
      border-radius:10px;
      cursor:pointer;
      font-size:12px;
    }
    .imgModal img{
      display:block;
      max-width: 96vw;
      max-height: 86vh;
    }

    .plotCard img{
      width:100%;
      display:block;
      background: rgba(255,255,255,.02);
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div class="title">
        <div class="logo"></div>
        <div>
          <h2>Predict API 交互界面</h2>
          <div class="sub">只输出关键数值 + 报告图（不再输出 Top 图）</div>
        </div>
      </div>
      <div class="badge" id="modelBadge">
        <span class="dot" id="modelDot"></span>
        <span id="modelText">loading...</span>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <div class="hd">
          <h3>数据选择 & 预测</h3>
          <span class="hint">先上传/刷新 ticker → 载入 → 预测</span>
        </div>
        <div class="bd">
          <div class="hint">当前数据源：<code id="dataPathHint"></code></div>

          <div class="note">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:10px; flex-wrap:wrap;">
              <div class="hint"><b>上传文件</b>（CSV 或 JSON）</div>
              <button class="btn" id="fmtBtn" onclick="toggleFormat()">显示格式</button>
            </div>

            <div id="fmtBox" style="display:none; margin-top:10px;">
              <div class="hint">- CSV：首行表头，至少包含 <code>ticker</code> 与 <code>trade_date</code>（或 date/time 类列名），其余为特征列</div>
              <div class="hint">- JSON：内容为 <code>[{...},{...}]</code> 的数组</div>

              <div class="hint" style="margin-top:10px;">CSV 示例（可直接复制保存为 .csv）：</div>
              <textarea readonly style="width:100%; height:160px; font-family:Consolas, Menlo, monospace; font-size:12px; padding:10px; border-radius:10px; border:1px solid #e5e7eb; background:#0b1220; color:#e5e7eb; outline:none;">
ticker,trade_date,ret_1_L1_z60,ret_5_L1_z60,ret_20_L1_z60,vol_10_L1_z60,vol_20_L1_z60,sma_5_L1_z60,sma_10_L1_z60,sma_20_L1_z60,ema_12_L1_z60,ema_26_L1_z60,ma_gap_5_20_L1_z60,ema_gap_12_26_L1_z60,macd_L1_z60,macd_signal_L1_z60,macd_hist_L1_z60,rsi_L1_z60,kdj_k_L1_z60,kdj_d_L1_z60,atr_L1_z60,pdi_L1_z60,mdi_L1_z60,adx_L1_z60,bb_mid_L1_z60,bb_up_L1_z60,bb_dn_L1_z60,bb_bw_L1_z60,bb_pb_L1_z60,obv_L1_z60,vpt_L1_z60,mfi_L1_z60,cci_L1_z60,vwap_win_L1_z60,z_close_20_L1_z60,z_turnover_20_L1_z60,range_hl_L1_z60,gap_oc_L1_z60,log_turn_L1_z60,rv_20_L1_z60,ret_1_L1_z120,ret_5_L1_z120,ret_20_L1_z120,vol_10_L1_z120,vol_20_L1_z120,sma_5_L1_z120,sma_10_L1_z120,sma_20_L1_z120,ema_12_L1_z120,ema_26_L1_z120,ma_gap_5_20_L1_z120,ema_gap_12_26_L1_z120,macd_L1_z120,macd_signal_L1_z120,macd_hist_L1_z120,rsi_L1_z120,kdj_k_L1_z120,kdj_d_L1_z120,atr_L1_z120,pdi_L1_z120,mdi_L1_z120,adx_L1_z120,bb_mid_L1_z120,bb_up_L1_z120,bb_dn_L1_z120,bb_bw_L1_z120,bb_pb_L1_z120,obv_L1_z120,vpt_L1_z120,mfi_L1_z120,cci_L1_z120,vwap_win_L1_z120,z_close_20_L1_z120,z_turnover_20_L1_z120,range_hl_L1_z120,gap_oc_L1_z120,log_turn_L1_z120,rv_20_L1_z120,y_ret_k,y_up_k,y_vol_k
300086.SZ,2024/2/6,-2.5456089,-2.253114756,-2.564340758,2.384608036,2.569283012,-2.80493338,-2.410834255,-1.98116161,-2.521150739,-2.259497513,-2.361042408,-3.315523377,-2.993628555,-2.570186726,-2.150927932,-1.2834278,-1.155675496,-0.925697052,2.42070736,0.05063176,1.908199399,0.006331205,-1.98116161,-0.232119299,-2.832066428,2.73947706,-1.318169632,-1.538927085,-3.203259168,-0.475763375,-1.74344895,-3.236061114,-1.318169632,0.144798749,2.633938891,-0.0372005,0.310056258,-0.05293136,-2.5456089,-2.253114756,-2.564340758,2.384608036,2.569283012,-2.80493338,-2.410834255,-1.98116161,-2.521150739,-2.259497513,-2.361042408,-3.315523377,-2.993628555,-2.570186726,-2.150927932,-1.2834278,-1.155675496,-0.925697052,2.42070736,0.05063176,1.908199399,0.006331205,-1.98116161,-0.232119299,-2.832066428,2.73947706,-1.318169632,-1.538927085,-3.203259168,-0.475763375,-1.74344895,-3.236061114,-1.318169632,0.144798749,2.633938891,-0.0372005,0.310056258,-0.05293136,-0.014354067,0,0.221813394
              </textarea>
            </div>

            <div style="margin-top:10px; display:flex; gap:10px; flex-wrap:wrap;">
              <input id="fileInput" type="file" accept=".csv,.json" />
              <button class="btn" onclick="uploadFile()">上传并切换数据源</button>
            </div>
            <div class="hint" id="uploadHint" style="margin-top:6px;"></div>
          </div>

          <div class="controls">
            <div>
              <label>ticker（可多选，按住 Ctrl/Shift）</label>
              <select id="ticker" multiple size="8"></select>
            </div>
            <div>
              <label>start</label>
              <input id="start" type="date" />
            </div>
            <div>
              <label>end</label>
              <input id="end" type="date" />
            </div>
            <div>
              <label>n（最多行）</label>
              <input id="nrows" type="number" value="500" min="1" />
            </div>
          </div>

          <div class="controls2">
            <div>
              <label>topk（请求K）</label>
              <input id="topk" type="number" value="3" min="1" />
            </div>
            <div>
              <label>gamma（可选）</label>
              <input id="gamma" type="number" step="0.1" placeholder="可选" />
            </div>
          </div>

          <textarea id="rows" placeholder="rows 会在“载入数据(当前数据源)”后自动填充..."></textarea>

          <div class="btns">
            <button class="btn ghost" onclick="loadTickers()">刷新Ticker</button>
            <button class="btn" onclick="loadSample()">载入数据(当前数据源)</button>
            <button class="btn primary" onclick="doPredict()">预测</button>
            <button class="btn" onclick="checkHealth()">/health</button>
          </div>

          <div class="footer">
            <div>接口：<a href="/docs" target="_blank">/docs</a> ｜ <a href="/health" target="_blank">/health</a> ｜ <a href="/predict" target="_blank">/predict</a></div>
            <div>访问：<code>http://127.0.0.1:8002/</code></div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="hd">
          <h3>结果（关键数值 + 报告图）</h3>
          <span class="hint" id="metricHint"></span>
        </div>
        <div class="bd">
            <div id="summary" class="hint">等待预测...</div>
            
            <div class="plotControls" id="plotControls" style="display:none;">
              <div class="row">
                <label>K（每天选前K）</label>
                <input id="plotK" type="range" min="1" max="50" step="1" value="3" />
                <div class="val" id="plotKVal">3</div>
              </div>
              <div class="row">
                <label>cost_rate（交易成本）</label>
                <input id="plotCost" type="range" min="0" max="0.01" step="0.0005" value="0.001" />
                <div class="val" id="plotCostVal">0.001</div>
              </div>
              <div class="row">
                <label>window（Sharpe窗口）</label>
                <input id="plotWin" type="range" min="20" max="252" step="1" value="60" />
                <div class="val" id="plotWinVal">60</div>
              </div>
              <div class="hint" style="margin-top:6px;">滑动后自动刷新右侧三张图（净值/回撤/滚动Sharpe）</div>
            </div>
            
            <div id="plotsContainer" class="plotsGrid"></div>

          <div id="plotsHint" class="hint" style="margin-top:10px;"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- 图片放大弹窗 -->
  <div class="imgModal" id="imgModal" onclick="closeImgModalIfBg(event)">
    <div class="inner">
      <div class="bar">
        <span id="imgModalTitle"></span>
        <button class="closeBtn" onclick="closeImgModal()">关闭</button>
      </div>
      <img id="imgModalImg" src="" alt="preview"/>
    </div>
  </div>

<script>
function esc(s){ return (''+s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;'); }

function openImgModal(title, dataUrl){
  const modal = document.getElementById('imgModal');
  document.getElementById('imgModalTitle').textContent = title || '';
  document.getElementById('imgModalImg').src = dataUrl || '';
  modal.classList.add('show');
}
function closeImgModal(){
  const modal = document.getElementById('imgModal');
  modal.classList.remove('show');
  document.getElementById('imgModalImg').src = '';
}
function closeImgModalIfBg(e){
  // 点到遮罩背景才关闭，点到内容不关
  if (e && e.target && e.target.id === 'imgModal') closeImgModal();
}

function toggleFormat(){
  const box = document.getElementById('fmtBox');
  const btn = document.getElementById('fmtBtn');
  const show = (box.style.display === 'none' || box.style.display === '');
  box.style.display = show ? 'block' : 'none';
  btn.textContent = show ? '关闭格式' : '显示格式';
}

function getSelectedTickers(){
  const sel = document.getElementById('ticker');
  const arr = [];
  for (const opt of sel.options){
    if (opt.selected) arr.push(opt.value);
  }
  return arr;
}

async function checkHealth() {
  try {
    const r = await fetch('/health');
    const j = await r.json();
    const dot = document.getElementById('modelDot');
    const text = document.getElementById('modelText');
    document.getElementById('dataPathHint').textContent = j.current_data_path || j.data_path || '';
    if (j.model_exists) {
      dot.className = 'dot ok';
      text.textContent = 'model ok';
    } else {
      dot.className = 'dot bad';
      text.textContent = 'model missing';
    }
  } catch (e) {
    document.getElementById('modelDot').className = 'dot bad';
    document.getElementById('modelText').textContent = 'health error';
  }
}

async function loadTickers() {
  const sel = document.getElementById('ticker');
  sel.innerHTML = '';
  try {
    const r = await fetch('/tickers?limit=2000');
    const j = await r.json();
    if (!j.ok) {
      sel.innerHTML = '<option value="">(no tickers)</option>';
      return;
    }
    const arr = j.tickers || [];
    if (arr.length === 0) {
      sel.innerHTML = '<option value="">(no tickers)</option>';
      return;
    }
    for (const t of arr) {
      const opt = document.createElement('option');
      opt.value = t;
      opt.textContent = t;
      sel.appendChild(opt);
    }
  } catch (e) {
    sel.innerHTML = '<option value="">(load tickers failed)</option>';
  }
}

async function uploadFile(){
  const f = document.getElementById('fileInput').files[0];
  const hint = document.getElementById('uploadHint');
  if (!f){
    hint.textContent = '请选择一个 .csv 或 .json 文件';
    return;
  }
  hint.textContent = '上传中...';

  const fd = new FormData();
  fd.append('file', f);

  try{
    const r = await fetch('/upload', { method:'POST', body: fd });
    const j = await r.json();
    if (!r.ok || !j.ok){
      hint.textContent = '上传失败：' + (j.detail || 'unknown');
      return;
    }
    hint.textContent = '上传成功，已切换数据源：' + (j.current_data_path || '');
    await checkHealth();
    await loadTickers();
  }catch(e){
    hint.textContent = '上传失败：' + e;
  }
}

async function loadSample() {
  const tickers = getSelectedTickers();
  const start = (document.getElementById('start').value || '').trim();
  const end = (document.getElementById('end').value || '').trim();
  const n = parseInt(document.getElementById('nrows').value || '500');

  let url = `/sample?n=${encodeURIComponent(n)}`;
  if (tickers.length>0) url += `&tickers=${encodeURIComponent(tickers.join(','))}`;
  if (start) url += `&start=${encodeURIComponent(start)}`;
  if (end) url += `&end=${encodeURIComponent(end)}`;

  document.getElementById('summary').innerHTML = '<div class="hint">加载数据中...</div>';
  try {
    const r = await fetch(url);
    const j = await r.json();
    if (!j.ok) {
      document.getElementById('summary').innerHTML = `<div class="hint">载入失败：${esc(j.detail || 'unknown')}</div>`;
      return;
    }
    document.getElementById('rows').value = JSON.stringify(j.rows, null, 2);

    document.getElementById('summary').innerHTML =
      `<div class="kv"><b>载入行数</b><span>${esc(j.n_rows)}</span></div>` +
      `<div class="kv"><b>tickers</b><span>${esc((j.tickers||[]).join(','))}</span></div>` +
      `<div class="kv"><b>时间范围</b><span>${esc(j.start || '')} ~ ${esc(j.end || '')}</span></div>` +
      `<div class="kv"><b>date_col</b><span>${esc(j.date_col || '')}</span></div>` +
      `<div class="kv"><b>ticker_col</b><span>${esc(j.ticker_col || '')}</span></div>`;
  } catch (e) {
    document.getElementById('summary').innerHTML = `<div class="hint">sample error: ${esc(e)}</div>`;
  }
}

async function doPredict() {
  let rowsText = document.getElementById('rows').value.trim();
  if (!rowsText) {
    document.getElementById('summary').innerHTML = '<div class="hint">请先点“载入数据(当前数据源)”获得 rows。</div>';
    return;
  }

  let rows;
  try {
    rows = JSON.parse(rowsText);
    if (!Array.isArray(rows)) throw new Error('rows 必须是 JSON 数组');
  } catch (e) {
    document.getElementById('summary').innerHTML = `<div class="hint">rows JSON 解析失败: ${esc(e)}</div>`;
    return;
  }

  const topk = parseInt(document.getElementById('topk').value || '3');
  const gammaText = (document.getElementById('gamma').value || '').trim();
  const payload = { rows: rows, topk: topk };
  if (gammaText !== '') payload.gamma = parseFloat(gammaText);

  document.getElementById('summary').innerHTML = '<div class="hint">预测中...</div>';
  try {
    const r = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const j = await r.json();
    if (!r.ok) {
      document.getElementById('summary').innerHTML = `<div class="hint">predict error: ${esc(j.detail || 'unknown')}</div>`;
      return;
    }

    const s = j.stats || {};
    document.getElementById('summary').innerHTML =
      `<div class="kv"><b>n_rows</b><span>${esc(j.n_rows)}</span></div>` +
      `<div class="kv"><b>metric</b><span>${esc(j.metric || '')}</span></div>` +
      `<div class="kv"><b>req_topk</b><span>${esc(j.req_topk ?? '')}</span></div>` +
      `<div class="kv"><b>actual_k</b><span>${esc(j.actual_k ?? '')}</span></div>` +
      `<div class="kv"><b>mean</b><span>${esc(s.mean ?? '')}</span></div>` +
      `<div class="kv"><b>std</b><span>${esc(s.std ?? '')}</span></div>` +
      `<div class="kv"><b>min</b><span>${esc(s.min ?? '')}</span></div>` +
      `<div class="kv"><b>p50</b><span>${esc(s.p50 ?? '')}</span></div>` +
      `<div class="kv"><b>max</b><span>${esc(s.max ?? '')}</span></div>`;

    await loadPlots();
  } catch (e) {
    document.getElementById('summary').innerHTML = `<div class="hint">predict error: ${esc(e)}</div>`;
  }
}

function syncPlotControlUI(){
  const k = document.getElementById('plotK').value;
  const cost = document.getElementById('plotCost').value;
  const win = document.getElementById('plotWin').value;
  document.getElementById('plotKVal').textContent = k;
  document.getElementById('plotCostVal').textContent = cost;
  document.getElementById('plotWinVal').textContent = win;
}

let _plotTimer = null;
function scheduleReloadPlots(){
  if (_plotTimer) clearTimeout(_plotTimer);
  _plotTimer = setTimeout(() => loadPlots(), 150);
}

async function loadPlots() {
  const box = document.getElementById('plotsContainer');
  const hint = document.getElementById('plotsHint');
  box.innerHTML = '';
  hint.textContent = '';

  const controls = document.getElementById('plotControls');
  controls.style.display = 'block';

  syncPlotControlUI();
  const k = encodeURIComponent(document.getElementById('plotK').value);
  const cost = encodeURIComponent(document.getElementById('plotCost').value);
  const win = encodeURIComponent(document.getElementById('plotWin').value);

  try {
    const r = await fetch(`/plots?k=${k}&cost_rate=${cost}&window=${win}`);
    const j = await r.json();
    if (!j.ok) {
      hint.textContent = j.detail || 'plots error';
      return;
    }

    document.getElementById('metricHint').textContent =
      `metric: ${j.metric} | k=${j.meta?.k} | cost=${j.meta?.cost_rate} | win=${j.meta?.window}`;

    const imgs = j.images || [];
    if (imgs.length === 0) {
      hint.textContent = '未找到可展示的报告图。';
      return;
    }

    for (const it of imgs) {
      const card = document.createElement('div');
      card.className = 'plotCard';

      const cap = document.createElement('div');
      cap.className = 'cap';

      const title = document.createElement('span');
      title.textContent = it.name || '';

      const btn = document.createElement('button');
      btn.className = 'zoomBtn';
      btn.textContent = '放大';
      const dataUrl = 'data:image/png;base64,' + it.png_base64;
      btn.onclick = () => openImgModal(it.name || '', dataUrl);

      cap.appendChild(title);
      cap.appendChild(btn);

      const img = document.createElement('img');
      img.src = dataUrl;
      img.onclick = () => openImgModal(it.name || '', dataUrl);

      card.appendChild(cap);
      card.appendChild(img);
      box.appendChild(card);
    }

    if (j.meta && j.meta.warning) {
      hint.textContent = j.meta.warning;
    }
  } catch (e) {
    hint.textContent = 'plots error: ' + e;
  }
}

// 绑定滑钮事件（页面加载就绑）
document.addEventListener('DOMContentLoaded', () => {
  const kEl = document.getElementById('plotK');
  const cEl = document.getElementById('plotCost');
  const wEl = document.getElementById('plotWin');
  if (kEl) kEl.oninput = () => { syncPlotControlUI(); scheduleReloadPlots(); };
  if (cEl) cEl.oninput = () => { syncPlotControlUI(); scheduleReloadPlots(); };
  if (wEl) wEl.oninput = () => { syncPlotControlUI(); scheduleReloadPlots(); };
});


checkHealth();
loadTickers();
</script>
</body>
</html>
"""


# ====== 请求体 ======
class PredictReq(BaseModel):
    rows: List[Dict[str, Any]]
    topk: int = DEFAULT_TOPK
    gamma: Optional[float] = None


# ====== 工具函数 ======
def _read_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DATA_PATH not found: {path}")
    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError("JSON must be a list like: [{...},{...}]")
        return pd.DataFrame(obj)
    return pd.read_csv(p)


def _pick_metric(all_df: pd.DataFrame) -> Optional[str]:
    if all_df is None or all_df.empty:
        return None
    prefer = [
        "integrated_score",
        "score", "pred_score", "prob", "proba", "confidence",
        "final_score", "rank_score", "risk_adjusted_score",
        "综合得分", "得分", "评分", "置信度", "概率"
    ]
    cols = list(all_df.columns)
    low = {c: str(c).lower() for c in cols}
    for p in prefer:
        for c in cols:
            if p == low[c] or p in low[c]:
                if pd.api.types.is_numeric_dtype(all_df[c]):
                    return c
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(all_df[c])]
    return num_cols[0] if num_cols else None


def _find_ticker_col(df: pd.DataFrame) -> Optional[str]:
    cand = ["ticker", "symbol", "code", "stock_code", "ts_code", "代码", "股票代码", "证券代码"]
    cols = list(df.columns)
    low = {c: str(c).lower() for c in cols}
    for c in cols:
        if low[c] in cand:
            return c
    for c in cols:
        name = low[c]
        if "ticker" in name or "symbol" in name or name.endswith("_code"):
            return c
    return None


def _find_date_col(df: pd.DataFrame) -> Optional[str]:
    cand = ["trade_date", "date", "datetime", "time", "timestamp", "交易日期", "日期", "时间"]
    cols = list(df.columns)
    low = {c: str(c).lower() for c in cols}
    for c in cols:
        if low[c] in cand:
            return c
    for c in cols:
        name = low[c]
        if "date" in name or "time" in name:
            return c
    return None


def _to_dt_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce", utc=False)


def _png_file_to_base64(png_path: Path) -> str:
    data = png_path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _fig_to_base64(fig) -> str:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    bio.seek(0)
    return base64.b64encode(bio.read()).decode("utf-8")

def _find_return_col(df: pd.DataFrame) -> Optional[str]:
    # 训练常见：y_ret_k；兼容列名有空格/BOM/大小写
    cand = ["y_ret_k", "y_ret", "target_ret", "future_ret", "ret_fwd", "label_ret"]

    cols = list(df.columns)
    norm_map = {}
    for c in cols:
        key = str(c).strip().lstrip("\ufeff").lower()
        norm_map[key] = c

    # 1) 先精确匹配候选
    for name in cand:
        key = name.lower()
        if key in norm_map:
            real = norm_map[key]
            s = pd.to_numeric(df[real], errors="coerce").dropna()
            if len(s) > 0:
                return real

    # 2) 再做兜底：列名里包含 y_ret
    for key, real in norm_map.items():
        if "y_ret" in key:
            s = pd.to_numeric(df[real], errors="coerce").dropna()
            if len(s) > 0:
                return real

    return None



def _compute_turnover(prev_syms: List[str], cur_syms: List[str]) -> float:
    a = set([str(x) for x in prev_syms if x is not None])
    b = set([str(x) for x in cur_syms if x is not None])
    if len(a) == 0 and len(b) == 0:
        return 0.0
    denom = max(1, min(len(a), len(b)))
    overlap = len(a.intersection(b))
    # 简单定义：1 - 重叠比例
    return float(max(0.0, 1.0 - overlap / denom))


def _rolling_sharpe(daily_ret: pd.Series, window: int) -> pd.Series:
    r = pd.to_numeric(daily_ret, errors="coerce")
    m = r.rolling(window).mean()
    s = r.rolling(window).std(ddof=1)
    out = (m / s) * (252 ** 0.5)
    return out



# ====== 路由 ======
@app.get("/", response_class=HTMLResponse)
def root():
    return UI_HTML


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "Predict API",
        "model_script": MODEL_SCRIPT,
        "data_path": DATA_PATH,
        "current_data_path": CURRENT_DATA_PATH,
        "model_path": MODEL_PATH,
        "model_exists": Path(MODEL_PATH).exists() if MODEL_PATH else False,
        "report_dir": REPORT_DIR,
        "docs": "/docs",
        "predict": "/predict",
        "sample": "/sample",
        "tickers": "/tickers",
        "plots": "/plots",
        "upload": "/upload",
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    上传 CSV 或 JSON（数组）：
    - CSV：包含 ticker + trade_date（或 date/time 类列）+ 若干特征列
    - JSON：[{...},{...}] 数组
    """
    global CURRENT_DATA_PATH, LAST_ALL_DF, LAST_TOP_DF, LAST_METRIC

    name = (file.filename or "").lower()
    if not (name.endswith(".csv") or name.endswith(".json")):
        raise HTTPException(status_code=400, detail="Only .csv or .json is allowed")

    up_dir = Path(__file__).resolve().parent / "uploads"
    up_dir.mkdir(parents=True, exist_ok=True)

    save_path = up_dir / file.filename
    content = await file.read()
    save_path.write_bytes(content)

    try:
        df = _read_data(str(save_path))
        tcol = _find_ticker_col(df)
        dcol = _find_date_col(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload file parse failed: {e}")

    CURRENT_DATA_PATH = str(save_path)
    LAST_ALL_DF, LAST_TOP_DF, LAST_METRIC = None, None, None

    return {
        "ok": True,
        "current_data_path": CURRENT_DATA_PATH,
        "ticker_col": tcol,
        "date_col": dcol,
        "columns_preview": list(df.columns)[:30],
    }


@app.get("/tickers")
def tickers(limit: int = 5000):
    global CURRENT_DATA_PATH
    try:
        df = _read_data(CURRENT_DATA_PATH)
        tcol = _find_ticker_col(df)
        if not tcol:
            return {"ok": True, "ticker_col": None, "tickers": []}
        vals = df[tcol].dropna().astype(str).unique().tolist()
        vals = sorted(vals)[:max(1, int(limit))]
        return {"ok": True, "ticker_col": tcol, "tickers": vals}
    except Exception as e:
        return {"ok": False, "detail": str(e)}


@app.get("/sample")
def sample(
    n: int = 500,
    tickers: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    global CURRENT_DATA_PATH
    try:
        df = _read_data(CURRENT_DATA_PATH)
        original_cols = list(df.columns)

        tcol = _find_ticker_col(df)
        dcol = _find_date_col(df)

        ticker_list = []
        if tickers:
            ticker_list = [x.strip() for x in tickers.split(",") if x.strip()]
            if tcol and ticker_list:
                df = df[df[tcol].astype(str).isin([str(x) for x in ticker_list])].copy()

        if dcol and (start or end):
            dt = _to_dt_series(df[dcol])
            df = df.assign(__dt__=dt)
            if start:
                sdt = pd.to_datetime(start, errors="coerce")
                if pd.notnull(sdt):
                    df = df[df["__dt__"] >= sdt]
            if end:
                edt = pd.to_datetime(end, errors="coerce")
                if pd.notnull(edt):
                    df = df[df["__dt__"] <= edt]
            df = df.drop(columns=["__dt__"], errors="ignore")

        df = df.head(max(1, int(n))).copy()

        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].astype(str)

        rows = df.where(pd.notnull(df), None).to_dict(orient="records")

        return {
            "ok": True,
            "current_data_path": CURRENT_DATA_PATH,
            "n_rows": int(len(rows)),
            "tickers": ticker_list,
            "start": start,
            "end": end,
            "ticker_col": tcol,
            "date_col": dcol,
            "columns": original_cols,
            "rows": rows,
        }
    except Exception as e:
        return {"ok": False, "detail": str(e)}


@app.post("/predict")
def predict(req: PredictReq):
    global LAST_ALL_DF, LAST_TOP_DF, LAST_METRIC, MODEL_PATH, LAST_REQ_TOPK

    if not MODEL_PATH:
        MODEL_PATH = r"D:\PythonFile\JCAI\data_ak\model_cache_v5_tune_best\final_model.joblib"

    if not MODEL_PATH or (not Path(MODEL_PATH).exists()):
        raise HTTPException(status_code=500, detail="FINAL_MODEL_PATH not found. Please generate final_model.joblib first.")

    try:
        pack = joblib.load(MODEL_PATH)
        df_in = pd.DataFrame(req.rows)

        all_df, top_df = logic.predict_with_final_model(pack, df_in, topk=req.topk, gamma=req.gamma)

        # 把输入里的收益/标签列补回（有些逻辑函数会丢掉 y_ret_k）
        ret_in = _find_return_col(df_in)
        if ret_in and (ret_in in df_in.columns) and (ret_in not in all_df.columns):
            all_df[ret_in] = df_in[ret_in].values

        LAST_ALL_DF = all_df.copy()
        LAST_METRIC = _pick_metric(LAST_ALL_DF)
        LAST_REQ_TOPK = int(req.topk)

        # --- 改为：每天 TopK（每个 trade_date 选分数最高的前 K 个 ticker）---
        tcol = _find_ticker_col(LAST_ALL_DF)
        dcol = _find_date_col(LAST_ALL_DF)
        metric = LAST_METRIC

        if tcol and dcol and metric and (metric in LAST_ALL_DF.columns):
            tmp = LAST_ALL_DF[[tcol, dcol, metric]].copy()
            tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
            tmp["__dt__"] = _to_dt_series(tmp[dcol])
            tmp = tmp.dropna(subset=[metric, "__dt__"])

            # 归一到“日期”（避免同一天不同时间导致分组错）
            tmp["__day__"] = tmp["__dt__"].dt.date

            # 同一天同一 ticker 如果有多条，取分数最高那条
            tmp = tmp.sort_values(metric, ascending=False).drop_duplicates(subset=["__day__", tcol], keep="first")

            k = max(1, int(req.topk))
            top_parts = []
            for day, sub in tmp.groupby("__day__", sort=True):
                sub = sub.sort_values(metric, ascending=False).head(k)
                top_parts.append(sub)

            if len(top_parts) > 0:
                top_day_df = pd.concat(top_parts, ignore_index=True)
                # 让输出更稳定：按日期升序、分数降序
                top_day_df = top_day_df.sort_values(["__day__", metric], ascending=[True, False]).copy()

                # 用原列名还原日期字段（保持给前端/下游一致）
                top_day_df[dcol] = top_day_df["__dt__"].astype(str)
                top_day_df = top_day_df.drop(columns=["__dt__", "__day__"], errors="ignore")

                LAST_TOP_DF = top_day_df.copy()
            else:
                LAST_TOP_DF = top_df.copy()
        else:
            # 兜底：如果缺少 ticker/date/metric，就用逻辑函数返回的 top_df
            LAST_TOP_DF = top_df.copy()

        metric = LAST_METRIC
        stats = {}
        if metric and metric in LAST_ALL_DF.columns:
            s = pd.to_numeric(LAST_ALL_DF[metric], errors="coerce").dropna()
            if len(s) > 0:
                stats = {
                    "count": int(len(s)),
                    "mean": float(s.mean()),
                    "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                    "min": float(s.min()),
                    "p25": float(s.quantile(0.25)),
                    "p50": float(s.quantile(0.50)),
                    "p75": float(s.quantile(0.75)),
                    "max": float(s.max()),
                }

        model_meta = {
            "version": pack.get("version"),
            "created_at": pack.get("created_at"),
            "use_garch_risk": pack.get("use_garch_risk"),
            "garch_blend": pack.get("garch_blend"),
        }

        actual_k = int(len(LAST_TOP_DF)) if LAST_TOP_DF is not None else 0

        return {
            "ok": True,
            "n_rows": int(len(all_df)),
            "metric": metric,
            "req_topk": int(req.topk),
            "actual_k": actual_k,
            "stats": stats,
            "model_meta": model_meta,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
@app.get("/plots")
def plots(
    k: int = 0,
    cost_rate: float = 0.001,
    window: int = 60,
):
    # 交互：通过 query params 动态生成 3 张图：净值、回撤、Rolling Sharpe
    if LAST_ALL_DF is None or LAST_ALL_DF.empty:
        return {"ok": False, "detail": "No cached prediction yet. Please call /predict first."}

    metric = (LAST_METRIC or "")
    tcol = _find_ticker_col(LAST_ALL_DF)
    dcol = _find_date_col(LAST_ALL_DF)
    if not tcol or not dcol:
        return {"ok": False, "detail": "Cannot find ticker/date columns for realtime report plotting."}

    ret_col = _find_return_col(LAST_ALL_DF)
    if not ret_col:
        return {"ok": False, "detail": "Cannot find a usable return/label column (e.g. y_ret_k) for portfolio report."}

    # 参数兜底/裁剪
    K = int(k) if int(k) > 0 else int(LAST_REQ_TOPK)
    K = max(1, min(200, K))
    cost_rate = float(cost_rate)
    cost_rate = max(0.0, min(0.05, cost_rate))
    window = int(window)
    window = max(5, min(400, window))

    # 准备全量数据（按“天”）
    use_cols = [tcol, dcol, ret_col]
    if metric and metric in LAST_ALL_DF.columns:
        use_cols.append(metric)

    tmp_all = LAST_ALL_DF[use_cols].copy()
    tmp_all[ret_col] = pd.to_numeric(tmp_all[ret_col], errors="coerce")
    tmp_all["__dt__"] = _to_dt_series(tmp_all[dcol])
    tmp_all = tmp_all.dropna(subset=[ret_col, "__dt__"])
    tmp_all["__day__"] = tmp_all["__dt__"].dt.date

    def _build_daily_pick_from_metric(k_: int) -> Dict[Any, List[str]]:
        if not (metric and (metric in tmp_all.columns)):
            return {}
        t = tmp_all[[tcol, "__day__", metric]].copy()
        t[metric] = pd.to_numeric(t[metric], errors="coerce")
        t = t.dropna(subset=[metric, "__day__"])
        t = t.sort_values(metric, ascending=False).drop_duplicates(subset=["__day__", tcol], keep="first")
        out = {}
        for day, sub in t.groupby("__day__", sort=True):
            sub = sub.sort_values(metric, ascending=False).head(max(1, int(k_)))
            out[day] = [str(x) for x in sub[tcol].dropna().astype(str).tolist()]
        return out

    def _portfolio_series_from_daily_pick(daily_pick: Dict[Any, List[str]]):
        days = sorted(list(daily_pick.keys()))
        if len(days) == 0:
            return None, None, None, None, None

        gross_ret = []
        turnover = []
        prev_syms = []

        for day in days:
            syms = daily_pick.get(day, [])
            sub_all_day = tmp_all[tmp_all["__day__"] == day]

            if len(syms) > 0:
                sub_p = sub_all_day[sub_all_day[tcol].astype(str).isin(syms)]
                pr = float(sub_p[ret_col].mean()) if len(sub_p) > 0 else 0.0
            else:
                pr = 0.0

            to = _compute_turnover(prev_syms, syms)
            prev_syms = syms

            gross_ret.append(pr)
            turnover.append(to)

        idx = pd.to_datetime([str(d) for d in days])
        s_gross = pd.Series(gross_ret, index=idx)
        s_turn = pd.Series(turnover, index=idx)

        s_net_ret = s_gross - (s_turn * cost_rate)
        eq_net = (1.0 + s_net_ret.fillna(0.0)).cumprod()
        roll_max = eq_net.cummax()
        dd_net = (eq_net / roll_max) - 1.0
        rs = _rolling_sharpe(s_net_ret, window)

        return s_net_ret, s_turn, eq_net, dd_net, rs

    warn = None
    daily_pick = _build_daily_pick_from_metric(K)

    # 没有 metric 时退化：用 LAST_TOP_DF（此时 K 不保证严格生效）
    if len(daily_pick) == 0:
        tmp_top = LAST_TOP_DF.copy() if (LAST_TOP_DF is not None and len(LAST_TOP_DF) > 0) else None
        if tmp_top is None or tmp_top.empty or (tcol not in tmp_top.columns) or (dcol not in tmp_top.columns):
            return {"ok": False, "detail": "No valid selection found to build daily portfolio."}
        tmp_top = tmp_top[[tcol, dcol]].copy()
        tmp_top["__dt__"] = _to_dt_series(tmp_top[dcol])
        tmp_top = tmp_top.dropna(subset=["__dt__"])
        tmp_top["__day__"] = tmp_top["__dt__"].dt.date
        daily_pick = tmp_top.groupby("__day__")[tcol].apply(
            lambda s: [str(x) for x in s.dropna().astype(str).tolist()]
        ).to_dict()
        warn = "metric column missing -> K slider may not take effect (fallback to LAST_TOP_DF)."

    s_net_ret, s_turn, eq_net, dd_net, rs = _portfolio_series_from_daily_pick(daily_pick)
    if s_net_ret is None:
        return {"ok": False, "detail": "No daily selections found to plot."}

    images = []

    def _format_xdate(fig, ax):
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.autofmt_xdate(rotation=30)

    # 1) equity (net)
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(eq_net.index, eq_net.values)
    ax.set_title(f"Equity Curve (Net) | K={K} cost={cost_rate:g}")
    ax.set_ylabel("equity")
    _format_xdate(fig, ax)
    images.append({"name": "equity_net.png", "png_base64": _fig_to_base64(fig)})

    # 2) drawdown
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(dd_net.index, dd_net.values)
    ax.set_title("Drawdown (Net)")
    ax.set_ylabel("drawdown")
    _format_xdate(fig, ax)
    images.append({"name": "drawdown_net.png", "png_base64": _fig_to_base64(fig)})

    # 3) rolling sharpe (window)
    fig = plt.figure()
    ax = plt.gca()
    ax.plot(rs.index, rs.values)
    ax.set_title(f"Rolling Sharpe ({window}d)")
    ax.set_ylabel("sharpe")
    _format_xdate(fig, ax)
    images.append({"name": f"rolling_sharpe_{window}d.png", "png_base64": _fig_to_base64(fig)})

    return {
        "ok": True,
        "metric": metric,
        "report_dir": "generated_in_memory",
        "images": images,
        "missing": [],
        "meta": {
            "return_col": ret_col,
            "cost_rate": cost_rate,
            "window": window,
            "n_days": int(len(s_net_ret)),
            "k": K,
            "warning": warn,
        },
    }

if __name__ == "__main__":
    import uvicorn

    if not MODEL_PATH:
        MODEL_PATH = r"D:\PythonFile\JCAI\data_ak\model_cache_v5_tune_best\final_model.joblib"

    uvicorn.run(app, host="127.0.0.1", port=8002)
