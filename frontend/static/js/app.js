/* ═══════════════════════════════════════════════════════════════════════
   ViziGenesis — Main JavaScript (ES6+)
   Handles API calls, charting (Plotly), WebSocket feeds, & UI logic.
   ═══════════════════════════════════════════════════════════════════════ */

function inferApiBase() {
    const host = window.location.hostname;
    const port = window.location.port;
    const isLiveServer = port === "5500" || port === "5501" || window.location.pathname.includes("/frontend/");
    if (isLiveServer || window.location.protocol === "file:") {
        return `http://${host === '127.0.0.1' || host === 'localhost' ? '127.0.0.1' : host}:8000`;
    }
    return "";
}

const API = inferApiBase();

function getWsBase() {
    if (!API) {
        const proto = location.protocol === "https:" ? "wss" : "ws";
        return `${proto}://${location.host}`;
    }

    if (API.startsWith("https://")) return API.replace("https://", "wss://");
    if (API.startsWith("http://")) return API.replace("http://", "ws://");
    return API;
}

// ── Utility helpers ──────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const SESSION_CACHE_PREFIX = "vizigenesis:cache:";
const SESSION_CACHE_TTL_MS = {
    price: 20_000,
    watchlist: 5 * 60_000,
    ribbon: 90_000,
    history: 12 * 60_000,
    news: 6 * 60_000,
    prediction: 8 * 60_000,
    modelStatus: 90_000,
};

function saveActiveSymbol(symbol) {
    const sym = (symbol || "").toUpperCase().trim();
    if (!sym) return;
    try {
        sessionStorage.setItem("vizigenesis-active-symbol", sym);
        localStorage.setItem("vizigenesis-last-symbol", sym);
    } catch (_) {}
}

function getActiveSymbol() {
    try {
        return (
            (sessionStorage.getItem("vizigenesis-active-symbol") || "").toUpperCase().trim()
            || (localStorage.getItem("vizigenesis-last-symbol") || "").toUpperCase().trim()
            || ""
        );
    } catch (_) {
        return "";
    }
}

function cacheKey(type, symbol, extra = "") {
    const sym = (symbol || "").toUpperCase().trim();
    return `${SESSION_CACHE_PREFIX}${type}:${sym}:${extra}`;
}

function readCache(type, symbol, extra = "", ttlMs = 60_000) {
    try {
        const raw = sessionStorage.getItem(cacheKey(type, symbol, extra));
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed.ts !== "number") return null;
        if (Date.now() - parsed.ts > ttlMs) return null;
        return parsed.data ?? null;
    } catch (_) {
        return null;
    }
}

function readCacheAnyAge(type, symbol, extra = "") {
    try {
        const raw = sessionStorage.getItem(cacheKey(type, symbol, extra));
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        return parsed?.data ?? null;
    } catch (_) {
        return null;
    }
}

function writeCache(type, symbol, data, extra = "") {
    if (data === undefined || data === null) return;
    try {
        sessionStorage.setItem(cacheKey(type, symbol, extra), JSON.stringify({ ts: Date.now(), data }));
    } catch (_) {}
}

function getCssVar(name) {
    return getComputedStyle(document.body).getPropertyValue(name).trim();
}

function applyTheme(theme) {
    document.body.setAttribute("data-theme", theme);
    localStorage.setItem("vizigenesis-theme", theme);

    const icon = document.querySelector("#theme-toggle i");
    if (icon) {
        icon.className = theme === "dark" ? "fa-solid fa-sun" : "fa-solid fa-moon";
    }
}

function initThemeToggle() {
    const stored = localStorage.getItem("vizigenesis-theme");
    const initial = stored || "light";
    applyTheme(initial);

    const btn = document.getElementById("theme-toggle");
    if (!btn) return;
    btn.onclick = () => {
        const next = (document.body.getAttribute("data-theme") || "light") === "dark" ? "light" : "dark";
        applyTheme(next);
        if (activeDashboardSymbol) {
            renderMainChart(activeDashboardSymbol, activeHistoryData);
        }
    };
}

function initRouteLinks() {
    const isLiveServer = location.port === "5500" || location.port === "5501" || location.pathname.includes("/frontend/");
    document.querySelectorAll("a[data-route]").forEach((link) => {
        const route = link.dataset.route;
        const remembered = getActiveSymbol();
        const hash = remembered ? `#${encodeURIComponent(remembered)}` : "";
        if (route === "home") {
            link.href = (isLiveServer ? "home.html" : "/") + hash;
        } else if (route === "predict") {
            link.href = (isLiveServer ? "predict.html" : "/predict") + hash;
        }

        link.addEventListener("click", () => {
            const currentInput = (document.getElementById("sym-input")?.value || "").toUpperCase().trim();
            const hashSym = (location.hash || "").replace("#", "").toUpperCase().trim();
            const chosen = currentInput || hashSym || getActiveSymbol();
            if (!chosen) return;
            saveActiveSymbol(chosen);
        });
    });
}

function showToast(msg, duration = 3000) {
    let t = document.getElementById("toast");
    if (!t) {
        t = document.createElement("div");
        t.id = "toast";
        t.className = "toast";
        document.body.appendChild(t);
    }
    t.textContent = msg;
    t.classList.add("show");
    setTimeout(() => t.classList.remove("show"), duration);
}

function setLoading(el, on) {
    if (on) {
        el.dataset.origHtml = el.innerHTML;
        el.innerHTML = '<div class="spinner" style="width:20px;height:20px;margin:0;border-width:3px"></div>';
    } else if (el.dataset.origHtml !== undefined) {
        el.innerHTML = el.dataset.origHtml;
    }
}

function trendBadge(trend) {
    const map = {
        UP: "badge-up", DOWN: "badge-down", NEUTRAL: "badge-neutral",
        BULLISH: "badge-up", BEARISH: "badge-down",
    };
    const vi = {
        UP: "TĂNG",
        DOWN: "GIẢM",
        NEUTRAL: "TRUNG LẬP",
        BULLISH: "TĂNG DÀI HẠN",
        BEARISH: "GIẢM DÀI HẠN",
    };
    return `<span class="badge ${map[trend] || 'badge-neutral'}">${vi[trend] || trend}</span>`;
}

function formatNum(n) {
    if (n === null || n === undefined) return "—";
    if (n >= 1e12) return (n / 1e12).toFixed(2) + "T";
    if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
    return n.toLocaleString();
}

function isVietnamSymbol(symbol, exchange = "") {
    const sym = String(symbol || "").toUpperCase().trim();
    const ex = String(exchange || "").toUpperCase().trim();
    return sym.endsWith(".VN") || ["HOSE", "HNX", "UPCOM"].includes(ex);
}

function formatPriceBySymbol(value, symbol, exchange = "") {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
    const num = Number(value);
    if (isVietnamSymbol(symbol, exchange)) {
        return `${num.toLocaleString("vi-VN", { maximumFractionDigits: 2 })} ₫`;
    }
    return `$${num.toFixed(2)}`;
}

// ═══════════════════════════════════════════════════════════════════════
// Real-time price
// ═══════════════════════════════════════════════════════════════════════
let priceWs = null;
let candleChart = null;
let candleSeries = null;
let volumeSeries = null;
let ma20Series = null;
let ma50Series = null;
let activeDashboardSymbol = null;
let activeHistoryData = null;
let dashboardRequestId = 0;
let newsRefreshTimer = null;

const MARKET_RIBBON_SYMBOLS = ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "VIC.VN", "VNM.VN", "HPG.VN"];
const WATCHLIST_SYMBOLS = [
    ["NVDA", "NVIDIA"],
    ["AAPL", "Apple"],
    ["MSFT", "Microsoft"],
    ["AMZN", "Amazon"],
    ["GOOGL", "Alphabet"],
    ["META", "Meta"],
    ["TSLA", "Tesla"],
    ["AVGO", "Broadcom"],
    ["AMD", "Advanced Micro Devices"],
    ["NFLX", "Netflix"],
    ["ORCL", "Oracle"],
    ["ADBE", "Adobe"],
    ["CRM", "Salesforce"],
    ["INTC", "Intel"],
    ["CSCO", "Cisco"],
    ["QCOM", "Qualcomm"],
    ["TXN", "Texas Instruments"],
    ["JPM", "JPMorgan"],
    ["BAC", "Bank of America"],
    ["WMT", "Walmart"],
    ["DIS", "Disney"],
    ["KO", "Coca-Cola"],
    ["PEP", "PepsiCo"],
    ["NKE", "Nike"],
    ["XOM", "Exxon Mobil"],
    ["CVX", "Chevron"],
    ["VIC.VN", "Vingroup"],
    ["VNM.VN", "Vinamilk"],
    ["HPG.VN", "Hoa Phat"],
    ["FPT.VN", "FPT"],
    ["MWG.VN", "The Gioi Di Dong"],
    ["VCB.VN", "Vietcombank"],
    ["ACB.VN", "ACB"],
    ["SSI.VN", "SSI Securities"],
];

async function loadMarketRibbon() {
    const watchSymbols = WATCHLIST_SYMBOLS.map(([symbol]) => symbol.toUpperCase());
    const haveAllInWatch = MARKET_RIBBON_SYMBOLS.every((s) => watchSymbols.includes(String(s || "").toUpperCase()));
    if (haveAllInWatch) {
        return;
    }

    const el = document.getElementById("market-ribbon");
    const wrap = document.querySelector(".market-ribbon-wrap");
    if (!el) return;

    const batch = await fetchBulkQuotes(MARKET_RIBBON_SYMBOLS, { silent: true });
    const quoteMap = new Map(batch.map((item) => [String(item.symbol || "").toUpperCase(), item]));

    const chips = MARKET_RIBBON_SYMBOLS
        .map((symbol) => {
            const item = quoteMap.get(symbol.toUpperCase());
            if (!item) return null;
            const delta = item.prev_close ? item.price - item.prev_close : 0;
            const pct = item.prev_close ? ((delta / item.prev_close) * 100).toFixed(2) : "0.00";
            const cls = delta >= 0 ? "text-green" : "text-red";
            return `
                <div class="market-chip">
                    <div class="symbol">${item.symbol}</div>
                    <div class="price">${item.price}</div>
                    <div class="change ${cls}">${delta >= 0 ? '+' : ''}${delta.toFixed(2)} (${pct}%)</div>
                </div>`;
        })
        .filter(Boolean);

    if (chips.length > 0) {
        el.innerHTML = chips.join("");
        if (wrap) wrap.classList.remove("hidden");
    } else {
        el.innerHTML = "";
        if (wrap) wrap.classList.add("hidden");
    }
}

function renderMarketRibbonFromQuotes(items) {
    const el = document.getElementById("market-ribbon");
    const wrap = document.querySelector(".market-ribbon-wrap");
    if (!el) return;

    const quoteMap = new Map((items || []).map((item) => [String(item.symbol || "").toUpperCase(), item]));
    const chips = MARKET_RIBBON_SYMBOLS
        .map((symbol) => {
            const item = quoteMap.get(symbol.toUpperCase());
            if (!item) return null;
            const delta = item.prev_close ? item.price - item.prev_close : 0;
            const pct = item.prev_close ? ((delta / item.prev_close) * 100).toFixed(2) : "0.00";
            const cls = delta >= 0 ? "text-green" : "text-red";
            return `
                <div class="market-chip">
                    <div class="symbol">${item.symbol}</div>
                    <div class="price">${item.price}</div>
                    <div class="change ${cls}">${delta >= 0 ? '+' : ''}${delta.toFixed(2)} (${pct}%)</div>
                </div>`;
        })
        .filter(Boolean);

    if (chips.length > 0) {
        el.innerHTML = chips.join("");
        if (wrap) wrap.classList.remove("hidden");
    }
}

async function loadWatchlistPanel() {
    const el = document.getElementById("watchlist-panel");
    if (!el) return;

    const symbols = WATCHLIST_SYMBOLS.map(([symbol]) => symbol);
    const renderWatchlistRows = (items) => {
        const quoteMap = new Map((items || []).map((item) => [String(item.symbol || "").toUpperCase(), item]));
        const preparedRows = WATCHLIST_SYMBOLS
            .map((meta) => {
                const item = quoteMap.get(meta[0].toUpperCase());
                const price = Number(item?.price);
                const prevClose = Number(item?.prev_close);
                const hasPrice = Number.isFinite(price);
                const hasPrevClose = Number.isFinite(prevClose) && prevClose !== 0;
                const delta = hasPrice && hasPrevClose ? (price - prevClose) : null;
                const pct = delta !== null ? ((delta / prevClose) * 100).toFixed(2) : null;
                const priceClass = delta === null ? "text-muted" : (delta >= 0 ? "text-green" : "text-red");
                const priceLabel = hasPrice ? price : "—";
                const changeLabel = pct !== null ? `${delta >= 0 ? '+' : ''}${pct}%` : "--";
                return {
                    symbol: meta[0],
                    html: `
                        <button type="button" class="watchlist-row" data-symbol="${meta[0]}">
                            <div>
                                <div class="watchlist-symbol">${meta[0]}</div>
                                <div class="watchlist-name">${meta[1]}</div>
                            </div>
                            <div></div>
                            <div class="watchlist-price ${priceClass}">${priceLabel}<br><span style="font-size:.78rem">${changeLabel}</span></div>
                        </button>`,
                };
            })
            .filter(Boolean);

        const listId = `watchlist-list-${Date.now()}`;
        el.onscroll = null;
        el.innerHTML = `<div class="watchlist-list" id="${listId}">${preparedRows.map((r) => r.html).join("")}</div>`;
        bindWatchlistInteractions();
    };

    const stale = readCacheAnyAge("quote_batch", symbols.join(","), "");
    if (stale?.length) {
        renderWatchlistRows(stale);
        renderMarketRibbonFromQuotes(stale);
    }

    const batch = await fetchBulkQuotes(symbols, { silent: true });
    renderWatchlistRows(batch);
    renderMarketRibbonFromQuotes(batch);
}

async function fetchBulkQuotes(symbols, { silent = false } = {}) {
    const normalized = (symbols || []).map((s) => String(s || "").toUpperCase().trim()).filter(Boolean);
    if (!normalized.length) return [];

    const batchKey = normalized.join(",");
    const ttl = normalized.length >= 20 ? SESSION_CACHE_TTL_MS.watchlist : SESSION_CACHE_TTL_MS.ribbon;
    const cached = readCache("quote_batch", batchKey, "", ttl);
    if (cached?.length) return cached;

    const staleCached = readCacheAnyAge("quote_batch", batchKey, "");

    const requestBatch = async (timeoutMs = 7000) => {
        const controller = new AbortController();
        const timer = timeoutMs > 0 ? setTimeout(() => controller.abort(), timeoutMs) : null;
        try {
            const res = await fetch(`${API}/api/quotes?symbols=${encodeURIComponent(batchKey)}`, {
                signal: controller.signal,
            });
            if (!res.ok) throw new Error("Không lấy được watchlist");
            const payload = await res.json();
            return payload?.items || [];
        } finally {
            if (timer) clearTimeout(timer);
        }
    };

    try {
        let items = await requestBatch(7000);
        if (!items.length) {
            items = await requestBatch(0);
        }
        writeCache("quote_batch", batchKey, items);
        for (const item of items) {
            if (item?.symbol) writeCache("price", item.symbol, item);
        }
        return items;
    } catch (e) {
        if (staleCached?.length) return staleCached;
        if (!silent) showToast("Lỗi tải watchlist: " + e.message);
        return [];
    }
}

function bindWatchlistInteractions() {
    document.querySelectorAll(".watchlist-row[data-symbol]").forEach((row) => {
        row.addEventListener("click", () => {
            const symbol = (row.dataset.symbol || "").toUpperCase().trim();
            if (!symbol) return;

            const symInput = document.getElementById("sym-input");
            if (symInput) symInput.value = symbol;

            if (typeof loadDashboard === "function" && document.querySelector(".dashboard-section")) {
                loadDashboard(symbol);
            }
        });
    });
}

let symbolSuggestTimer = null;
let symbolSuggestAbort = null;

async function fetchSymbolSuggestions(query, limit = 12) {
    const res = await fetch(`${API}/api/symbols/search?q=${encodeURIComponent(query)}&limit=${limit}`);
    if (!res.ok) throw new Error("Không lấy được gợi ý mã");
    return await res.json();
}

function setupSymbolAutocomplete() {
    const input = document.getElementById("sym-input");
    const datalist = document.getElementById("symbol-suggest");
    if (!input || !datalist) return;

    const renderEmpty = () => { datalist.innerHTML = ""; };

    input.addEventListener("input", () => {
        const q = (input.value || "").trim();
        if (q.length < 1) {
            renderEmpty();
            return;
        }

        if (symbolSuggestTimer) clearTimeout(symbolSuggestTimer);
        symbolSuggestTimer = setTimeout(async () => {
            try {
                if (symbolSuggestAbort) symbolSuggestAbort.abort();
                symbolSuggestAbort = new AbortController();

                const payload = await fetch(`${API}/api/symbols/search?q=${encodeURIComponent(q)}&limit=10`, {
                    signal: symbolSuggestAbort.signal,
                }).then((r) => r.ok ? r.json() : { items: [] });

                const items = payload?.items || [];
                datalist.innerHTML = items.map((item) => {
                    const symbol = (item.symbol || "").toUpperCase();
                    const name = item.name || "";
                    const exchange = item.exchange ? ` (${item.exchange})` : "";
                    const label = `${symbol} — ${name}${exchange}`;
                    return `<option value="${symbol}" label="${label}"></option>`;
                }).join("");
            } catch (e) {
                if (e?.name !== "AbortError") renderEmpty();
            }
        }, 180);
    });
}

async function fetchNews(symbol, { force = false } = {}) {
    const sym = (symbol || "").toUpperCase().trim();
    if (!force) {
        const cached = readCache("news", sym, "8", SESSION_CACHE_TTL_MS.news);
        if (cached) return cached;
    }

    const res = await fetch(`${API}/api/news/${sym}?limit=8`);
    if (!res.ok) throw new Error("Không lấy được tin tức liên quan");
    const data = await res.json();
    writeCache("news", sym, data, "8");
    return data;
}

function formatDateTime(value) {
    if (!value) return "Mới cập nhật";
    const d = new Date(typeof value === "number" ? value * 1000 : value);
    if (Number.isNaN(d.getTime())) return "Mới cập nhật";
    return d.toLocaleString("vi-VN");
}

function renderNews(feed) {
    const el = document.getElementById("news-feed");
    if (!el) return;
    const items = feed?.items || [];
    if (!items.length) {
        el.innerHTML = '<p class="text-muted">Chưa có tin tức phù hợp cho mã này.</p>';
        return;
    }

    el.className = "news-feed";
    el.innerHTML = items.map((item) => `
        <article class="news-item">
            <h3><a href="${item.link || '#'}" target="_blank" rel="noopener noreferrer">${item.title || 'Tin tức thị trường'}</a></h3>
            <p>${item.summary || 'Cập nhật mới nhất liên quan đến mã đang theo dõi.'}</p>
            <div class="news-meta">
                <span>${item.publisher || 'Nguồn tin thị trường'}</span>
                <span>${formatDateTime(item.published_at)}</span>
            </div>
        </article>
    `).join("");
}

function renderInsight(priceData, pred) {
    const el = document.getElementById("insight-display");
    if (!el) return;
    if (!priceData && !pred) {
        el.innerHTML = '<p class="text-muted">Chưa có dữ liệu để kết luận.</p>';
        return;
    }

    const price = priceData?.price ?? pred?.current_price;
    const forecast = pred?.predicted_price;
    const delta = (price && forecast) ? (forecast - price) : null;
    const pct = (delta !== null && price) ? ((delta / price) * 100) : null;

    const lines = [
        pred?.next_day_trend ? `Xu hướng ngắn hạn: <b>${pred.next_day_trend}</b>.` : null,
        pred?.long_term_trend ? `Xu hướng 30 ngày: <b>${pred.long_term_trend}</b>.` : null,
        pct !== null ? `Biên độ AI dự kiến: <b class="${pct >= 0 ? 'text-green' : 'text-red'}">${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%</b>.` : null,
        priceData?.volume ? `Khối lượng gần nhất: <b>${formatNum(priceData.volume)}</b>.` : null,
        pred?.confidence ? `Độ tin cậy hiện tại: <b>${pred.confidence}%</b>.` : null,
    ].filter(Boolean);

    el.innerHTML = `<div class="insight-list">${lines.map((line) => `<div class="insight-item">${line}</div>`).join("")}</div>`;
}

async function fetchPrice(symbol, { silent = false, force = false } = {}) {
    const sym = (symbol || "").toUpperCase().trim();
    if (!force) {
        const cached = readCache("price", sym, "", SESSION_CACHE_TTL_MS.price);
        if (cached && cached.source !== "local_sample_csv") return cached;
    }

    try {
        const res = await fetch(`${API}/api/price/${sym}`);
        if (!res.ok) throw new Error("Không tìm thấy mã cổ phiếu");
        const data = await res.json();
        if (data?.source === "local_sample_csv") {
            throw new Error("Nguồn giá không phải realtime");
        }
        writeCache("price", sym, data);
        return data;
    } catch (e) {
        if (!silent) showToast("Lỗi lấy giá: " + e.message);
        return null;
    }
}

function renderPrice(data) {
    if (!data) return;
    const el = $("#price-display");
    if (!el) return;

    const change = data.prev_close ? data.price - data.prev_close : 0;
    const pct = data.prev_close ? ((change / data.prev_close) * 100).toFixed(2) : "0.00";
    const cls = change >= 0 ? "text-green" : "text-red";
    const arrow = change >= 0 ? "▲" : "▼";

    el.innerHTML = `
        <div class="stat">
            <div class="value ${cls}">${formatPriceBySymbol(data.price, data.symbol, data.exchange)}</div>
            <div class="label">${data.symbol}</div>
        </div>
        <div class="mt-1 ${cls}" style="font-size:1.1rem;font-weight:700;text-align:center">
            ${arrow} ${change >= 0 ? "+" : ""}${change.toFixed(2)} (${pct}%)
        </div>
        <table class="info-table mt-2">
            <tr><td>Mở cửa</td><td>${formatPriceBySymbol(data.open, data.symbol, data.exchange)}</td></tr>
            <tr><td>Cao nhất</td><td>${formatPriceBySymbol(data.high, data.symbol, data.exchange)}</td></tr>
            <tr><td>Thấp nhất</td><td>${formatPriceBySymbol(data.low, data.symbol, data.exchange)}</td></tr>
            <tr><td>Khối lượng</td><td>${formatNum(data.volume)}</td></tr>
            <tr><td>Vốn hóa</td><td>${formatNum(data.market_cap)}</td></tr>
        </table>
        <div class="text-muted mt-1" style="font-size:.72rem;text-align:center">
            Nguồn dữ liệu: ${data.source || "unknown"}
        </div>
        <div class="text-muted mt-1" style="font-size:.75rem;text-align:center">
            Cập nhật: ${new Date(data.timestamp).toLocaleTimeString()}
        </div>`;
}

function startPriceWs(symbol) {
    if (priceWs) priceWs.close();
    priceWs = new WebSocket(`${getWsBase()}/ws/price/${symbol}`);
    priceWs.onmessage = (e) => renderPrice(JSON.parse(e.data));
    priceWs.onerror = () => showToast("Lỗi WebSocket — chuyển sang polling");
    priceWs.onclose = () => { priceWs = null; };
}

// ═══════════════════════════════════════════════════════════════════════
// Historical candlestick chart (TradingView Lightweight Charts style)
// ═══════════════════════════════════════════════════════════════════════
async function fetchHistory(symbol, period = "1y") {
    const sym = (symbol || "").toUpperCase().trim();
    const cached = readCache("history", sym, period, SESSION_CACHE_TTL_MS.history);
    if (cached) return cached;

    const res = await fetch(`${API}/api/history/${sym}?period=${period}`);
    if (!res.ok) throw new Error("Không tìm thấy dữ liệu lịch sử");
    const data = await res.json();
    writeCache("history", sym, data, period);
    return data;
}

function calcSMA(rows, window) {
    const out = [];
    let sum = 0;
    for (let i = 0; i < rows.length; i++) {
        sum += rows[i].close;
        if (i >= window) sum -= rows[i - window].close;
        if (i >= window - 1) {
            out.push({ time: rows[i].date, value: +(sum / window).toFixed(2) });
        }
    }
    return out;
}

function toTradingViewSymbol(symbol) {
    const sym = (symbol || "").toUpperCase().trim();
    if (!sym) return "NASDAQ:AAPL";
    if (sym.includes(":")) return sym;

    if (sym.endsWith(".VN")) {
        const base = sym.replace(".VN", "");
        return `HOSE:${base}`;
    }

    return `NASDAQ:${sym}`;
}

function renderTradingViewWidget(symbol, containerId = "chart-candle") {
    const container = document.getElementById(containerId);
    if (!container) return false;
    if (!window.TradingView || typeof window.TradingView.widget !== "function") {
        return false;
    }

    if (isVietnamSymbol(symbol)) {
        return false;
    }

    try {
        container.innerHTML = "<div id=\"tv-main-chart\" style=\"width:100%;height:100%\"></div>";
        const theme = (document.body.getAttribute("data-theme") || "light") === "dark" ? "dark" : "light";

        new window.TradingView.widget({
            autosize: true,
            symbol: toTradingViewSymbol(symbol),
            interval: "D",
            timezone: "Asia/Ho_Chi_Minh",
            theme,
            style: "1",
            locale: "vi_VN",
            enable_publishing: false,
            hide_top_toolbar: false,
            allow_symbol_change: true,
            container_id: "tv-main-chart",
        });

        return true;
    } catch (error) {
        console.error("TradingView widget failed:", error);
        return false;
    }
}

function renderMainChart(symbol, historyData) {
    const usedTradingView = renderTradingViewWidget(symbol, "chart-candle");
    if (!usedTradingView && historyData) {
        renderCandlestick(historyData, "chart-candle");
    }
    return usedTradingView;
}

function renderCandlestickPlotly(rows, containerId = "chart-candle") {
    const trace = {
        x: rows.map((r) => r.date),
        open: rows.map((r) => r.open),
        high: rows.map((r) => r.high),
        low: rows.map((r) => r.low),
        close: rows.map((r) => r.close),
        type: "candlestick",
        increasing: { line: { color: getCssVar("--green") || "#22c55e" } },
        decreasing: { line: { color: getCssVar("--red") || "#ef4444" } },
    };
    const vol = {
        x: rows.map((r) => r.date),
        y: rows.map((r) => r.volume),
        type: "bar",
        marker: { color: "rgba(37,99,235,0.22)" },
        yaxis: "y2",
        name: "Volume",
    };
    const layout = {
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: getCssVar("--text-muted") || "#64748b", family: "Inter, sans-serif" },
        margin: { t: 20, r: 30, b: 35, l: 55 },
        xaxis: { rangeslider: { visible: false }, gridcolor: "rgba(148,163,184,0.12)" },
        yaxis: { title: "Giá", gridcolor: "rgba(148,163,184,0.12)", side: "left" },
        yaxis2: { overlaying: "y", side: "right", showgrid: false, title: "KL" },
        showlegend: false,
    };
    Plotly.newPlot(containerId, [trace, vol], layout, { responsive: true });
}

function renderCandlestick(data, containerId = "chart-candle") {
    const rows = data.data || [];
    const container = document.getElementById(containerId);
    if (!container) return;

    if (!rows.length) {
        container.innerHTML = '<p class="text-muted text-center">Không có dữ liệu</p>';
        return;
    }

    try {
        container.innerHTML = "";
        if (candleChart) {
            candleChart.remove();
            candleChart = null;
        }

    const upColor = getCssVar("--green") || "#22c55e";
    const downColor = getCssVar("--red") || "#ef4444";
    const textColor = getCssVar("--text-muted") || "#94a3b8";
    const borderColor = getCssVar("--border") || "rgba(148,163,184,0.25)";
    const bgColor = getCssVar("--bg-secondary") || "#111827";

        candleChart = LightweightCharts.createChart(container, {
        width: container.clientWidth || 900,
        height: Math.max(430, container.clientHeight || 430),
        layout: {
            background: { color: bgColor },
            textColor,
            fontFamily: "Inter, sans-serif",
        },
        grid: {
            vertLines: { color: "rgba(148,163,184,0.12)" },
            horzLines: { color: "rgba(148,163,184,0.12)" },
        },
        rightPriceScale: { borderColor },
        timeScale: { borderColor, timeVisible: true },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    });

        const chartRows = rows.map((r) => ({
        ...r,
        timeSec: Math.floor(new Date(`${r.date}T00:00:00Z`).getTime() / 1000),
        }));

    // Compatibility with multiple lightweight-charts versions
        if (typeof candleChart.addCandlestickSeries === "function") {
            candleSeries = candleChart.addCandlestickSeries({
            upColor,
            downColor,
            wickUpColor: upColor,
            wickDownColor: downColor,
            borderVisible: false,
        });
        volumeSeries = candleChart.addHistogramSeries({
            color: "rgba(0,212,255,0.28)",
            priceFormat: { type: "volume" },
            priceScaleId: "",
            scaleMargins: { top: 0.80, bottom: 0 },
        });
        ma20Series = candleChart.addLineSeries({ color: "#60a5fa", lineWidth: 1.5, title: "MA20" });
        ma50Series = candleChart.addLineSeries({ color: "#f59e0b", lineWidth: 1.5, title: "MA50" });
        } else {
            candleSeries = candleChart.addSeries(LightweightCharts.CandlestickSeries, {
            upColor,
            downColor,
            wickUpColor: upColor,
            wickDownColor: downColor,
            borderVisible: false,
        });
        volumeSeries = candleChart.addSeries(LightweightCharts.HistogramSeries, {
            color: "rgba(0,212,255,0.28)",
            priceFormat: { type: "volume" },
            priceScaleId: "",
            scaleMargins: { top: 0.80, bottom: 0 },
        });
        ma20Series = candleChart.addSeries(LightweightCharts.LineSeries, { color: "#60a5fa", lineWidth: 1.5, title: "MA20" });
        ma50Series = candleChart.addSeries(LightweightCharts.LineSeries, { color: "#f59e0b", lineWidth: 1.5, title: "MA50" });
        }

        candleSeries.setData(chartRows.map((r) => ({
        time: r.timeSec,
        open: r.open,
        high: r.high,
        low: r.low,
        close: r.close,
    })));

        volumeSeries.setData(chartRows.map((r) => ({
        time: r.timeSec,
        value: r.volume,
        color: r.close >= r.open ? "rgba(34,197,94,0.35)" : "rgba(239,68,68,0.35)",
    })));

        const sma20 = calcSMA(rows, 20).map((r) => ({ ...r, time: Math.floor(new Date(`${r.time}T00:00:00Z`).getTime() / 1000) }));
        const sma50 = calcSMA(rows, 50).map((r) => ({ ...r, time: Math.floor(new Date(`${r.time}T00:00:00Z`).getTime() / 1000) }));
        ma20Series.setData(sma20);
        ma50Series.setData(sma50);
        candleChart.timeScale().fitContent();

        window.addEventListener("resize", () => {
            if (candleChart && container) {
                candleChart.applyOptions({ width: container.clientWidth || 900 });
            }
        });
    } catch (error) {
        console.error("Lightweight chart failed, fallback to Plotly:", error);
        renderCandlestickPlotly(rows, containerId);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Predictions
// ═══════════════════════════════════════════════════════════════════════
async function fetchPrediction(symbol, autoTrain = false, mode = "simple") {
    const sym = (symbol || "").toUpperCase().trim();
    const m = (mode || "simple").toLowerCase();
    const cached = readCache("prediction", sym, m, SESSION_CACHE_TTL_MS.prediction);
    if (cached) return cached;

    const res = await fetch(`${API}/api/predict/${sym}?auto_train=${autoTrain ? "true" : "false"}&mode=${encodeURIComponent(m)}`);
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Dự đoán thất bại");
    }
    const data = await res.json();
    writeCache("prediction", sym, data, m);
    return data;
}

async function fetchModelStatus(symbol, mode = "simple") {
    const sym = (symbol || "").toUpperCase().trim();
    const m = (mode || "simple").toLowerCase();
    const cached = readCache("model_status", sym, m, SESSION_CACHE_TTL_MS.modelStatus);
    if (cached) return cached;

    const res = await fetch(`${API}/api/model-status/${sym}?mode=${encodeURIComponent(m)}`);
    if (!res.ok) throw new Error("Không lấy được trạng thái model");
    const data = await res.json();
    writeCache("model_status", sym, data, m);
    return data;
}

function renderPrediction(pred) {
    const el = $("#prediction-display");
    if (!el) return;

    const indicators = pred.technical_indicators || {};
    const indicatorItems = Object.entries(indicators)
        .filter(([, v]) => v !== null && v !== undefined)
        .map(([k, v]) => `<tr><td>${k}</td><td>${typeof v === "number" ? v.toLocaleString() : v}</td></tr>`)
        .join("");

    const isVn = isVietnamSymbol(pred.symbol);
    const predictedLabel = formatPriceBySymbol(pred.predicted_price, pred.symbol);

    el.innerHTML = `
        <div class="grid-2" style="gap:1rem">
            <div class="stat">
                <div class="label">Ngày kế tiếp</div>
                <div class="value" style="font-size:1.5rem">${trendBadge(pred.next_day_trend)}</div>
                <div class="text-muted mt-1">${predictedLabel}</div>
            </div>
            <div class="stat">
                <div class="label">Triển vọng 30 ngày</div>
                <div class="value" style="font-size:1.5rem">${trendBadge(pred.long_term_trend)}</div>
            </div>
        </div>
        <div class="mt-2">
            <div class="label" style="font-size:.8rem;color:var(--text-muted)">Độ tin cậy</div>
            <div class="progress-bar"><div class="fill" style="width:${pred.confidence}%"></div></div>
            <div style="text-align:right;font-size:.8rem;color:var(--accent)">${pred.confidence}%</div>
        </div>
        <div class="text-muted mt-1" style="font-size:.75rem">Chế độ AI: ${(pred.mode || "simple").toUpperCase()} | Thiết bị model: ${pred.model_device} | Tiền tệ: ${isVn ? "VND" : "USD"}</div>
        ${indicatorItems ? `<table class="info-table mt-2">${indicatorItems}</table>` : ""}`;

    // Long-term chart
    renderLongTermChart(pred);
}

function renderLongTermChart(pred) {
    const container = $("#chart-lt");
    if (!container) return;
    container.classList.remove("hidden");

    const days = pred.long_term_predictions.map((_, i) => `Ngày ${i + 1}`);
    const trace = {
        x: days,
        y: pred.long_term_predictions,
        type: "scatter",
        mode: "lines+markers",
        line: { color: "#00d4ff", width: 2.5 },
        marker: { size: 5, color: "#6366f1" },
        fill: "tozeroy",
        fillcolor: "rgba(0,212,255,0.08)",
    };
    const currentLine = {
        type: "line", x0: days[0], x1: days[days.length - 1],
        y0: pred.current_price, y1: pred.current_price,
        line: { color: "#facc15", width: 1.5, dash: "dash" },
    };
    const layout = {
        paper_bgcolor: "transparent", plot_bgcolor: "transparent",
        font: { color: "#94a3b8", family: "Inter, sans-serif" },
        margin: { t: 25, r: 20, b: 35, l: 60 },
        xaxis: { gridcolor: "rgba(148,163,184,0.08)" },
        yaxis: { title: `Giá dự đoán (${isVietnamSymbol(pred.symbol) ? "VND" : "USD"})`, gridcolor: "rgba(148,163,184,0.08)" },
        shapes: [currentLine],
        showlegend: false,
    };
    Plotly.newPlot("chart-lt", [trace], layout, { responsive: true });
}

// ═══════════════════════════════════════════════════════════════════════
// Training via WebSocket (with progress)
// ═══════════════════════════════════════════════════════════════════════
function trainModelWs(symbol, epochs, onProgress, onDone, mode = "simple") {
    const ws = new WebSocket(`${getWsBase()}/ws/train/${symbol}`);
    ws.onopen = () => ws.send(JSON.stringify({ epochs, mode }));
    ws.onmessage = (e) => {
        const d = JSON.parse(e.data);
        if (d.status === "done") {
            onDone(d);
            ws.close();
        } else if (d.status === "error") {
            showToast("Lỗi huấn luyện: " + d.detail);
            ws.close();
        } else {
            onProgress(d);
        }
    };
    ws.onerror = () => showToast("Lỗi WebSocket khi huấn luyện");
    return ws;
}

// ═══════════════════════════════════════════════════════════════════════
// Training via REST (fallback)
// ═══════════════════════════════════════════════════════════════════════
async function trainModelRest(symbol, epochs = 50, mode = "simple") {
    const res = await fetch(`${API}/api/train/${symbol}?epochs=${epochs}&mode=${encodeURIComponent(mode)}`, { method: "POST" });
    if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Huấn luyện thất bại");
    }
    return await res.json();
}

// ═══════════════════════════════════════════════════════════════════════
// Training loss chart
// ═══════════════════════════════════════════════════════════════════════
function renderTrainingChart(history, containerId = "chart-train") {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.classList.remove("hidden");

    const epochs = history.map((h) => h.epoch);
    const tLoss = { x: epochs, y: history.map((h) => h.train_loss), name: "Huấn luyện", type: "scatter", line: { color: "#00d4ff" } };
    const vLoss = { x: epochs, y: history.map((h) => h.val_loss), name: "Validation", type: "scatter", line: { color: "#c084fc" } };
    const layout = {
        paper_bgcolor: "transparent", plot_bgcolor: "transparent",
        font: { color: "#94a3b8", family: "Inter, sans-serif" },
        margin: { t: 25, r: 20, b: 35, l: 60 },
        xaxis: { title: "Epoch", gridcolor: "rgba(148,163,184,0.08)" },
        yaxis: { title: "Loss (MSE)", gridcolor: "rgba(148,163,184,0.08)" },
        legend: { x: 0.8, y: 1 },
    };
    Plotly.newPlot(containerId, [tLoss, vLoss], layout, { responsive: true });
}

// ═══════════════════════════════════════════════════════════════════════
// CSV download
// ═══════════════════════════════════════════════════════════════════════
function downloadCsv(symbol) {
    window.open(`${API}/api/download-csv/${symbol}`, "_blank");
}

// ═══════════════════════════════════════════════════════════════════════
// Home page orchestrator
// ═══════════════════════════════════════════════════════════════════════
async function loadDashboard(symbol) {
    if (!symbol) return;
    symbol = symbol.toUpperCase().trim();
    const reqId = ++dashboardRequestId;
    activeDashboardSymbol = symbol;
    saveActiveSymbol(symbol);
    try {
        if (location.hash !== `#${symbol}`) {
            history.replaceState(null, "", `#${symbol}`);
        }
    } catch (_) {}
    const symInput = $("#sym-input");
    if (symInput) symInput.value = symbol;

    if (newsRefreshTimer) {
        clearInterval(newsRefreshTimer);
        newsRefreshTimer = null;
    }

    // Show sections
    document.querySelectorAll(".dashboard-section").forEach((s) => s.classList.remove("hidden"));

    const priceCard = $("#price-display");
    const predCard = $("#prediction-display");
    const symbolTitle = document.getElementById("hero-symbol");
    if (symbolTitle) symbolTitle.textContent = symbol;

    const candleContainer = document.getElementById("chart-candle");
    const tvShownInstantly = renderMainChart(symbol, null);
    if (!tvShownInstantly && candleContainer) candleContainer.innerHTML = '<div class="spinner"></div>';

    const cachedPrice = readCache("price", symbol, "", SESSION_CACHE_TTL_MS.price);
    const cachedHistory = readCache("history", symbol, "6mo", SESSION_CACHE_TTL_MS.history);
    const cachedNews = readCache("news", symbol, "8", SESSION_CACHE_TTL_MS.news);
    let cachedPred = readCache("prediction", symbol, "simple", SESSION_CACHE_TTL_MS.prediction);
    if (cachedPred && cachedPrice?.price && cachedPred?.predicted_price) {
        const cp = Number(cachedPrice.price);
        const pp = Number(cachedPred.predicted_price);
        if (cp > 0 && pp > 0) {
            const ratio = Math.max(cp / pp, pp / cp);
            if (ratio > 4.0) {
                cachedPred = null;
            }
        }
    }

    if (cachedPrice) {
        renderPrice(cachedPrice);
        renderInsight(cachedPrice, cachedPred || null);
        startPriceWs(symbol);
    } else if (priceCard) {
        setLoading(priceCard, true);
    }

    if (cachedHistory) {
        activeHistoryData = cachedHistory;
        if (!tvShownInstantly) renderMainChart(symbol, cachedHistory);
    }

    if (cachedNews) renderNews(cachedNews);

    if (cachedPred && predCard) {
        renderPrediction(cachedPred);
    } else if (predCard) {
        predCard.innerHTML = `<p class="text-muted text-center">Đang dự đoán (nếu chưa có model sẽ tự huấn luyện lần đầu)...</p>`;
    }

    const [priceRes, histRes, newsRes] = await Promise.allSettled([
        cachedPrice ? Promise.resolve(cachedPrice) : fetchPrice(symbol),
        cachedHistory ? Promise.resolve(cachedHistory) : fetchHistory(symbol, "6mo"),
        cachedNews ? Promise.resolve(cachedNews) : fetchNews(symbol),
    ]);

    if (reqId !== dashboardRequestId) return;

    const priceData = priceRes.status === "fulfilled" ? priceRes.value : null;
    if (!cachedPrice && priceCard) setLoading(priceCard, false);
    if (priceData) {
        renderPrice(priceData);
        renderInsight(priceData, cachedPred || null);
        startPriceWs(symbol);
    } else if (priceCard) {
        priceCard.innerHTML = `
            <p class="text-muted text-center">Không lấy được giá trực tiếp cho <b>${symbol}</b>.</p>
            <p class="text-muted text-center" style="font-size:.82rem">Nguồn realtime (TradingView/Yahoo) đang lỗi hoặc bị chặn mạng. Thử lại sau hoặc đổi mã khác.</p>`;
        renderInsight(null, cachedPred || null);
    }

    if (histRes.status === "fulfilled") {
        activeHistoryData = histRes.value;
        try {
            if (!tvShownInstantly) renderMainChart(symbol, histRes.value);
        } catch (e) {
            console.error(e);
            if (candleContainer) candleContainer.innerHTML = `<p class="text-muted text-center">Lỗi hiển thị biểu đồ nến</p>`;
        }
    } else {
        activeHistoryData = null;
        const ok = renderMainChart(symbol, null);
        if (!ok && candleContainer) {
            candleContainer.innerHTML = `<p class="text-muted text-center">Không có dữ liệu</p>`;
        }
    }

    if (newsRes.status === "fulfilled") {
        renderNews(newsRes.value);
    } else {
        renderNews({ items: [] });
    }

    newsRefreshTimer = window.setInterval(async () => {
        if (!activeDashboardSymbol) return;
        const current = activeDashboardSymbol.toUpperCase().trim();
        if (!current) return;
        try {
            const freshNews = await fetchNews(current, { force: true });
            if (current === (activeDashboardSymbol || "").toUpperCase().trim()) {
                renderNews(freshNews);
            }
        } catch (_) {}
    }, 60_000);

    // Prediction từ web: nếu chưa có model sẽ auto-train trên máy/server
    if (predCard && !cachedPred) {
        try {
            const pred = await fetchPrediction(symbol, true, "simple");
            if (reqId !== dashboardRequestId) return;
            renderPrediction(pred);
            renderInsight(priceData, pred);
        } catch (e) {
            if (reqId !== dashboardRequestId) return;
            predCard.innerHTML = `<p class="text-muted text-center">${e.message}</p>`;
        }
    }
}

document.addEventListener("DOMContentLoaded", () => {
    initRouteLinks();
    setupSymbolAutocomplete();

    const scheduleMarketDataLoad = () => {
        loadWatchlistPanel().catch(() => {
            loadMarketRibbon().catch(() => {});
        });
    };

    if (typeof window.requestIdleCallback === "function") {
        window.requestIdleCallback(() => scheduleMarketDataLoad(), { timeout: 1200 });
    } else {
        setTimeout(scheduleMarketDataLoad, 200);
    }
});

window.addEventListener("beforeunload", () => {
    if (newsRefreshTimer) {
        clearInterval(newsRefreshTimer);
        newsRefreshTimer = null;
    }
});

// ═══════════════════════════════════════════════════════════════════════
// Predict page orchestrator
// ═══════════════════════════════════════════════════════════════════════
async function loadPredictPage(symbol, mode = "simple") {
    if (!symbol) return;
    symbol = symbol.toUpperCase().trim();
    const aiMode = (mode || "simple").toLowerCase();
    saveActiveSymbol(symbol);
    try {
        if (location.hash !== `#${symbol}`) {
            history.replaceState(null, "", `#${symbol}`);
        }
    } catch (_) {}

    document.querySelectorAll(".predict-section").forEach((s) => s.classList.remove("hidden"));

    // Model status
    const modelEl = $("#model-status");
    const cachedStatus = readCache("model_status", symbol, aiMode, SESSION_CACHE_TTL_MS.modelStatus);
    if (cachedStatus && modelEl) {
        const version = cachedStatus.latest_version?.version ? ` | version: ${cachedStatus.latest_version.version}` : "";
        modelEl.innerHTML = cachedStatus.trained
            ? `<span class="text-green">● Đã huấn luyện (${aiMode.toUpperCase()})</span> — ${cachedStatus.meta?.updated || ""} trên ${cachedStatus.meta?.device || "cpu"}${version}`
            : `<span class="text-yellow">● Chưa huấn luyện (${aiMode.toUpperCase()})</span>`;
    }

    try {
        const status = await fetchModelStatus(symbol, aiMode);
        const el = $("#model-status");
        if (el && status) {
            const version = status.latest_version?.version ? ` | version: ${status.latest_version.version}` : "";
            el.innerHTML = status.trained
                ? `<span class="text-green">● Đã huấn luyện (${aiMode.toUpperCase()})</span> — ${status.meta?.updated || ""} trên ${status.meta?.device || "cpu"}${version}`
                : `<span class="text-yellow">● Chưa huấn luyện (${aiMode.toUpperCase()})</span>`;
        }
    } catch (_) {}

    // Prediction
    const predCard = $("#predict-result");
    if (predCard) {
        const cachedPred = readCache("prediction", symbol, aiMode, SESSION_CACHE_TTL_MS.prediction);
        if (cachedPred) {
            renderPrediction(cachedPred);
            return;
        }

        setLoading(predCard, true);
        try {
            const pred = await fetchPrediction(symbol, false, aiMode);
            setLoading(predCard, false);
            renderPrediction(pred);
            const ltDiv = document.getElementById("chart-lt");
            if (ltDiv) renderLongTermChart(pred);
        } catch (e) {
            setLoading(predCard, false);
            predCard.innerHTML = `<p class="text-muted">${e.message}. Vào "Huấn luyện" để train trước, sau đó dự đoán sẽ hiện rất nhanh.</p>`;
        }
    }
}
