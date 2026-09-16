/**
 * IBVAP Command & Control Dashboard — app.js
 *
 * Responsibilities:
 *  - Live clock & date
 *  - Polling GET /api/events every POLL_INTERVAL_MS
 *  - Dedup via seen-IDs Set (only prepend genuinely new events)
 *  - Render alert cards with bounding-box overlay, confidence bar, snapshot
 *  - Filter (ALL / PERSON / VEHICLE)
 *  - Stats counters (total, persons, vehicles, avg confidence, threat level)
 *  - Polling progress bar animation
 *  - Toast notifications for new alerts
 *  - Lightbox for snapshot enlargement
 *  - API connection state management
 */

'use strict';

// ── Configuration ────────────────────────────────────────────────────────────
const API_BASE          = 'http://localhost:8000';
const EVENTS_ENDPOINT   = `${API_BASE}/api/events`;
const POLL_INTERVAL_MS  = 3000;
const MAX_CARDS         = 150;   // cap DOM cards to avoid memory bloat

// ── State ────────────────────────────────────────────────────────────────────
const seenIds      = new Set();
let   allEvents    = [];          // master list (newest first)
let   activeFilter = 'ALL';
let   pollTimer    = null;
let   countdownInt = null;
let   pollBarRaf   = null;
let   isConnected  = false;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const $clock          = document.getElementById('clock');
const $dateDisplay    = document.getElementById('date-display');
const $apiDot         = document.getElementById('api-dot');
const $apiStatusText  = document.getElementById('api-status-text');
const $apiStatusPill  = document.getElementById('api-status-pill');
const $threatLabel    = document.getElementById('threat-level-label');
const $threatSub      = document.getElementById('threat-sub');
const $tlBar1         = document.getElementById('tl-bar-1');
const $tlBar2         = document.getElementById('tl-bar-2');
const $tlBar3         = document.getElementById('tl-bar-3');
const $statTotal      = document.getElementById('stat-total');
const $statPersons    = document.getElementById('stat-persons');
const $statVehicles   = document.getElementById('stat-vehicles');
const $statLast       = document.getElementById('stat-last');
const $statAvgConf    = document.getElementById('stat-avg-conf');
const $pollCountdown  = document.getElementById('poll-countdown');
const $pollBar        = document.getElementById('poll-bar');
const $feedCount      = document.getElementById('feed-count');
const $emptyState     = document.getElementById('empty-state');
const $cardsGrid      = document.getElementById('cards-grid');
const $toastContainer = document.getElementById('toast-container');
const $lightbox       = document.getElementById('lightbox');
const $lightboxImg    = document.getElementById('lightbox-img');
const $lightboxCap    = document.getElementById('lightbox-caption');
const $lightboxClose  = document.getElementById('lightbox-close');
const $btnRefresh     = document.getElementById('btn-refresh');
const $btnClear       = document.getElementById('btn-clear');

// ── Live Clock ───────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  $clock.textContent = now.toLocaleTimeString('en-GB', { hour12: false });
  $dateDisplay.textContent = now.toLocaleDateString('en-GB', {
    weekday: 'short', year: 'numeric', month: 'short', day: '2-digit',
  }).toUpperCase();
}
updateClock();
setInterval(updateClock, 1000);

// ── API Status ───────────────────────────────────────────────────────────────
function setConnected(ok) {
  if (ok === isConnected) return;
  isConnected = ok;

  if (ok) {
    $apiDot.classList.remove('bg-red', 'conn-error');
    $apiDot.classList.add('bg-neon', 'live-dot');
    $apiStatusText.textContent = 'API ONLINE';
    $apiStatusPill.classList.remove('border-red/40', 'bg-red/5', 'text-red');
    $apiStatusPill.classList.add('border-neon/20', 'bg-neon/5', 'text-neon');
  } else {
    $apiDot.classList.remove('bg-neon', 'live-dot');
    $apiDot.classList.add('bg-red', 'conn-error');
    $apiStatusText.textContent = 'API OFFLINE';
    $apiStatusPill.classList.remove('border-neon/20', 'bg-neon/5', 'text-neon');
    $apiStatusPill.classList.add('border-red/40', 'bg-red/5', 'text-red');
  }
}

// ── Threat Level ─────────────────────────────────────────────────────────────
function updateThreatLevel(events) {
  // Count events in last 60 seconds
  const now = Date.now();
  const recent = events.filter(e => {
    const ts = new Date(e.timestamp).getTime();
    return (now - ts) < 60_000;
  });
  const rCount = recent.length;

  let level, colour, sub, bars;
  if (rCount === 0) {
    level = 'LOW';    colour = 'threat-low';    sub = 'System nominal';
    bars = [true, false, false];
  } else if (rCount <= 3) {
    level = 'MEDIUM'; colour = 'threat-medium'; sub = `${rCount} events / 60s`;
    bars = [true, true, false];
  } else {
    level = 'HIGH';   colour = 'threat-high';   sub = `${rCount} events / 60s — ALERT`;
    bars = [true, true, true];
  }

  $threatLabel.className = `font-display text-4xl font-black tracking-wider leading-none ${colour}`;
  $threatLabel.textContent = level;
  $threatSub.textContent = sub;

  const barColours = ['bg-neon/80', 'bg-amber/80', 'bg-red/80'];
  [$tlBar1, $tlBar2, $tlBar3].forEach((b, i) => {
    b.className = `h-1.5 w-8 rounded-full ${bars[i] ? barColours[i] : 'bg-border'}`;
  });
}

// ── Stats ─────────────────────────────────────────────────────────────────────
function updateStats(events) {
  const persons  = events.filter(e => e.event_type === 'PERSON_DETECTED').length;
  const vehicles = events.filter(e => e.event_type === 'VEHICLE_DETECTED').length;
  const avgConf  = events.length
    ? (events.reduce((s, e) => s + e.confidence, 0) / events.length * 100).toFixed(1)
    : null;

  $statTotal.textContent   = events.length;
  $statPersons.textContent = persons;
  $statVehicles.textContent= vehicles;
  $statAvgConf.textContent = avgConf ? `${avgConf}%` : '—';

  if (events.length > 0) {
    const last = events[0];
    const ts   = new Date(last.timestamp);
    $statLast.innerHTML =
      `<span class="text-text">${last.event_type.replace('_DETECTED','')}</span><br/>` +
      `<span class="text-subtext">${last.camera_id}</span><br/>` +
      `<span class="text-subtext">${ts.toLocaleTimeString('en-GB', {hour12:false})}</span>`;
  }

  updateThreatLevel(events);
}

// ── Card Builder ──────────────────────────────────────────────────────────────
function isPerson(ev)  { return ev.event_type === 'PERSON_DETECTED'; }
function isVehicle(ev) { return ev.event_type === 'VEHICLE_DETECTED'; }

function getEventColour(ev) {
  if (isPerson(ev))  return { border: 'border-neon',  badge: 'bg-neon/15 text-neon  border-neon/40',  bar: 'bg-neon',  flash: 'alert-flash' };
  if (isVehicle(ev)) return { border: 'border-amber', badge: 'bg-amber/15 text-amber border-amber/40', bar: 'bg-amber', flash: 'alert-flash-amber' };
  return               { border: 'border-red',   badge: 'bg-red/15  text-red   border-red/40',   bar: 'bg-red',   flash: 'alert-flash' };
}

function formatTimestamp(isoStr) {
  const d = new Date(isoStr);
  const date = d.toLocaleDateString('en-GB', { day:'2-digit', month:'short', year:'numeric' });
  const time = d.toLocaleTimeString('en-GB', { hour12: false });
  return { date, time, relative: timeAgo(d) };
}

function timeAgo(date) {
  const diff = Math.floor((Date.now() - date.getTime()) / 1000);
  if (diff < 5)   return 'just now';
  if (diff < 60)  return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
  return `${Math.floor(diff/3600)}h ago`;
}

function buildBBoxOverlay(bb, containerW, containerH) {
  if (!bb || containerW === 0) return '';
  // bb = {x, y, width, height} in pixel coords of original frame (assume 1280x720)
  const scaleX = 100 / 1280;
  const scaleY = 100 / 720;
  const left   = (bb.x * scaleX).toFixed(2);
  const top    = (bb.y * scaleY).toFixed(2);
  const w      = (bb.width  * scaleX).toFixed(2);
  const h      = (bb.height * scaleY).toFixed(2);
  return `<div class="bb-overlay border-neon/70"
               style="left:${left}%;top:${top}%;width:${w}%;height:${h}%;"></div>`;
}

function buildCard(ev, isNew = false) {
  const col  = getEventColour(ev);
  const ts   = formatTimestamp(ev.timestamp);
  const conf = Math.round(ev.confidence * 100);
  const label = ev.event_type.replace('_DETECTED', '');

  const snapHtml = ev.snapshot_url
    ? `<img src="${ev.snapshot_url}"
            alt="Snapshot ${ev.id}"
            loading="lazy"
            class="absolute inset-0 w-full h-full object-cover cursor-zoom-in"
            data-snap-url="${ev.snapshot_url}"
            data-snap-cap="${label} · ${ev.camera_id} · ${ts.time}"
            onerror="this.parentElement.innerHTML='<div class=\'absolute inset-0 flex items-center justify-center text-subtext font-mono text-xs\'>NO SNAPSHOT</div>'" />`
    : `<div class="absolute inset-0 flex items-center justify-center text-subtext font-mono text-xs tracking-wider">
         NO SNAPSHOT
       </div>`;

  const bbHtml = ev.bounding_box && ev.snapshot_url
    ? buildBBoxOverlay(ev.bounding_box, 1280, 720)
    : '';

  const card = document.createElement('div');
  card.id = `card-${ev.id}`;
  card.className = [
    'card-enter relative rounded border bg-card overflow-hidden shadow-card',
    `border-l-2 ${col.border}`,
    isNew ? col.flash : '',
  ].join(' ');

  card.innerHTML = `
    <!-- Snapshot -->
    <div class="snap-container">
      ${snapHtml}
      ${bbHtml}
      <!-- Camera ID badge -->
      <div class="absolute top-2 left-2 font-mono text-xs px-2 py-0.5 rounded-sm
                  bg-black/60 text-text/80 backdrop-blur-sm border border-white/10 tracking-wider">
        ${ev.camera_id}
      </div>
      <!-- Confidence badge -->
      <div class="absolute top-2 right-2 font-display text-xs px-2 py-0.5 rounded-sm
                  bg-black/70 backdrop-blur-sm tracking-wider ${col.badge.split(' ')[1]}">
        ${conf}%
      </div>
    </div>

    <!-- Card body -->
    <div class="p-3 space-y-2.5">

      <!-- Event type + time-ago -->
      <div class="flex items-center justify-between gap-2">
        <span class="font-mono text-xs px-2 py-0.5 rounded-sm border tracking-widest ${col.badge}">
          ${label}
        </span>
        <span class="font-mono text-subtext text-xs tracking-wider">${ts.relative}</span>
      </div>

      <!-- Confidence bar -->
      <div>
        <div class="flex justify-between mb-1">
          <span class="font-mono text-subtext text-xs tracking-wider">CONFIDENCE</span>
          <span class="font-mono text-xs tracking-wider ${col.badge.split(' ')[1]}">${conf}%</span>
        </div>
        <div class="w-full h-1.5 bg-border rounded-full conf-bar">
          <div class="h-full rounded-full ${col.bar} transition-all duration-700"
               style="width: ${conf}%"></div>
        </div>
      </div>

      <!-- Timestamp -->
      <div class="flex items-center justify-between pt-1 border-t border-border">
        <span class="font-mono text-subtext text-xs">${ts.time}</span>
        <span class="font-mono text-subtext/50 text-xs">${ts.date}</span>
      </div>

    </div>
  `;

  // Lightbox on image click
  const img = card.querySelector('img[data-snap-url]');
  if (img) {
    img.addEventListener('click', () => openLightbox(img.dataset.snapUrl, img.dataset.snapCap));
  }

  return card;
}

// ── Render / Filter ───────────────────────────────────────────────────────────
function applyFilter() {
  const cards = $cardsGrid.querySelectorAll('[id^="card-"]');
  let visible = 0;
  cards.forEach(card => {
    const eventId = card.id.replace('card-', '');
    const ev = allEvents.find(e => e.id === eventId);
    if (!ev) return;

    const show = activeFilter === 'ALL'
      || (activeFilter === 'PERSON'  && ev.event_type === 'PERSON_DETECTED')
      || (activeFilter === 'VEHICLE' && ev.event_type === 'VEHICLE_DETECTED');

    card.style.display = show ? '' : 'none';
    if (show) visible++;
  });
  $feedCount.textContent = visible;
}

// ── Polling ──────────────────────────────────────────────────────────────────
let pollStartTime = Date.now();

function animatePollBar() {
  const elapsed  = Date.now() - pollStartTime;
  const progress = Math.max(0, 1 - elapsed / POLL_INTERVAL_MS);
  $pollBar.style.width = `${progress * 100}%`;
  const secs = Math.ceil(progress * POLL_INTERVAL_MS / 1000);
  $pollCountdown.textContent = `${secs}s`;

  if (progress > 0) {
    pollBarRaf = requestAnimationFrame(animatePollBar);
  }
}

async function fetchEvents() {
  try {
    const resp = await fetch(EVENTS_ENDPOINT, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const events = await resp.json();
    setConnected(true);
    processNewEvents(events);
  } catch (err) {
    setConnected(false);
    console.warn('[IBVAP] Fetch failed:', err.message);
  } finally {
    // Reset poll bar
    pollStartTime = Date.now();
    cancelAnimationFrame(pollBarRaf);
    animatePollBar();
  }
}

function processNewEvents(events) {
  const newOnes = events.filter(e => !seenIds.has(e.id));
  if (newOnes.length === 0) return;

  newOnes.forEach(e => {
    seenIds.add(e.id);
    allEvents.unshift(e);
  });

  // Cap master list
  if (allEvents.length > MAX_CARDS) allEvents.splice(MAX_CARDS);

  // Show grid, hide empty state
  $emptyState.style.display = 'none';
  $cardsGrid.classList.remove('hidden');

  // Prepend new cards (newest at top)
  newOnes.forEach((ev, i) => {
    const card = buildCard(ev, true);
    if ($cardsGrid.firstChild) {
      $cardsGrid.insertBefore(card, $cardsGrid.firstChild);
    } else {
      $cardsGrid.appendChild(card);
    }

    // Staggered toast for first batch only (avoid spam)
    if (i < 3) {
      setTimeout(() => showToast(ev), i * 180);
    }
  });

  // Prune excess DOM cards
  const cards = $cardsGrid.querySelectorAll('[id^="card-"]');
  if (cards.length > MAX_CARDS) {
    for (let i = MAX_CARDS; i < cards.length; i++) {
      cards[i].remove();
    }
  }

  updateStats(allEvents);
  applyFilter();
}

// ── Toast ─────────────────────────────────────────────────────────────────────
function showToast(ev) {
  const col   = getEventColour(ev);
  const label = ev.event_type.replace('_DETECTED', '');
  const conf  = Math.round(ev.confidence * 100);

  const toast = document.createElement('div');
  toast.className = [
    'toast-enter pointer-events-auto',
    'flex items-center gap-3 px-4 py-3 rounded border shadow-card',
    'bg-card backdrop-blur-sm',
    `border-l-2 ${col.border}`,
  ].join(' ');

  toast.innerHTML = `
    <div class="font-mono text-xs ${col.badge.split(' ')[1]} tracking-widest">${label}</div>
    <div class="flex-1 min-w-0">
      <div class="font-mono text-text text-xs truncate">${ev.camera_id}</div>
      <div class="font-mono text-subtext text-xs">${conf}% confidence</div>
    </div>
    <div class="font-mono text-subtext text-xs">NEW</div>
  `;

  $toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.classList.remove('toast-enter');
    toast.classList.add('toast-exit');
    setTimeout(() => toast.remove(), 350);
  }, 3500);
}

// ── Lightbox ─────────────────────────────────────────────────────────────────
function openLightbox(url, caption) {
  $lightboxImg.src = url;
  $lightboxCap.textContent = caption || '';
  $lightbox.classList.remove('hidden');
  $lightbox.classList.add('flex');
}

function closeLightbox() {
  $lightbox.classList.add('hidden');
  $lightbox.classList.remove('flex');
  $lightboxImg.src = '';
}

$lightboxClose.addEventListener('click', closeLightbox);
$lightbox.addEventListener('click', e => { if (e.target === $lightbox) closeLightbox(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// ── Filter Buttons ────────────────────────────────────────────────────────────
function setFilter(f) {
  activeFilter = f;
  document.querySelectorAll('.filter-btn').forEach(b => {
    const isAmber = b.classList.contains('filter-btn-amber');
    b.classList.remove('active');
    if (isAmber) b.classList.remove('active');
  });

  const map = { ALL: 'filter-all', PERSON: 'filter-person', VEHICLE: 'filter-vehicle' };
  const btn = document.getElementById(map[f]);
  if (btn) btn.classList.add('active');
  applyFilter();
}

document.getElementById('filter-all').addEventListener('click',     () => setFilter('ALL'));
document.getElementById('filter-person').addEventListener('click',  () => setFilter('PERSON'));
document.getElementById('filter-vehicle').addEventListener('click', () => setFilter('VEHICLE'));

// ── Refresh & Clear ───────────────────────────────────────────────────────────
$btnRefresh.addEventListener('click', () => {
  clearInterval(pollTimer);
  fetchEvents().then(() => {
    pollTimer = setInterval(fetchEvents, POLL_INTERVAL_MS);
  });
});

$btnClear.addEventListener('click', () => {
  $cardsGrid.innerHTML = '';
  allEvents = [];
  seenIds.clear();
  $cardsGrid.classList.add('hidden');
  $emptyState.style.display = '';
  $feedCount.textContent = '0';
  updateStats([]);
});

// ── Relative time refresh ─────────────────────────────────────────────────────
// Update "X ago" labels every 30s without a full re-fetch
setInterval(() => {
  $cardsGrid.querySelectorAll('[id^="card-"]').forEach(card => {
    const ev = allEvents.find(e => `card-${e.id}` === card.id);
    if (!ev) return;
    const relEl = card.querySelector('.tracking-wider:last-child');
    // Find the time-ago span (second span in first row of body)
    const spans = card.querySelectorAll('.p-3 .flex:first-child span');
    if (spans.length >= 2) {
      spans[1].textContent = timeAgo(new Date(ev.timestamp));
    }
  });
}, 30_000);

// ── Start ────────────────────────────────────────────────────────────────────
pollStartTime = Date.now();
animatePollBar();
fetchEvents();  // immediate first fetch
pollTimer = setInterval(fetchEvents, POLL_INTERVAL_MS);

console.log(
  '%cIBVAP C&C Dashboard%c loaded — polling every %dms',
  'color:#00ff88;font-weight:bold;font-size:14px',
  'color:#6c7a95',
  POLL_INTERVAL_MS,
);
