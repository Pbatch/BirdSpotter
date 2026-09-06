const sightingsTab = document.querySelector('#sightings-tab');
const roiTab = document.querySelector('#roi-tab');
const sightingsPanel = document.querySelector('#sightings-panel');
const roiPanel = document.querySelector('#roi-panel');
const canvas = document.querySelector('#roi-view');
const context = canvas.getContext('2d');
const apply = document.querySelector('#roi-apply');
const status = document.querySelector('#roi-status');
let frame = new Image();
let start = null;
let selection = null;
let dragging = false;
let loaded = false;

function selectTab(showRoi) {
  sightingsPanel.hidden = showRoi;
  roiPanel.hidden = !showRoi;
  sightingsTab.classList.toggle('active', !showRoi);
  roiTab.classList.toggle('active', showRoi);
  sightingsTab.setAttribute('aria-selected', String(!showRoi));
  roiTab.setAttribute('aria-selected', String(showRoi));
  if (showRoi) loadRoi();
}

async function loadRoi() {
  let response;
  try {
    response = await fetch('/roi.json', {cache: 'no-store'});
  } catch (error) {
    status.textContent = 'Camera preview is unavailable.';
    return;
  }
  if (!response.ok) {
    status.textContent = 'Camera preview is unavailable.';
    return;
  }
  const configured = await response.json();
  selection = null;
  if (configured.roi !== null) {
    const [left, top, right, bottom] = configured.roi;
    selection = {left, top, right, bottom};
  }
  start = null;
  dragging = false;
  apply.disabled = true;
  refreshFrame();
  if (!loaded) {
    loaded = true;
    setInterval(refreshFrame, 1000);
  }
}

function refreshFrame() {
  if (dragging) return;
  const next = new Image();
  next.onload = () => {
    frame = next;
    canvas.width = frame.naturalWidth;
    canvas.height = frame.naturalHeight;
    draw();
  };
  next.src = `/camera.jpg?t=${Date.now()}`;
}

function point(event) {
  const bounds = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(canvas.width,
      Math.round((event.clientX - bounds.left) * canvas.width / bounds.width))),
    y: Math.max(0, Math.min(canvas.height,
      Math.round((event.clientY - bounds.top) * canvas.height / bounds.height)))
  };
}

function squareSelection(anchor, current) {
  const directionX = current.x < anchor.x ? -1 : 1;
  const directionY = current.y < anchor.y ? -1 : 1;
  const horizontalLimit = directionX < 0 ? anchor.x : canvas.width - anchor.x;
  const verticalLimit = directionY < 0 ? anchor.y : canvas.height - anchor.y;
  const side = Math.min(
    Math.max(Math.abs(current.x - anchor.x), Math.abs(current.y - anchor.y)),
    horizontalLimit,
    verticalLimit
  );
  const end = {x: anchor.x + directionX * side, y: anchor.y + directionY * side};
  return {
    left: Math.min(anchor.x, end.x), top: Math.min(anchor.y, end.y),
    right: Math.max(anchor.x, end.x), bottom: Math.max(anchor.y, end.y)
  };
}

function draw() {
  context.drawImage(frame, 0, 0, canvas.width, canvas.height);
  if (!selection) {
    status.textContent = `${canvas.width} x ${canvas.height} camera frame; full frame active`;
    return;
  }
  context.fillStyle = 'rgba(0, 0, 0, .55)';
  context.fillRect(0, 0, canvas.width, selection.top);
  context.fillRect(0, selection.bottom, canvas.width, canvas.height - selection.bottom);
  context.fillRect(0, selection.top, selection.left, selection.bottom - selection.top);
  context.fillRect(selection.right, selection.top, canvas.width - selection.right,
                   selection.bottom - selection.top);
  context.strokeStyle = '#50dc50';
  context.lineWidth = Math.max(2, canvas.width / 500);
  context.strokeRect(selection.left, selection.top,
                     selection.right - selection.left, selection.bottom - selection.top);
  const width = selection.right - selection.left;
  const height = selection.bottom - selection.top;
  status.textContent = `ROI ${selection.left}, ${selection.top}, ` +
    `${selection.right}, ${selection.bottom} — ${width} x ${height}, ` +
    `aspect ${(width / height).toFixed(3)}`;
}

canvas.addEventListener('pointerdown', event => {
  dragging = true;
  start = point(event);
  selection = {left: start.x, top: start.y, right: start.x, bottom: start.y};
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove', event => {
  if (!dragging) return;
  selection = squareSelection(start, point(event));
  draw();
});
canvas.addEventListener('pointerup', event => {
  if (!dragging) return;
  selection = squareSelection(start, point(event));
  dragging = false;
  canvas.releasePointerCapture(event.pointerId);
  apply.disabled = selection.right <= selection.left || selection.bottom <= selection.top;
  draw();
});

async function saveRoi(value) {
  const response = await fetch('/roi', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(value)
  });
  if (!response.ok) throw new Error(await response.text());
  apply.disabled = true;
  status.textContent += ' — saved and active';
}

apply.addEventListener('click', async () => {
  try { await saveRoi(selection); } catch (error) { status.textContent = error; }
});
document.querySelector('#roi-reset').addEventListener('click', () => {
  selection = null;
  apply.disabled = true;
  draw();
});
document.querySelector('#roi-full').addEventListener('click', async () => {
  selection = null;
  try { await saveRoi(null); draw(); } catch (error) { status.textContent = error; }
});

async function refreshSightings() {
  if (document.hidden) return;
  try {
    const response = await fetch('/sightings.html', {cache: 'no-store'});
    if (!response.ok) return;
    document.querySelector('#sightings-grid').innerHTML = await response.text();
  } catch (error) {
    console.debug('Could not refresh sightings', error);
  }
}

setInterval(refreshSightings, 30000);
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) refreshSightings();
});
sightingsTab.addEventListener('click', () => selectTab(false));
roiTab.addEventListener('click', () => selectTab(true));
