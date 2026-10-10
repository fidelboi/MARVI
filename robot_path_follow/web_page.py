"""HTML for the phone drawing page. __CAM_WIDTH__ / __CAM_HEIGHT__ are substituted at request time."""

PAGE = """
<!DOCTYPE html>

<html>

<head>

<meta name="viewport"
      content="width=device-width,
               initial-scale=1.0,
               maximum-scale=1.0,
               user-scalable=no">

<title>Robot Path Drawing</title>

<style>

html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
    background: black;
    overflow: hidden;
    overscroll-behavior: none;
}

/*
   The stage has EXACTLY the camera's aspect ratio and is centred.
   The video and the drawing canvas both fill the stage completely,
   so they are always pixel-aligned (no letterboxing to compensate for).
*/
#stage {
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);

    width: min(100vw, calc(100vh * __CAM_WIDTH__ / __CAM_HEIGHT__));
    width: min(100vw, calc(100dvh * __CAM_WIDTH__ / __CAM_HEIGHT__));
    aspect-ratio: __CAM_WIDTH__ / __CAM_HEIGHT__;
}

#video, #drawing {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    display: block;
}

#video {
    object-fit: fill;
    user-select: none;
    -webkit-user-drag: none;
    pointer-events: none;
}

#drawing {
    touch-action: none;
}

#controls {
    position: fixed;
    top: 15px;
    right: 15px;
    z-index: 10;
    display: flex;
    gap: 10px;
}

#controls button {
    padding: 12px 22px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    border-radius: 10px;
    background: white;
    color: black;
}

#goButton {
    background: #22c55e !important;
    color: white !important;
}

#controls button:disabled {
    opacity: 0.35;
}

#status {
    position: fixed;
    bottom: 15px;
    left: 15px;
    z-index: 10;
    color: white;
    background: rgba(0, 0, 0, 0.6);
    padding: 10px;
    border-radius: 8px;
    font-family: monospace;
}

</style>

</head>


<body>

<div id="stage">
    <img id="video" src="/video" draggable="false">
    <canvas id="drawing"></canvas>
</div>

<div id="controls">
    <button id="clearButton">CLEAR</button>
    <button id="goButton" disabled>GO</button>
</div>

<div id="status">Draw a path</div>


<script>

const CAM_WIDTH = __CAM_WIDTH__;
const CAM_HEIGHT = __CAM_HEIGHT__;

const canvas = document.getElementById("drawing");
const ctx = canvas.getContext("2d");

const statusEl = document.getElementById("status");
const goButton = document.getElementById("goButton");
const clearButton = document.getElementById("clearButton");

// The canvas' internal resolution IS the camera resolution.
// CSS scales it to fit the stage, so canvas pixels == camera pixels.
canvas.width = CAM_WIDTH;
canvas.height = CAM_HEIGHT;

const MIN_POINT_DISTANCE = 2;   // camera pixels

let path = [];                  // points in camera pixels
let activePointer = null;
let published = false;


// ============================================================
// SCREEN -> CAMERA PIXELS
// ============================================================

function getCameraPoint(event) {

    const rect = canvas.getBoundingClientRect();

    let x = (event.clientX - rect.left) / rect.width * CAM_WIDTH;
    let y = (event.clientY - rect.top) / rect.height * CAM_HEIGHT;

    x = Math.min(Math.max(x, 0), CAM_WIDTH);
    y = Math.min(Math.max(y, 0), CAM_HEIGHT);

    return { x: x, y: y };
}


// ============================================================
// DRAWING
// ============================================================

function strokeColor() {
    return published ? "#22c55e" : "red";
}

function setStyle() {
    ctx.strokeStyle = strokeColor();
    ctx.fillStyle = strokeColor();
    ctx.lineWidth = 4;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
}

function redraw() {

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (path.length === 0)
        return;

    setStyle();

    if (path.length === 1) {
        ctx.beginPath();
        ctx.arc(path[0].x, path[0].y, 3, 0, Math.PI * 2);
        ctx.fill();
        return;
    }

    ctx.beginPath();
    ctx.moveTo(path[0].x, path[0].y);

    for (let i = 1; i < path.length; i++)
        ctx.lineTo(path[i].x, path[i].y);

    ctx.stroke();
}

function updateButtons() {
    goButton.disabled = path.length < 2 || published;
}


// ============================================================
// POINTER EVENTS  (nothing is sent to the PC while drawing)
// ============================================================

canvas.addEventListener("pointerdown", function(event) {

    event.preventDefault();

    // Ignore extra fingers
    if (activePointer !== null)
        return;

    activePointer = event.pointerId;
    canvas.setPointerCapture(event.pointerId);

    // A new stroke replaces the previous, unpublished drawing
    path = [getCameraPoint(event)];
    published = false;

    redraw();
    updateButtons();

    statusEl.innerText = "Drawing...";
});


canvas.addEventListener("pointermove", function(event) {

    if (event.pointerId !== activePointer)
        return;

    event.preventDefault();

    const p = getCameraPoint(event);
    const last = path[path.length - 1];

    if (Math.hypot(p.x - last.x, p.y - last.y) < MIN_POINT_DISTANCE)
        return;

    path.push(p);

    setStyle();
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
});


function endStroke(event) {

    if (event.pointerId !== activePointer)
        return;

    event.preventDefault();

    if (canvas.hasPointerCapture(event.pointerId))
        canvas.releasePointerCapture(event.pointerId);

    activePointer = null;

    updateButtons();

    statusEl.innerText =
        path.length >= 2
            ? "Path ready (" + path.length + " points) - press GO"
            : "Path too short - draw again";
}

canvas.addEventListener("pointerup", endStroke);
canvas.addEventListener("pointercancel", endStroke);


// ============================================================
// GO  ->  publish path to ROS
// ============================================================

goButton.addEventListener("click", async function() {

    if (path.length < 2 || published)
        return;

    goButton.disabled = true;
    statusEl.innerText = "Publishing...";

    try {

        const response = await fetch("/go", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ points: path })
        });

        if (!response.ok)
            throw new Error("HTTP " + response.status);

        const result = await response.json();

        published = true;
        redraw();                       // turns the path green

        statusEl.innerText =
            "Published " + result.points + " points";

    } catch (err) {

        statusEl.innerText = "Publish failed: " + err.message;
        updateButtons();
    }
});


// ============================================================
// CLEAR
// ============================================================

clearButton.addEventListener("click", function() {

    path = [];
    published = false;

    redraw();
    updateButtons();

    fetch("/clear", { method: "POST" }).catch(function() {});

    statusEl.innerText = "Path cleared";
});

</script>

</body>

</html>
"""
