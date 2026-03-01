// ================================================================
// Metaworld Web Teleoperation — Frontend Logic
// ================================================================
// Features: WebSocket, camera rotation, orientation mode,
//           gamepad/JoyCon, data visualization, rerun integration
// ================================================================

(function () {
    'use strict';

    // ---- DOM Elements ----
    const canvas = document.getElementById('renderCanvas');
    const ctx = canvas.getContext('2d');
    const container = document.getElementById('canvasContainer');

    // ---- WebSocket ----
    let ws = null;
    let reconnectTimer = null;
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const WS_URL = `${wsProtocol}//${window.location.host}/ws`;

    // ---- State ----
    let isRecording = false;
    let hasFocus = false;
    let inputMode = 'keyboard';
    let controlMode = 'camera'; // 'camera' or 'orientation'
    let gamepadIndex = null;
    let gamepadPollId = null;

    // ================================================================
    //  WebSocket Connection
    // ================================================================

    function connect() {
        if (ws && ws.readyState === WebSocket.OPEN) return;
        console.log('[Teleop] Connecting WebSocket:', WS_URL);
        ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log('[Teleop] WebSocket connected');
            document.getElementById('statusDot').classList.add('connected');
            document.getElementById('statusText').textContent = 'Connected';
            if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === 'frame') {
                    renderFrame(msg);
                    updateHUD(msg);
                }
            } catch (e) {
                console.error('[Teleop] Message error:', e);
            }
        };

        ws.onclose = (e) => {
            console.log('[Teleop] WebSocket closed:', e.code);
            document.getElementById('statusDot').classList.remove('connected');
            document.getElementById('statusText').textContent = 'Disconnected';
            if (!reconnectTimer) {
                reconnectTimer = setTimeout(() => { reconnectTimer = null; connect(); }, 2000);
            }
        };

        ws.onerror = (err) => {
            console.error('[Teleop] WebSocket error:', err);
            ws.close();
        };
    }

    function send(msg) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(msg));
        }
    }

    // ================================================================
    //  Frame Rendering
    // ================================================================

    const frameImg = new window.Image();
    let lastFrameW = 0, lastFrameH = 0;
    frameImg.onload = () => {
        // Only update canvas internal resolution when source size changes
        if (frameImg.width !== lastFrameW || frameImg.height !== lastFrameH) {
            canvas.width = frameImg.width;
            canvas.height = frameImg.height;
            lastFrameW = frameImg.width;
            lastFrameH = frameImg.height;
        }
        ctx.drawImage(frameImg, 0, 0);
    };

    function renderFrame(msg) {
        frameImg.src = 'data:image/jpeg;base64,' + msg.image;
    }

    // ================================================================
    //  HUD Update
    // ================================================================

    function updateHUD(msg) {
        document.getElementById('hudFps').textContent = msg.fps;
        document.getElementById('hudStep').textContent = msg.step;
        document.getElementById('hudEpisode').textContent = msg.episode;

        const rewardEl = document.getElementById('hudReward');
        rewardEl.textContent = msg.reward.toFixed(4);
        rewardEl.className = 'value ' + (msg.reward >= 0 ? 'positive' : 'negative');

        const successEl = document.getElementById('hudSuccess');
        successEl.textContent = msg.success ? '✓' : '—';
        successEl.className = 'value ' + (msg.success ? 'positive' : '');

        isRecording = msg.recording;
        document.getElementById('recBadge').classList.toggle('active', isRecording);
        document.getElementById('recEpisodes').textContent = msg.episodes_recorded + ' eps';
        document.getElementById('btnRecord').textContent = isRecording ? '⏹ Stop Recording' : '⏺ Start Recording';

        const fill = document.getElementById('gripperFill');
        fill.className = msg.gripper === 'closed' ? 'gripper-fill closed' : 'gripper-fill open';

        // Update control mode display from server state
        if (msg.control_mode && msg.control_mode !== controlMode) {
            controlMode = msg.control_mode;
            updateControlModeUI();
        }
    }

    // ================================================================
    //  Keyboard Input (WASDQE movement + IJKLUO camera/orientation + R + SPACE gripper)
    // ================================================================

    const movementKeys = new Set(['w', 'a', 's', 'd', 'q', 'e']);
    const secondaryKeys = new Set(['i', 'j', 'k', 'l', 'u', 'o']);
    const allTrackedKeys = new Set([...movementKeys, ...secondaryKeys]);
    const activeKeys = new Set();
    const CAM_KEY_RATE = 3.0;

    container.addEventListener('keydown', (e) => {
        if (inputMode !== 'keyboard') return;
        const key = e.key.toLowerCase();

        if (allTrackedKeys.has(key) && !activeKeys.has(key)) {
            activeKeys.add(key);
            if (movementKeys.has(key)) {
                send({ type: 'key_down', key: key });
            }
            updateKeyIndicators();
        }

        if (key === 'r') send({ type: 'reset' });
        if (e.code === 'Space') {
            send({ type: 'key_down', key: 'space' });
        }
        if (key === 'c') toggleControlMode(); // C to toggle camera/orientation
        if (allTrackedKeys.has(key) || key === 'r' || key === 'c' || e.code === 'Space') e.preventDefault();
    });

    container.addEventListener('keyup', (e) => {
        const key = e.key.toLowerCase();
        if (allTrackedKeys.has(key)) {
            activeKeys.delete(key);
            if (movementKeys.has(key)) {
                send({ type: 'key_up', key: key });
            }
            updateKeyIndicators();
        }
        if (e.code === 'Space') {
            send({ type: 'key_up', key: 'space' });
            e.preventDefault();
        }
    });

    // Secondary keys loop: camera rotation or orientation
    function secondaryKeyLoop() {
        if (controlMode === 'camera') {
            let dx = 0, dy = 0;
            if (activeKeys.has('j')) dx -= CAM_KEY_RATE;
            if (activeKeys.has('l')) dx += CAM_KEY_RATE;
            if (activeKeys.has('i')) dy -= CAM_KEY_RATE;
            if (activeKeys.has('k')) dy += CAM_KEY_RATE;
            if (activeKeys.has('u')) dx -= CAM_KEY_RATE * 1.5;
            if (activeKeys.has('o')) dx += CAM_KEY_RATE * 1.5;
            if (dx !== 0 || dy !== 0) {
                send({ type: 'camera_rotate', dx, dy });
            }
        } else {
            // Orientation mode: IJKLUO -> roll/pitch/yaw
            let roll = 0, pitch = 0, yaw = 0;
            if (activeKeys.has('j')) yaw -= 1;
            if (activeKeys.has('l')) yaw += 1;
            if (activeKeys.has('i')) pitch -= 1;
            if (activeKeys.has('k')) pitch += 1;
            if (activeKeys.has('u')) roll -= 1;
            if (activeKeys.has('o')) roll += 1;
            if (roll !== 0 || pitch !== 0 || yaw !== 0) {
                send({ type: 'orientation_input', roll, pitch, yaw });
            }
        }
        requestAnimationFrame(secondaryKeyLoop);
    }
    secondaryKeyLoop();

    function updateKeyIndicators() {
        ['q', 'w', 'e', 'a', 's', 'd', 'i', 'j', 'k', 'l', 'u', 'o'].forEach(k => {
            const el = document.getElementById('key' + k.toUpperCase());
            if (el) el.classList.toggle('active', activeKeys.has(k));
        });
    }

    // ================================================================
    //  Control Mode Toggle (Camera vs Orientation)
    // ================================================================

    function toggleControlMode() {
        controlMode = controlMode === 'camera' ? 'orientation' : 'camera';
        send({ type: 'set_control_mode', mode: controlMode });
        updateControlModeUI();
    }

    function updateControlModeUI() {
        const labelEl = document.getElementById('controlModeLabel');
        const btnEl = document.getElementById('btnToggleMode');
        if (labelEl) {
            labelEl.textContent = controlMode === 'camera'
                ? '📷 Camera Mode (IJKLUO → view)'
                : '🔧 Orientation Mode (IJKLUO → end-effector)';
        }
        if (btnEl) {
            btnEl.textContent = controlMode === 'camera'
                ? '🔧 Switch to Orientation'
                : '📷 Switch to Camera';
        }
    }

    // ================================================================
    //  Mouse Camera Rotation
    // ================================================================

    let isDragging = false;
    let lastMouseX = 0, lastMouseY = 0;

    container.addEventListener('mousedown', (e) => {
        if (e.button === 1 || (e.button === 2 && e.shiftKey)) {
            isDragging = true;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
            e.preventDefault();
            return;
        }
    });

    container.addEventListener('mousemove', (e) => {
        if (isDragging) {
            const dx = e.clientX - lastMouseX;
            const dy = e.clientY - lastMouseY;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
            send({ type: 'camera_rotate', dx, dy });
        }
    });

    container.addEventListener('mouseup', (e) => {
        if (e.button === 1 || (e.button === 2 && isDragging)) {
            isDragging = false;
        }
    });

    container.addEventListener('wheel', (e) => {
        e.preventDefault();
        send({ type: 'camera_zoom', delta: e.deltaY > 0 ? 1 : -1 });
    }, { passive: false });

    container.addEventListener('contextmenu', (e) => e.preventDefault());

    // ================================================================
    //  Focus Management
    // ================================================================

    const focusHint = document.getElementById('focusHint');

    container.addEventListener('focus', () => {
        hasFocus = true;
        focusHint.style.display = 'none';
    });

    container.addEventListener('blur', () => {
        hasFocus = false;
        isDragging = false;
        activeKeys.forEach(key => {
            if (movementKeys.has(key)) send({ type: 'key_up', key: key });
        });
        // Ensure gripper is released when focus is lost
        send({ type: 'key_up', key: 'space' });
        activeKeys.clear();
        updateKeyIndicators();
        focusHint.style.display = 'block';
    });

    focusHint.addEventListener('click', () => container.focus());
    focusHint.style.display = 'block';

    // ================================================================
    //  Gamepad / JoyCon (Web Gamepad API)
    // ================================================================

    // ================================================================
    //  Data Visualization
    // ================================================================

    // ----------- End Dataset UI -----------

    // ================================================================
    //  WebHID JoyCon Driver (Direct from Browser)
    //  Raw HID bytes are forwarded to backend for decoding
    // ================================================================
    let hidDevice = null;
    let hidPacketCounter = 0;
    let hidProductId = 0;  // Track which JoyCon (L=0x2006, R=0x2007)

    const btnConnectWebHID = document.getElementById('btnConnectWebHID');
    const gamepadStatus = document.getElementById('gamepadStatus');

    function resetHidUI() {
        hidDevice = null;
        hidPacketCounter = 0;
        hidProductId = 0;
        if (btnConnectWebHID) {
            btnConnectWebHID.disabled = false;
            btnConnectWebHID.textContent = "Connect Joy-Con (WebHID)";
            btnConnectWebHID.style.background = "";
        }
        if (gamepadStatus) {
            gamepadStatus.textContent = "None";
            gamepadStatus.style.color = "";
        }
    }

    // Listen for HID device disconnection
    if ("hid" in navigator) {
        navigator.hid.addEventListener("disconnect", (event) => {
            if (hidDevice && event.device === hidDevice) {
                console.log("[WebHID] Joy-Con disconnected");
                resetHidUI();
            }
        });
    }

    async function sendSubcommand(cmd, subcmd, args) {
        if (!hidDevice) return;
        const data = new Uint8Array(10 + args.length);
        data[0] = hidPacketCounter & 0x0F; hidPacketCounter++;
        data[1] = 0x00;
        data[2] = 0x01; // rumble flags
        data[3] = 0x40; data[4] = 0x40; data[5] = 0x00;
        data[6] = 0x01; data[7] = 0x40; data[8] = 0x40;
        data[9] = subcmd;
        for (let i = 0; i < args.length; i++) data[10 + i] = args[i];
        await hidDevice.sendReport(cmd, data);
    }

    // Convert Uint8Array to base64
    function uint8ToBase64(u8arr) {
        let binary = '';
        for (let i = 0; i < u8arr.length; i++) {
            binary += String.fromCharCode(u8arr[i]);
        }
        return btoa(binary);
    }

    if (btnConnectWebHID) {
        btnConnectWebHID.addEventListener('click', async () => {
            if (!("hid" in navigator)) {
                alert("WebHID Not Supported in this browser!");
                return;
            }
            try {
                const devices = await navigator.hid.requestDevice({ filters: [{ vendorId: 0x057E }] });
                if (devices.length === 0) return;

                hidDevice = devices[0];
                hidProductId = hidDevice.productId;
                await hidDevice.open();
                console.log(`[WebHID] Connected: ${hidDevice.productName} (pid=0x${hidProductId.toString(16)})`);

                btnConnectWebHID.disabled = true;
                btnConnectWebHID.textContent = "Joy-Con Connected";
                btnConnectWebHID.style.background = "linear-gradient(135deg, #1b5e20, #004d40)";
                if (gamepadStatus) {
                    gamepadStatus.textContent = hidDevice.productName;
                    gamepadStatus.style.color = '#4caf50';
                }

                // Notify backend which JoyCon is connected
                send({
                    type: 'joycon_connected',
                    product_id: hidProductId,
                    product_name: hidDevice.productName
                });

                hidDevice.addEventListener("inputreport", handleJoyconReport);

                // Enable IMU Mode (0x40, 0x01)
                await sendSubcommand(0x01, 0x40, [0x01]);
                // Standard Full mode 60Hz (0x03, 0x30)
                await sendSubcommand(0x01, 0x03, [0x30]);

            } catch (err) {
                console.error(err);
                alert("Joy-Con WebHID connection failed: " + err.message);
            }
        });
    }

    function handleJoyconReport(event) {
        const { reportId, data } = event;
        if (reportId !== 0x30) return;

        // Forward the raw HID report to backend for decoding
        const rawBytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        send({
            type: 'joycon_raw',
            report_id: reportId,
            product_id: hidProductId,
            data: uint8ToBase64(rawBytes),
            ts: Date.now()
        });
    }

    let selectedDataset = null;

    async function loadDatasets() {
        const listEl = document.getElementById('datasetList');
        if (!listEl) return;

        try {
            const resp = await fetch('/api/datasets');
            const data = await resp.json();
            listEl.innerHTML = '';

            if (data.datasets.length === 0) {
                listEl.innerHTML = '<div style="color:#666;font-size:12px">No datasets saved yet</div>';
                return;
            }

            data.datasets.forEach(ds => {
                const item = document.createElement('div');
                item.className = 'dataset-item';
                item.innerHTML = `
                    <div class="dataset-name">${ds.name.replace('teleop_', '').replace('.zarr', '')}</div>
                    <div class="dataset-meta">${ds.steps} steps · ${ds.episodes} eps</div>
                `;
                item.addEventListener('click', () => {
                    selectedDataset = ds.name;
                    loadDatasetDetail(ds.name);
                });
                listEl.appendChild(item);
            });
        } catch (e) {
            listEl.innerHTML = '<div style="color:#f44336;font-size:12px">Failed to load</div>';
        }
    }

    async function loadDatasetDetail(name) {
        const detailEl = document.getElementById('datasetDetail');
        if (!detailEl) return;

        try {
            const resp = await fetch(`/api/datasets/${name}`);
            const data = await resp.json();

            let html = `
                <div style="font-size:11px;color:#888;margin-bottom:6px">${name}</div>
                <div style="font-size:12px;color:#e0e0e0">
                    Steps: <b>${data.total_steps}</b> · Eps: <b>${data.episodes}</b><br>
                    Obs: <b>${data.obs_shape.join('×')}</b> · Act: <b>${data.action_shape.join('×')}</b><br>
                    Σrew: <b>${data.reward_sum.toFixed(2)}</b> · μ: <b>${data.reward_mean.toFixed(4)}</b>
                </div>
            `;

            if (data.rewards && data.rewards.length > 0) {
                html += '<canvas id="rewardCanvas" width="240" height="80" style="margin-top:8px;border-radius:4px;background:#0a0a0f;width:100%"></canvas>';
            }

            detailEl.innerHTML = html;

            if (data.rewards && data.rewards.length > 0) {
                setTimeout(() => drawRewardChart(data.rewards, data.episode_ends), 50);
            }
        } catch (e) {
            detailEl.innerHTML = '<div style="color:#f44336;font-size:12px">Error loading</div>';
        }
    }



    function drawRewardChart(rewards, episodeEnds) {
        const cvs = document.getElementById('rewardCanvas');
        if (!cvs) return;
        const c = cvs.getContext('2d');
        const w = cvs.width, h = cvs.height;
        c.clearRect(0, 0, w, h);

        const maxR = Math.max(...rewards, 0.001);
        const minR = Math.min(...rewards, 0);
        const range = maxR - minR || 1;
        const step = Math.max(1, Math.floor(rewards.length / w));

        c.beginPath();
        c.strokeStyle = '#7c4dff';
        c.lineWidth = 1;
        for (let i = 0; i < rewards.length; i += step) {
            const x = (i / rewards.length) * w;
            const y = h - ((rewards[i] - minR) / range) * (h - 4) - 2;
            if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
        }
        c.stroke();

        c.strokeStyle = 'rgba(76, 175, 80, 0.3)';
        c.lineWidth = 1;
        episodeEnds.forEach(end => {
            const x = (end / rewards.length) * w;
            c.beginPath(); c.moveTo(x, 0); c.lineTo(x, h); c.stroke();
        });
    }

    // ================================================================
    //  Panel Controls
    // ================================================================

    async function loadTasks() {
        try {
            const resp = await fetch('/api/tasks');
            const data = await resp.json();
            const select = document.getElementById('taskSelect');
            select.innerHTML = '';
            const labels = data.task_labels || {};
            data.tasks.forEach(task => {
                const opt = document.createElement('option');
                opt.value = task;
                const label = labels[task];
                opt.textContent = label ? `${task} - ${label}` : task;
                select.appendChild(opt);
            });

            if (data.current_task && data.tasks.includes(data.current_task)) {
                select.value = data.current_task;
            } else if (data.tasks.includes('pick-place-v3')) {
                select.value = 'pick-place-v3';
            } else if (data.tasks.length > 0) {
                select.value = data.tasks[0];
            }
        } catch (e) {
            console.error('[Teleop] Failed to load tasks:', e);
        }
    }

    document.getElementById('taskSelect').addEventListener('change', (e) => {
        send({ type: 'set_task', task: e.target.value });
    });

    const speedSlider = document.getElementById('speedSlider');
    const speedValue = document.getElementById('speedValue');
    speedSlider.addEventListener('input', () => {
        const val = parseFloat(speedSlider.value);
        speedValue.textContent = val.toFixed(2);
        send({ type: 'set_speed', speed: val });
    });

    document.getElementById('btnReset').addEventListener('click', () => {
        send({ type: 'reset' }); container.focus();
    });
    document.getElementById('btnRecord').addEventListener('click', () => {
        send({ type: 'toggle_record' }); container.focus();
    });
    document.getElementById('btnSave').addEventListener('click', () => {
        send({ type: 'save' }); container.focus();
    });

    // Control mode toggle
    const btnToggleMode = document.getElementById('btnToggleMode');
    if (btnToggleMode) {
        btnToggleMode.addEventListener('click', () => {
            toggleControlMode();
            container.focus();
        });
    }

    // Input mode selector
    const inputModeSelect = document.getElementById('inputModeSelect');
    if (inputModeSelect) {
        inputModeSelect.addEventListener('change', (e) => setInputMode(e.target.value));
    }

    // Refresh datasets
    const btnRefreshData = document.getElementById('btnRefreshData');
    if (btnRefreshData) {
        btnRefreshData.addEventListener('click', loadDatasets);
    }

    // ================================================================
    //  Initialize
    // ================================================================

    loadTasks();
    loadDatasets();
    connect();
    updateControlModeUI();
    setTimeout(() => container.focus(), 500);

})();
