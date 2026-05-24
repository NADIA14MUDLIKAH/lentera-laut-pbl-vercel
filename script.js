const API_BASE_URL = "http://localhost:8000";

let lineChart = null;
let pieChart = null;

// ELEMENT SELECTORS
const loginView = document.getElementById('login-view');
const mainLayout = document.getElementById('main-layout');
const btnLogin = document.getElementById('btn-login');
const inputPhone = document.getElementById('input-phone');
const selectLokasi = document.getElementById('select-lokasi');
const btnPrediksi = document.getElementById('btn-prediksi');

// 1. LOGIN SEDERHANA
btnLogin.addEventListener('click', () => {
    if(inputPhone.value.trim() === "") {
        alert("Masukkan ID / No HP"); return;
    }
    loginView.classList.add('hidden');
    mainLayout.classList.remove('hidden');
    initApp();
});
inputPhone.addEventListener('keypress', (e) => { if (e.key === 'Enter') btnLogin.click(); });

// 2. INISIALISASI LOKASI & GRAFIK KOSONG
async function initApp() {
    initCharts(); // Siapkan kanvas kosong
    try {
        const response = await fetch(`${API_BASE_URL}/locations`);
        const lokasi = await response.json();
        selectLokasi.innerHTML = '<option value="">Pilih Lokasi...</option>';
        lokasi.forEach(l => {
            selectLokasi.innerHTML += `<option value="${l.name}">${l.name}</option>`;
        });
        btnPrediksi.disabled = false;
    } catch (err) {
        console.error("Gagal muat lokasi:", err);
    }
}

// 3. TOMBOL PINDAI CUACA
btnPrediksi.addEventListener('click', async () => {
    const loc = selectLokasi.value;
    if(!loc) return;

    btnPrediksi.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Memproses...';
    btnPrediksi.disabled = true;

    try {
        // Tarik Data Prediksi (+1 Jam)
        const resPred = await fetch(`${API_BASE_URL}/predict?location=${encodeURIComponent(loc)}`);
        if(!resPred.ok) throw new Error("Gagal mengambil prediksi ML");
        const dataPred = await resPred.json();
        
        updateUICards(dataPred.prediction);

        // Tarik 24 Data Historis Mentah dari API untuk disajikan di Grafik
        const resHist = await fetch(`${API_BASE_URL}/marine-weather?location=${encodeURIComponent(loc)}&limit=24`);
        if(resHist.ok) {
            const dataHist = await resHist.json();
            updateChartsData(dataHist);
        }

    } catch (err) {
        alert(err.message);
    } finally {
        btnPrediksi.innerHTML = '<i class="fas fa-satellite-dish"></i> Pindai Cuaca';
        btnPrediksi.disabled = false;
    }
});

// 4. UPDATE ANGKA DI KARTU KIRI
function updateUICards(pred) {
    const timeFormat = new Date(pred.time_prediction).toLocaleTimeString('id-ID', {hour:'2-digit', minute:'2-digit'});
    document.getElementById('waktu-teks').innerText = `${timeFormat} WIB`;
    
    document.getElementById('val-wave').innerText = pred.wave_height.value.toFixed(2);
    document.getElementById('val-wind').innerText = pred.wind_speed_10m.value.toFixed(2);
    document.getElementById('val-rain').innerText = pred.precipitation.value.toFixed(2);
    document.getElementById('val-vis').innerText = pred.visibility.value.toLocaleString('id-ID');

    aturBadge('kat-wave', pred.wave_height.kategori);
    aturBadge('kat-wind', pred.wind_speed_10m.kategori);
    aturBadge('kat-rain', pred.precipitation.kategori);
    aturBadge('kat-vis', pred.visibility.kategori);
}

function aturBadge(id, teks) {
    const el = document.getElementById(id);
    el.innerText = teks || "--";
    el.className = "badge"; 
    
    if(!teks) { el.classList.add('normal'); return; }
    
    const txt = teks.toLowerCase();
    if(txt.includes("ekstrem") || txt.includes("tinggi") || txt.includes("storm") || txt.includes("poor") || txt.includes("heavy")) {
        el.classList.add('bahaya');
    } else if(txt.includes("sedang") || txt.includes("moderate") || txt.includes("breeze")) {
        el.classList.add('waspada');
    } else {
        el.classList.add('aman');
    }
}

// 5. INISIALISASI & UPDATE GRAFIK (CHART.JS)
function initCharts() {
    // Grafik Garis (Line Chart - 2 Dataset)
    const ctxLine = document.getElementById('trendChart').getContext('2d');
    lineChart = new Chart(ctxLine, {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'Tinggi Gelombang (m)', borderColor: '#2563eb', backgroundColor: 'rgba(37, 99, 235, 0.1)', data: [], fill: true, tension: 0.3, yAxisID: 'y' },
            { label: 'Kecepatan Angin (m/s)', borderColor: '#f59e0b', data: [], fill: false, tension: 0.3, yAxisID: 'y1' }
        ]},
        options: {
            responsive: true, maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            scales: {
                y: { type: 'linear', display: true, position: 'left', title: {display: true, text: 'Gelombang (m)'} },
                y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false }, title: {display: true, text: 'Angin (m/s)'} },
                x: { grid: { display: false } }
            },
            plugins: { legend: { position: 'bottom' } }
        }
    });

    // Grafik Donat (Pie Chart)
    const ctxPie = document.getElementById('pieChart').getContext('2d');
    pieChart = new Chart(ctxPie, {
        type: 'doughnut',
        data: {
            labels: ['Aman (< 1.25m)', 'Waspada (1.25 - 2.5m)', 'Bahaya (> 2.5m)'],
            datasets: [{
                data: [1, 1, 1], // Data awal kosong
                backgroundColor: ['#22c55e', '#f59e0b', '#ef4444'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { position: 'right' } },
            cutout: '65%'
        }
    });
}

function updateChartsData(histData) {
    // Balik urutan dari terlama ke terbaru
    const data = histData.reverse(); 

    // Ekstrak Array
    const labels = data.map(d => new Date(d.time).toLocaleTimeString('id-ID', {hour:'2-digit', minute:'2-digit'}));
    const waveData = data.map(d => d.wave_height);
    const windData = data.map(d => d.wind_speed_10m);

    // Update Line Chart
    lineChart.data.labels = labels;
    lineChart.data.datasets[0].data = waveData;
    lineChart.data.datasets[1].data = windData;
    lineChart.update();

    // Hitung Distribusi Kategori Gelombang untuk Doughnut Chart
    let countAman = 0, countWaspada = 0, countBahaya = 0;
    waveData.forEach(w => {
        if(w < 1.25) countAman++;
        else if (w < 2.5) countWaspada++;
        else countBahaya++;
    });

    // Update Pie Chart
    pieChart.data.datasets[0].data = [countAman, countWaspada, countBahaya];
    pieChart.update();
}