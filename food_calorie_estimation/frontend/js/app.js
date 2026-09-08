// 前端交互
// =====================================

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const btnAnalyze = document.getElementById('btnAnalyze');
const btnReset = document.getElementById('btnReset');
const resultsSection = document.getElementById('resultsSection');
const rgbImage = document.getElementById('rgbImage');
const nirImage = document.getElementById('nirImage');
const sampleList = document.getElementById('sampleList');

let selectedFile = null;
let currentObjectURL = null;

// ------------------------ 健康检查 + 示例图加载 ------------------------
async function bootstrap() {
    try {
        const resp = await fetch('/api/health');
        const data = await resp.json();
        if (data.status === 'ok') {
            updateStatusBar('ok', '后端已就绪', data.data);
        } else {
            updateStatusBar('error', '后端异常: ' + (data.error || 'unknown'));
        }
    } catch (e) {
        updateStatusBar('error', '后端无法连接, 请检查服务是否启动 (python app/server.py)');
    }
    loadSamples();
}

function updateStatusBar(level, text, data) {
    const dot = document.getElementById('statusDot');
    const textEl = document.getElementById('statusText');
    const deviceEl = document.getElementById('deviceText');
    const mtEl = document.getElementById('multitaskText');
    dot.className = 'status-dot ' + level;
    textEl.textContent = text;
    if (data) {
        deviceEl.textContent = 'device: ' + (data.device || '-');
        mtEl.textContent = data.multitask_loaded
            ? 'multitask: 已加载 (真实模型)'
            : 'multitask: 未加载 (将用 ImageNet 兜底)';
    }
}

async function loadSamples() {
    try {
        const resp = await fetch('/api/sample-images');
        const data = await resp.json();
        sampleList.innerHTML = '';
        if (!data.success || !data.data.samples || data.data.samples.length === 0) {
            sampleList.innerHTML = '<span class="sample-loading">无内置示例, 请上传图片</span>';
            return;
        }
        for (const s of data.data.samples) {
            const chip = document.createElement('div');
            chip.className = 'sample-chip';
            chip.title = s.name;
            const img = document.createElement('img');
            img.src = 'data:image/png;base64,' + s.image;
            img.alt = s.name;
            chip.appendChild(img);
            chip.addEventListener('click', () => useSample(s));
            sampleList.appendChild(chip);
        }
    } catch (e) {
        sampleList.innerHTML = '<span class="sample-loading">示例加载失败</span>';
    }
}

async function useSample(sample) {
    // 转为 Blob 然后用 File 包装, 与上传走同一逻辑
    const bin = atob(sample.image);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    const blob = new Blob([bytes], { type: 'image/png' });
    const file = new File([blob], sample.name, { type: 'image/png' });
    handleFileSelect(file);
}

// ------------------------ 文件选择 ------------------------
uploadArea.addEventListener('click', () => fileInput.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0 && e.dataTransfer.files[0].type.startsWith('image/')) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFileSelect(e.target.files[0]);
});

function handleFileSelect(file) {
    selectedFile = file;
    if (currentObjectURL) URL.revokeObjectURL(currentObjectURL);
    currentObjectURL = URL.createObjectURL(file);
    uploadArea.classList.add('has-image');
    uploadArea.innerHTML = `
        <img class="preview" src="${currentObjectURL}" alt="preview">
        <p class="preview-name">${file.name} (${(file.size / 1024).toFixed(1)} KB)</p>
        <p style="color:#999;font-size:0.8rem;margin-top:4px;">点击重新选择</p>
    `;
    btnAnalyze.disabled = false;
}

btnReset.addEventListener('click', () => {
    selectedFile = null;
    if (currentObjectURL) URL.revokeObjectURL(currentObjectURL);
    currentObjectURL = null;
    uploadArea.classList.remove('has-image');
    uploadArea.innerHTML = `
        <div class="upload-icon">📷</div>
        <p class="upload-text">点击 / 拖拽 上传食物图片</p>
        <p class="upload-hint">支持 JPG、PNG 格式</p>
    `;
    fileInput.value = '';
    btnAnalyze.disabled = true;
    resultsSection.style.display = 'none';
});

// ------------------------ 分析 ------------------------
btnAnalyze.addEventListener('click', async () => {
    if (!selectedFile) return;

    const btnText = btnAnalyze.querySelector('.btn-text');
    const btnLoading = btnAnalyze.querySelector('.btn-loading');
    btnAnalyze.disabled = true;
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline';
    resultsSection.style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);

        const response = await fetch('/api/predict', { method: 'POST', body: formData });
        const result = await response.json();

        if (result.success) {
            displayResults(result.data);
            showToast('分析完成', 'success');
        } else {
            showToast('分析失败: ' + result.error, 'error');
            console.error(result);
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('网络错误, 请检查后端服务是否启动', 'error');
    } finally {
        btnAnalyze.disabled = false;
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
    }
});

function displayResults(data) {
    rgbImage.src = 'data:image/png;base64,' + data.rgb_image;
    nirImage.src = 'data:image/png;base64,' + data.nir_image;

    const ms = data.multispectral;
    document.getElementById('msFoodClass').textContent = ms.food_class;
    document.getElementById('msConfidence').textContent = (ms.class_probability * 100).toFixed(1) + '%';
    document.getElementById('msCalories').textContent = ms.calories.toFixed(1) + ' kcal';
    document.getElementById('msWeight').textContent = ms.weight.toFixed(1) + ' g';
    document.getElementById('msTop5').innerHTML = renderTop5(ms.top5_classes);
    document.getElementById('msSource').textContent = '推理来源: ' + (ms.source || '-');

    const bl = data.baseline;
    document.getElementById('blFoodClass').textContent = bl.food_class;
    document.getElementById('blConfidence').textContent = (bl.class_probability * 100).toFixed(1) + '%';
    document.getElementById('blCalories').textContent = bl.calories.toFixed(1) + ' kcal';
    document.getElementById('blWeight').textContent = bl.weight.toFixed(1) + ' g';
    document.getElementById('blTop5').innerHTML = renderTop5(bl.top5_classes);
    document.getElementById('blSource').textContent = '推理来源: ' + (bl.source || '-');

    const comp = data.comparison;
    const calDiffEl = document.getElementById('calDiff');
    const calImpEl = document.getElementById('calImprovement');
    const calAbsEl = document.getElementById('calAbsDiff');

    const diff = comp.calories_diff;
    calDiffEl.textContent = (diff >= 0 ? '+' : '') + diff.toFixed(1) + ' kcal';
    calDiffEl.className = 'summary-value ' + (diff >= 0 ? 'positive' : 'negative');
    calImpEl.textContent = comp.relative_change_pct.toFixed(2) + '%';
    calAbsEl.textContent = comp.abs_mean_diff.toFixed(1) + ' kcal';

    resultsSection.style.display = 'block';
    setTimeout(() => {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function renderTop5(classes) {
    if (!classes || classes.length === 0) return '<div class="top5-title">无预测结果</div>';
    let html = '<div class="top5-title">Top 5 预测类别:</div>';
    classes.forEach((cls, index) => {
        const barWidth = Math.max(2, Math.round(cls.probability * 100));
        html += `
            <div class="top5-item">
                <span class="top5-name">${index + 1}. ${cls.name} <span style="color:#999;font-size:0.8em;">(${cls.category})</span></span>
                <span class="top5-prob">${(cls.probability * 100).toFixed(1)}%</span>
            </div>
        `;
    });
    return html;
}

// ------------------------ toast 提示 ------------------------
function showToast(message, type) {
    const existing = document.querySelector('.toast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.className = 'toast ' + (type || '');
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.classList.add('show'), 10);
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

bootstrap();
