'use strict';
const $ = id => document.getElementById(id);
let people = [];
let currentReport = null;
let toastTimer;
let recognitionVersion = 0;

function toast(message, error = false) {
  clearTimeout(toastTimer);
  $('toast').textContent = message;
  $('toast').classList.toggle('error', error);
  $('toast').hidden = false;
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, error ? 10000 : 5000);
}

async function api(path, options = {}) {
  const response = await fetch(path, {...options, signal: AbortSignal.timeout(180000)});
  const result = await response.json();
  if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : 'Не удалось выполнить запрос.');
  return result;
}

async function busy(button, label, action, onError) {
  const previous = button.innerHTML;
  button.disabled = true; button.textContent = label;
  try { await action(); }
  catch (error) {
    const message = error.name === 'TimeoutError' ? 'Обработка заняла слишком много времени. Попробуйте снимок меньшего размера.' : error.message;
    if (onError) onError(message);
    else toast(message, true);
  } finally { button.disabled = false; button.innerHTML = previous; }
}

function selectTab(name) {
  const names = {recognition: 'Распознавание', people: 'База людей', evaluation: 'Проверка качества'};
  document.querySelectorAll('.tab-panel').forEach(panel => { panel.hidden = panel.id !== `tab-${name}`; });
  document.querySelectorAll('.nav-button').forEach(button => {
    const active = button.dataset.tab === name;
    button.classList.toggle('active', active);
    if (active) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
  });
  $('breadcrumb').textContent = names[name];
  document.title = `${names[name]} — FaceLab`;
  if (name === 'evaluation') refreshReport().catch(error => toast(error.message, true));
}
document.querySelectorAll('.nav-button').forEach(button => button.addEventListener('click', () => selectTab(button.dataset.tab)));

function resetRecognition() {
  recognitionVersion += 1;
  $('result-content').hidden = true; $('result-empty').hidden = false;
  $('result-status').textContent = 'ОЖИДАНИЕ СНИМКА';
}

function validateFiles(input) {
  const max = input.multiple ? 20 : 1;
  if (input.files.length > max) throw new Error(`Выберите не больше ${max} файлов.`);
  for (const file of input.files) {
    if (file.size > 10 * 1024 * 1024) throw new Error(`«${file.name}» превышает 10 МБ.`);
  }
}

function updateFiles(input) {
  try { validateFiles(input); }
  catch (error) { input.value = ''; toast(error.message, true); }
  if (input.id === 'recognize-file') {
    const file = input.files[0];
    $('selected-file').hidden = !file;
    $('selected-file').textContent = file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} МБ` : '';
    resetRecognition();
  } else {
    const label = input.id === 'enroll-files' ? $('enroll-files-label') : $('eval-files-label');
    label.textContent = input.files.length ? `Выбрано снимков: ${input.files.length}` : 'Один человек на снимке · до 20 файлов';
  }
}
document.querySelectorAll('input[type=file]').forEach(input => {
  input.addEventListener('change', () => updateFiles(input));
  const zone = input.closest('.dropzone');
  ['dragenter', 'dragover'].forEach(event => zone.addEventListener(event, e => {e.preventDefault(); zone.classList.add('dragging');}));
  ['dragleave', 'drop'].forEach(event => zone.addEventListener(event, e => {e.preventDefault(); zone.classList.remove('dragging');}));
  zone.addEventListener('drop', event => {
    if (!input.multiple && event.dataTransfer.files.length !== 1) {toast('Выберите одну фотографию.', true); return;}
    input.files = event.dataTransfer.files; updateFiles(input);
  });
});
$('threshold').addEventListener('input', () => { $('threshold-value').value = Number($('threshold').value).toFixed(3); });

async function refreshPeople() {
  const [status, collection] = await Promise.all([api('/api/status'), api('/api/people')]);
  people = collection.people;
  $('model-status').classList.toggle('ready', status.ready);
  $('model-status').classList.toggle('error', !status.ready);
  $('model-status').lastElementChild.textContent = status.ready ? 'Модели готовы · офлайн' : 'Нужна установка моделей';
  $('setup-warning').hidden = status.ready;
  $('people-count').textContent = status.people_count;
  $('samples-count').textContent = status.samples_count;
  $('nav-count').textContent = status.people_count;
  $('people-empty').hidden = !!people.length;
  $('gallery-note').textContent = people.length ? `В базе: ${people.length} · снимок останется на вашем компьютере.` : 'База пуста: лица найдутся, но все получат метку «Неизвестный».';
  $('people-grid').replaceChildren();
  for (const person of people) {
    const card = document.createElement('article'); card.className = 'person-card';
    const image = document.createElement('img'); image.src = `/api/people/${person.id}/thumbnail`; image.alt = `Миниатюра: ${person.name}`;
    const name = document.createElement('h3'); name.textContent = person.name;
    const info = document.createElement('p'); info.textContent = `Эталонных снимков: ${person.samples}`;
    const actions = document.createElement('div'); actions.className = 'person-actions';
    const date = document.createElement('span'); date.textContent = new Date(person.created_at).toLocaleDateString('ru-RU');
    const remove = document.createElement('button'); remove.className = 'text-button muted'; remove.type = 'button'; remove.textContent = 'Удалить'; remove.setAttribute('aria-label', `Удалить: ${person.name}`);
    remove.addEventListener('click', async () => {
      if (!confirm(`Удалить «${person.name}», признаки и миниатюру? Текущая проверка качества также будет сброшена.`)) return;
      await busy(remove, 'Удаляем…', async () => {
        await api(`/api/people/${person.id}`, {method: 'DELETE'});
        resetRecognition(); await refreshPeople(); await refreshReport(); toast('Запись удалена.');
      });
    });
    actions.append(date, remove); card.append(image, name, info, actions); $('people-grid').append(card);
  }
  const previous = $('expected-person').value;
  $('expected-person').replaceChildren(new Option('Неизвестный — человека нет в базе', ''));
  people.forEach(person => $('expected-person').append(new Option(person.name, person.id)));
  if (people.some(person => person.id === previous)) $('expected-person').value = previous;
}

function openEnrollment() { $('enroll-error').hidden = true; $('enroll-dialog').showModal(); }
$('open-enroll').addEventListener('click', openEnrollment);
$('empty-enroll').addEventListener('click', openEnrollment);
$('close-enroll').addEventListener('click', () => $('enroll-dialog').close());
$('enroll-dialog').addEventListener('cancel', event => { if ($('enroll-button').disabled) event.preventDefault(); });
$('enroll-form').addEventListener('submit', async event => {
  event.preventDefault(); $('enroll-error').hidden = true;
  await busy($('enroll-button'), 'Обрабатываем фотографии…', async () => {
    validateFiles($('enroll-files'));
    const form = new FormData(); form.append('name', $('person-name').value);
    for (const file of $('enroll-files').files) form.append('files', file);
    const result = await api('/api/people', {method: 'POST', body: form});
    $('enroll-dialog').close(); $('enroll-form').reset(); updateFiles($('enroll-files'));
    resetRecognition(); await refreshPeople(); await refreshReport();
    toast(`«${result.person.name}» добавлен в базу.`);
  }, message => { $('enroll-error').textContent = message; $('enroll-error').hidden = false; });
});

async function drawResult(result, version) {
  const image = new Image(); image.src = result.image; await image.decode();
  if (version !== recognitionVersion) return;
  const canvas = $('result-canvas'); canvas.width = result.width; canvas.height = result.height;
  const ctx = canvas.getContext('2d'); ctx.drawImage(image, 0, 0);
  const fontSize = Math.max(15, Math.round(result.width / 55));
  ctx.font = `600 ${fontSize}px "Segoe UI", sans-serif`;
  for (const [index, face] of result.faces.entries()) {
    const [x, y, w, h] = face.box; const known = face.person_id !== null;
    const color = known ? '#bfe876' : '#f2c273';
    ctx.strokeStyle = color; ctx.lineWidth = Math.max(2, result.width / 500); ctx.strokeRect(x, y, w, h);
    let label = `${index + 1}. ${face.name}`;
    while (ctx.measureText(label).width > result.width - 24 && label.length > 10) label = label.slice(0, -2) + '…';
    const labelWidth = Math.min(ctx.measureText(label).width + 16, result.width);
    const tx = Math.min(Math.max(0, x), result.width - labelWidth), ty = y >= fontSize + 15 ? y - fontSize - 12 : y;
    ctx.fillStyle = color; ctx.fillRect(tx, ty, labelWidth, fontSize + 12);
    ctx.fillStyle = '#1b3b30'; ctx.fillText(label, tx + 8, ty + fontSize + 2);
  }
  $('result-empty').hidden = true; $('result-content').hidden = false;
  $('result-status').textContent = 'ОБРАБОТКА ЗАВЕРШЕНА';
  const known = result.faces.filter(face => face.person_id !== null).length;
  $('result-summary').textContent = `Лиц: ${result.faces.length} · узнано: ${known} · ${Math.round(result.elapsed_ms)} мс · порог ${result.threshold.toFixed(3)}`;
  $('face-list').replaceChildren();
  for (const [index, face] of result.faces.entries()) {
    const row = document.createElement('div'); row.className = `face-row${face.person_id ? '' : ' unknown'}`;
    const badge = document.createElement('span'); badge.className = 'face-index'; badge.textContent = index + 1;
    const name = document.createElement('span'); name.className = 'face-name'; name.textContent = face.name;
    const score = document.createElement('span'); score.className = 'face-score'; score.textContent = face.similarity === null ? 'База пуста' : `Сходство ${face.similarity.toFixed(3)}`;
    row.append(badge, name, score); $('face-list').append(row);
  }
  $('no-faces').hidden = result.faces.length > 0;
}
$('recognize-form').addEventListener('submit', async event => {
  event.preventDefault();
  await busy($('recognize-button'), 'Ищем знакомые лица…', async () => {
    validateFiles($('recognize-file'));
    const form = new FormData(); form.append('file', $('recognize-file').files[0]); form.append('threshold', $('threshold').value);
    resetRecognition(); $('result-status').textContent = 'ОБРАБОТКА…';
    const version = recognitionVersion;
    try {
      const result = await api('/api/recognize', {method: 'POST', body: form});
      if (version === recognitionVersion) await drawResult(result, version);
    } catch (error) {
      if (version === recognitionVersion) {
        $('result-status').textContent = 'НЕ УДАЛОСЬ ОБРАБОТАТЬ'; throw error;
      }
    }
  });
});

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$('download-image').addEventListener('click', () => $('result-canvas').toBlob(blob => { if (blob) downloadBlob(blob, 'facelab-result.png'); }, 'image/png'));

const percent = value => value === null ? '—' : `${(value * 100).toFixed(1)}%`;
function renderReport(report) {
  currentReport = report; const metrics = report.metrics;
  $('metric-accuracy').textContent = percent(metrics.accuracy);
  $('metric-total').textContent = metrics.total ? `${metrics.correct} верных из ${metrics.total} снимков` : 'Ещё нет проверенных снимков';
  $('metric-fa').textContent = percent(metrics.false_accept_rate);
  $('metric-fr').textContent = percent(metrics.false_reject_rate);
  $('metric-fa-help').textContent = metrics.unknown ? `${metrics.false_accepts} из ${metrics.unknown} неизвестных` : 'Нужны снимки неизвестных людей';
  $('metric-fr-help').textContent = metrics.known ? `${metrics.false_rejects} из ${metrics.known} знакомых` : 'Нужны новые снимки знакомых';
  $('metric-time').textContent = metrics.mean_ms === null ? '—' : `${Math.round(metrics.mean_ms)} мс`;
  $('evaluation-empty').hidden = metrics.total > 0;
  $('evaluation-rows').replaceChildren();
  for (const entry of report.cases) {
    const tr = document.createElement('tr');
    [entry.filename, entry.expected_name, entry.predicted_name, entry.similarity === null ? '—' : entry.similarity.toFixed(3)].forEach(value => { const td = document.createElement('td'); td.textContent = value; tr.append(td); });
    const td = document.createElement('td'), badge = document.createElement('span'); badge.className = entry.correct ? 'pass-tag' : 'fail-tag'; badge.textContent = entry.correct ? 'Верно' : 'Ошибка'; td.append(badge); tr.append(td); $('evaluation-rows').append(tr);
  }
  $('evaluation-note').textContent = metrics.total ? `Порог: ${report.threshold.toFixed(3)} · Ошибок выбора имени: ${metrics.wrong_identity}. При изменении базы проверка сбрасывается.` : '';
  if (report.threshold !== null) $('eval-threshold').value = report.threshold;
}
async function refreshReport() { renderReport(await api('/api/report')); }
$('evaluate-form').addEventListener('submit', async event => {
  event.preventDefault();
  await busy($('evaluate-button'), 'Проверяем снимки…', async () => {
    validateFiles($('evaluate-files'));
    const form = new FormData(); form.append('expected_id', $('expected-person').value); form.append('threshold', $('eval-threshold').value);
    for (const file of $('evaluate-files').files) form.append('files', file);
    renderReport(await api('/api/evaluate', {method: 'POST', body: form}));
    $('evaluate-files').value = ''; updateFiles($('evaluate-files')); toast('Результаты добавлены к проверке.');
  });
});
$('reset-report').addEventListener('click', async () => {
  if (currentReport?.metrics.total && !confirm('Сбросить текущие результаты? При необходимости сначала скачайте отчёт.')) return;
  await busy($('reset-report'), 'Сбрасываем…', async () => { renderReport(await api('/api/report', {method: 'DELETE'})); toast('Можно начать новую проверку.'); });
});
$('download-report').addEventListener('click', async () => {
  try { const report = await api('/api/report'); downloadBlob(new Blob([JSON.stringify(report, null, 2)], {type: 'application/json'}), 'facelab-evaluation.json'); }
  catch (error) { toast(error.message, true); }
});
Promise.all([refreshPeople(), refreshReport()]).catch(error => toast(`Не удалось загрузить приложение: ${error.message}`, true));
