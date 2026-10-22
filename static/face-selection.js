// Detection preview only. The user explicitly chooses an area, without identity matching.
export function renderQualityPreview(container, files, choices, onSelect) {
  container.replaceChildren();
  for (const [fileIndex, file] of files.entries()) {
    const item = document.createElement('div'); item.className = 'quality-item';
    const title = document.createElement('strong'); title.textContent = file.filename; item.append(title);
    const messages = file.face_count === 0
      ? ['Лиц не найдено. Выберите другой снимок.']
      : file.face_count > 1 ? [`Найдено лиц: ${file.face_count}. Выберите одно лицо для этой записи.`]
      : file.faces[0].quality.issues.map(issue => issue.message);
    if (messages.length) item.classList.add('warning');
    for (const message of messages.length ? messages : ['Замечаний к размеру, резкости и освещению нет.']) {
      const paragraph = document.createElement('p'); paragraph.textContent = message; item.append(paragraph);
    }
    if (file.face_count > 1) {
      const canvas = document.createElement('canvas'); canvas.className = 'face-selection-canvas';
      canvas.width = file.width; canvas.height = file.height;
      canvas.setAttribute('role', 'img'); canvas.setAttribute('aria-label', 'Обнаруженные области лица с номерами. Выберите номер кнопкой под снимком.');
      const buttons = document.createElement('div'); buttons.className = 'face-choice-buttons';
      const selectedNote = document.createElement('p'); selectedNote.className = 'face-choice-note';
      const image = new Image();
      const paint = () => {
        if (!image.complete || !image.naturalWidth) return;
        const ctx = canvas.getContext('2d'); ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        const size = Math.max(16, Math.round(Math.max(file.width, file.height) / 40));
        ctx.font = `bold ${size}px "Segoe UI",sans-serif`;
        file.faces.forEach((face, index) => {
          const [x, y, w, h] = face.box;
          ctx.strokeStyle = choices[fileIndex] === index ? '#bfe876' : '#ffc576';
          ctx.lineWidth = Math.max(2, file.width / 220); ctx.strokeRect(x, y, w, h);
          const labelY = Math.max(0, y - size - 8);
          ctx.fillStyle = ctx.strokeStyle; ctx.fillRect(x, labelY, size * 2, size + 8);
          ctx.fillStyle = '#183c30'; ctx.fillText(String(index + 1), x + 5, labelY + size + 1);
        });
      };
      const choose = index => {
        choices[fileIndex] = index; onSelect(fileIndex, index);
        [...buttons.children].forEach((button, n) => button.setAttribute('aria-pressed', String(n === index)));
        selectedNote.textContent = `Выбрано лицо ${index + 1}. Остальные области не будут добавлены.`;
        paint();
      };
      file.faces.forEach((face, index) => {
        const button = document.createElement('button'); button.type = 'button';
        button.className = 'face-choice-button'; button.textContent = `Лицо ${index + 1}`;
        button.setAttribute('aria-pressed', String(choices[fileIndex] === index));
        button.addEventListener('click', () => choose(index)); buttons.append(button);
      });
      image.onload = paint;
      image.onerror = () => { selectedNote.textContent = 'Не удалось показать снимок. Повторите проверку.'; buttons.querySelectorAll('button').forEach(button => {button.disabled = true;}); };
      image.src = file.image;
      item.append(canvas, buttons, selectedNote);
      if (choices[fileIndex] !== null) choose(choices[fileIndex]);
    }
    container.append(item);
  }
  container.hidden = false;
}
