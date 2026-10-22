export function drawThresholdChart(canvas, calibration) {
  const points = calibration?.points || [];
  const rect = canvas.getBoundingClientRect();
  if (!points.length || !rect.width) return;
  const width = Math.round(rect.width), height = Math.round(rect.height || 280);
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * scale); canvas.height = Math.round(height * scale);
  const ctx = canvas.getContext('2d'); ctx.scale(scale, scale);
  const left = 42, right = 14, top = 20, bottom = 45;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const low = points[0].threshold, high = points[points.length - 1].threshold;
  const x = value => left + (value - low) / (high - low || 1) * plotWidth;
  const y = value => top + (1 - value) * plotHeight;
  ctx.font = '11px "Segoe UI", sans-serif';
  for (const tick of [0, .25, .5, .75, 1]) {
    ctx.strokeStyle = '#e0e6d7'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(left, y(tick)); ctx.lineTo(width - right, y(tick)); ctx.stroke();
    ctx.fillStyle = '#839275'; ctx.textAlign = 'right'; ctx.fillText(`${Math.round(tick * 100)}%`, left - 9, y(tick) + 4);
  }
  ctx.textAlign = 'center';
  const ticks = width < 500 ? [low, (low + high) / 2, high] : [low, low+(high-low)/4, (low+high)/2, low+3*(high-low)/4, high];
  for (const tick of ticks) ctx.fillText(tick.toFixed(2), x(tick), height - 22);
  ctx.fillText('Порог узнавания', left + plotWidth / 2, height - 3);
  for (const [key, color] of [['accuracy', '#275946'], ['false_accept_rate', '#bd7144'], ['false_reject_rate', '#657bb4']]) {
    ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.beginPath(); let connected = false;
    for (const point of points) {
      const value = point.metrics[key];
      if (value === null) { connected = false; continue; }
      if (connected) ctx.lineTo(x(point.threshold), y(value)); else ctx.moveTo(x(point.threshold), y(value));
      connected = true;
    }
    ctx.stroke();
  }
  if (calibration.recommended_threshold !== null) {
    ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = '#91a27b'; ctx.lineWidth = 1;
    const position = x(calibration.recommended_threshold);
    ctx.beginPath(); ctx.moveTo(position, top); ctx.lineTo(position, top + plotHeight); ctx.stroke(); ctx.restore();
  }
}
