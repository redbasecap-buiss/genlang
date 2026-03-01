//! Convergence Dashboard: generates a self-contained HTML page visualizing
//! evolution statistics (fitness curves, bloat, diversity).

use crate::genetic::GenStats;

/// Generate a self-contained HTML convergence dashboard from evolution stats.
pub fn generate_dashboard(stats: &[GenStats], title: &str) -> String {
    let gens: Vec<usize> = stats.iter().map(|s| s.generation).collect();
    let best_fit: Vec<f64> = stats.iter().map(|s| s.best_fitness).collect();
    let avg_fit: Vec<f64> = stats.iter().map(|s| s.avg_fitness).collect();
    let avg_size: Vec<f64> = stats.iter().map(|s| s.avg_size).collect();

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — genlang Convergence Dashboard</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; padding: 24px; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 8px; color: #58a6ff; }}
  .subtitle {{ color: #8b949e; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; max-width: 1200px; }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
  .card h2 {{ font-size: 1rem; color: #58a6ff; margin-bottom: 12px; }}
  canvas {{ width: 100%; height: 250px; }}
  .stats {{ display: flex; gap: 24px; margin-bottom: 16px; }}
  .stat {{ text-align: center; }}
  .stat-val {{ font-size: 1.5rem; font-weight: bold; color: #58a6ff; }}
  .stat-label {{ font-size: 0.8rem; color: #8b949e; }}
  .best-prog {{ font-family: monospace; background: #0d1117; padding: 8px; border-radius: 4px; overflow-x: auto; font-size: 0.85rem; color: #7ee787; margin-top: 8px; }}
  @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>🧬 {title}</h1>
<p class="subtitle">genlang Convergence Dashboard — {gens_count} generations</p>

<div class="stats">
  <div class="stat"><div class="stat-val">{gens_count}</div><div class="stat-label">Generations</div></div>
  <div class="stat"><div class="stat-val">{final_best:.6}</div><div class="stat-label">Final Best Fitness</div></div>
  <div class="stat"><div class="stat-val">{improvement:.1}%</div><div class="stat-label">Improvement</div></div>
  <div class="stat"><div class="stat-val">{final_size:.1}</div><div class="stat-label">Avg Tree Size</div></div>
</div>

<div class="best-prog">🏆 {best_program}</div>

<div class="grid">
  <div class="card">
    <h2>📉 Best Fitness</h2>
    <canvas id="fitChart"></canvas>
  </div>
  <div class="card">
    <h2>📊 Average Fitness</h2>
    <canvas id="avgChart"></canvas>
  </div>
  <div class="card">
    <h2>🌳 Average Tree Size (Bloat)</h2>
    <canvas id="sizeChart"></canvas>
  </div>
  <div class="card">
    <h2>📈 Fitness Ratio (Avg/Best)</h2>
    <canvas id="ratioChart"></canvas>
  </div>
</div>

<script>
const gens = {gens_json};
const bestFit = {best_fit_json};
const avgFit = {avg_fit_json};
const avgSize = {avg_size_json};

function drawChart(canvasId, data, color, label) {{
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.offsetWidth * 2;
  canvas.height = canvas.offsetHeight * 2;
  ctx.scale(2, 2);
  const w = canvas.offsetWidth, h = canvas.offsetHeight;
  const padding = {{ top: 10, right: 10, bottom: 25, left: 50 }};
  const pw = w - padding.left - padding.right;
  const ph = h - padding.top - padding.bottom;

  let min = Math.min(...data);
  let max = Math.max(...data);
  if (max - min < 1e-10) {{ max = min + 1; }}

  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 0.5;
  ctx.fillStyle = '#8b949e';
  ctx.font = '10px system-ui';
  ctx.textAlign = 'right';
  for (let i = 0; i <= 4; i++) {{
    const y = padding.top + ph - (i / 4) * ph;
    ctx.beginPath(); ctx.moveTo(padding.left, y); ctx.lineTo(w - padding.right, y); ctx.stroke();
    const val = min + (i / 4) * (max - min);
    ctx.fillText(val < 0.01 ? val.toExponential(1) : val.toFixed(2), padding.left - 4, y + 3);
  }}

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  for (let i = 0; i < data.length; i++) {{
    const x = padding.left + (i / Math.max(data.length - 1, 1)) * pw;
    const y = padding.top + ph - ((data[i] - min) / (max - min)) * ph;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  ctx.stroke();

  ctx.fillStyle = '#8b949e';
  ctx.textAlign = 'center';
  ctx.font = '10px system-ui';
  const step = Math.max(1, Math.floor(gens.length / 5));
  for (let i = 0; i < gens.length; i += step) {{
    const x = padding.left + (i / Math.max(gens.length - 1, 1)) * pw;
    ctx.fillText(gens[i], x, h - 5);
  }}
}}

const ratio = bestFit.map((b, i) => b > 1e-10 ? avgFit[i] / b : 1.0);

drawChart('fitChart', bestFit, '#58a6ff', 'Best Fitness');
drawChart('avgChart', avgFit, '#d2a8ff', 'Avg Fitness');
drawChart('sizeChart', avgSize, '#7ee787', 'Avg Size');
drawChart('ratioChart', ratio, '#f78166', 'Avg/Best Ratio');
</script>
</body>
</html>"#,
        title = title,
        gens_count = stats.len(),
        final_best = best_fit.last().copied().unwrap_or(0.0),
        improvement = if best_fit.len() >= 2 && best_fit[0] > 1e-10 {
            (1.0 - best_fit.last().unwrap() / best_fit[0]) * 100.0
        } else {
            0.0
        },
        final_size = avg_size.last().copied().unwrap_or(0.0),
        best_program = stats
            .last()
            .map(|s| s.best_program.as_str())
            .unwrap_or("N/A"),
        gens_json = serde_json::to_string(&gens).unwrap_or_default(),
        best_fit_json = serde_json::to_string(&best_fit).unwrap_or_default(),
        avg_fit_json = serde_json::to_string(&avg_fit).unwrap_or_default(),
        avg_size_json = serde_json::to_string(&avg_size).unwrap_or_default(),
    )
}

/// Write dashboard to a file. Returns Ok(()) on success.
pub fn save_dashboard(path: &str, stats: &[GenStats], title: &str) -> std::io::Result<()> {
    let html = generate_dashboard(stats, title);
    std::fs::write(path, html)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_stats() -> Vec<GenStats> {
        (0..10)
            .map(|i| GenStats {
                generation: i,
                best_fitness: 100.0 / (i as f64 + 1.0),
                avg_fitness: 200.0 / (i as f64 + 1.0),
                avg_size: 10.0 + i as f64,
                best_program: format!("(x0 + {})", i),
            })
            .collect()
    }

    #[test]
    fn test_generate_dashboard_html() {
        let stats = sample_stats();
        let html = generate_dashboard(&stats, "Test Run");
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Test Run"));
        assert!(html.contains("fitChart"));
        assert!(html.contains("bestFit"));
    }

    #[test]
    fn test_generate_dashboard_empty() {
        let html = generate_dashboard(&[], "Empty");
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("0 generations"));
    }

    #[test]
    fn test_save_dashboard() {
        let stats = sample_stats();
        let path = "/tmp/genlang_test_dashboard.html";
        save_dashboard(path, &stats, "Test").unwrap();
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("<!DOCTYPE html>"));
        std::fs::remove_file(path).ok();
    }
}
