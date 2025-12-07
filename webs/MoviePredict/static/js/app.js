/**
 * MoviePredict - Cinematic Dark Theme Logic
 */

const App = {
  init() {
    this.setupEventListeners();
    this.setupCharts();
    this.renderGenreChips();
    this.setupParticles();
    this.setupStoryCharts();
    this.setupScrollAnimations();
  },

  setupScrollAnimations() {
    const observerOptions = {
      threshold: 0.15,
      rootMargin: "0px 0px -50px 0px"
    };

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
        }
      });
    }, observerOptions);

    const elements = document.querySelectorAll('.reveal-on-scroll, .reveal-left, .reveal-right, .text-reveal, .timeline-item, .simulation-card');
    elements.forEach(el => observer.observe(el));
  },

  setupStoryCharts() {
    // Genre Distribution Chart (Doughnut)
    const genreCtx = document.getElementById('genreChart');
    if (genreCtx) {
      new Chart(genreCtx, {
        type: 'doughnut',
        data: {
          labels: ['Action', 'Drama', 'Comedy', 'Adventure', 'Horror', 'Thriller', 'Sci-Fi', 'Romance'],
          datasets: [{
            data: [25, 20, 18, 15, 8, 7, 5, 2],
            backgroundColor: [
              '#00f2ea', '#00c3ff', '#0080ff', '#e2b714',
              '#ff8c00', '#ff4d4d', '#bfa1a1', '#ffffff'
            ],
            borderWidth: 0,
            hoverOffset: 10
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'right',
              labels: {
                color: '#94a3b8',
                font: { family: "'JetBrains Mono', monospace", size: 11 },
                boxWidth: 12
              }
            }
          },
          cutout: '75%'
        }
      });
    }
  },

  setupNavbar() {
    const nav = document.querySelector('.glass-nav');
    const navLinks = document.querySelectorAll('.nav-links a');
    let lastScroll = 0;

    // Scroll behavior
    window.addEventListener('scroll', () => {
      const currentScroll = window.pageYOffset;

      if (currentScroll <= 0) {
        nav.classList.remove('hidden');
        return;
      }

      if (currentScroll > lastScroll && currentScroll > 100) {
        // Scrolling down
        nav.classList.add('hidden');
      } else {
        // Scrolling up
        nav.classList.remove('hidden');
      }

      lastScroll = currentScroll;
    });

    // Smooth scroll on click
    navLinks.forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href');
        const targetSection = document.querySelector(targetId);

        if (targetSection) {
          const offsetTop = targetSection.offsetTop - 100;
          window.scrollTo({
            top: offsetTop,
            behavior: 'smooth'
          });
        }
      });
    });
  },

  setupParticles() {
    const canvas = document.getElementById('particles-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let particles = [];

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    window.addEventListener('resize', resize);
    resize();

    class Particle {
      constructor() {
        this.x = Math.random() * canvas.width;
        this.y = Math.random() * canvas.height;
        this.vx = (Math.random() - 0.5) * 0.3;
        this.vy = (Math.random() - 0.5) * 0.3;
        this.size = Math.random() * 2;
        this.alpha = Math.random() * 0.5 + 0.1;
      }

      update() {
        this.x += this.vx;
        this.y += this.vy;

        if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
        if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
      }

      draw() {
        ctx.fillStyle = `rgba(0, 242, 234, ${this.alpha})`; // Electric Teal
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    const initParticles = () => {
      particles = [];
      const numberOfParticles = Math.min(window.innerWidth / 15, 80);
      for (let i = 0; i < numberOfParticles; i++) {
        particles.push(new Particle());
      }
    };

    initParticles();

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach((p, index) => {
        p.update();
        p.draw();
      });

      requestAnimationFrame(animate);
    };

    animate();
  },

  setupEventListeners() {
    this.setupNavbar();

    // Form Submission
    const form = document.getElementById('prediction-form');
    if (form) {
      form.addEventListener('submit', (e) => this.handlePrediction(e));
    }

    // Random Data
    const randomBtn = document.getElementById('randomDataBtn');
    if (randomBtn) {
      randomBtn.addEventListener('click', () => this.fillRandomData());
    }

    // Reset
    const resetBtn = document.getElementById('resetBtn');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        document.getElementById('results').classList.add('hidden');
        document.getElementById('predict').scrollIntoView({ behavior: 'smooth' });
        form.reset();
      });
    }

    // Mouse Glow Effect
    document.querySelectorAll('.glass-card').forEach(card => {
      card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        // Update CSS variables for glow effect if implemented in CSS
      });
    });
  },

  renderGenreChips() {
    const genres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Horror', 'Romance', 'Sci-Fi', 'Thriller'];
    const container = document.getElementById('genre-selector');
    const input = document.getElementById('genres');
    let selected = [];

    if (!container) return;

    genres.forEach(genre => {
      const chip = document.createElement('div');
      chip.className = 'genre-chip';
      chip.textContent = genre;
      chip.addEventListener('click', () => {
        chip.classList.toggle('active');
        if (selected.includes(genre)) {
          selected = selected.filter(g => g !== genre);
        } else {
          selected.push(genre);
        }
        input.value = selected.join(',');
      });
      container.appendChild(chip);
    });
  },

  async handlePrediction(e) {
    e.preventDefault();
    const btn = document.getElementById('submitBtn');
    const btnText = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.btn-loader');

    // Show loading state
    btnText.style.display = 'none';
    loader.style.display = 'inline-block';
    btn.disabled = true;

    try {
      const formData = new FormData(e.target);
      const data = Object.fromEntries(formData.entries());

      // Convert types for Pre-Release prediction
      data.budget = parseFloat(data.budget) || 0;
      data.runtime = parseInt(data.runtime) || 120;
      data.releaseMonth = parseInt(data.releaseMonth) || 6;
      data.genres = data.genres ? data.genres.split(',').filter(g => g.trim()) : [];

      // Call API
      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });

      const result = await response.json();

      // Show results section
      document.getElementById('results').style.display = 'block';
      document.getElementById('results').scrollIntoView({ behavior: 'smooth' });

      if (result.success) {
        this.showResults(result);
      } else {
        alert('Có lỗi xảy ra: ' + result.error);
      }

    } catch (error) {
      console.error('Error:', error);
      alert('Có lỗi xảy ra khi dự đoán: ' + error.message);
    } finally {
      // Reset button state
      btnText.style.display = 'inline-block';
      loader.style.display = 'none';
      btn.disabled = false;
    }
  },

  showResults(data) {
    const resultsSection = document.getElementById('results');
    resultsSection.classList.remove('hidden');

    // Scroll to results
    setTimeout(() => {
      resultsSection.scrollIntoView({ behavior: 'smooth' });
    }, 100);

    // Update UI Elements
    const movieTitle = document.getElementById('title').value;
    const resultTitleEl = document.getElementById('result-movie-title');
    if (resultTitleEl) {
      resultTitleEl.textContent = movieTitle ? `PROJECT: ${movieTitle.toUpperCase()}` : 'PROJECT: UNTITLED';
    }

    const prediction = data.prediction;
    const metrics = data.metrics;
    const prob = prediction.success_probability;

    // Update Gauge
    this.renderGauge(prob);

    // Update Text
    const probPercent = Math.round(prob * 100);
    document.getElementById('confidence-value').innerText = probPercent + '%';

    // Status Text
    const statusEl = document.getElementById('prediction-status');
    const descEl = document.getElementById('prediction-desc');

    // Determine State
    let color = '#e2b714'; // Default warning
    if (prob >= 0.6) {
      color = '#00f2ea';
      statusEl.innerHTML = `<i class="fas fa-check-circle" style="color: ${color}"></i> <span style="color: ${color}">BLOCKBUSTER POTENTIAL</span>`;
      descEl.textContent = `Dự án có tín hiệu rất tích cực. Các chỉ số cho thấy khả năng sinh lời cao.`;
    } else if (prob <= 0.4) {
      color = '#ff4d4d';
      statusEl.innerHTML = `<i class="fas fa-times-circle" style="color: ${color}"></i> <span style="color: ${color}">HIGH RISK</span>`;
      descEl.textContent = `Cảnh báo: Dự án có rủi ro cao. Cần cân nhắc lại ngân sách hoặc chiến lược.`;
    } else {
      statusEl.innerHTML = `<i class="fas fa-exclamation-circle" style="color: ${color}"></i> <span style="color: ${color}">AVERAGE</span>`;
      descEl.textContent = `Tiềm năng ở mức trung bình. Thành công phụ thuộc vào yếu tố thị trường.`;
    }

    // Update Metrics Values for Pre-Release
    const estimatedRoi = metrics.estimated_roi || metrics.roi_category || 'N/A';
    const riskScore = metrics.risk_score || metrics.risk_level || 'N/A';
    const successScore = metrics.success_score || Math.round(prob * 100) || 0;

    // Display ROI as multiplier or category
    const roiDisplay = typeof estimatedRoi === 'number' ? `${estimatedRoi}x` : estimatedRoi;
    document.getElementById('roi-value').textContent = roiDisplay;

    // Display risk level
    const riskDisplay = typeof riskScore === 'number' ? `${riskScore}%` : riskScore;
    document.getElementById('risk-value').textContent = riskDisplay;

    // Display success score with animated counter
    const revenueEl = document.getElementById('revenue-value');
    if (typeof successScore === 'number') {
      this.animateCounter(revenueEl, successScore, 1200, '', '%');
    } else {
      revenueEl.textContent = `${successScore}%`;
    }

    // Render Charts
    this.renderFeatureChart(data.feature_importance);
    this.renderRadarChart(data.input_data);

    // Update Comparison Bars with user input data
    const budget = data.input_data.budget || 0;
    const runtime = data.input_data.runtime || 0;
    const genreCount = data.input_data.genres?.length || 1;
    this.updateComparisonBars(budget, runtime, genreCount);
  },

  renderRadarChart(inputData) {
    const ctx = document.getElementById('radarChart').getContext('2d');

    if (this.radarChartInstance) {
      this.radarChartInstance.destroy();
    }

    // Pre-Release benchmarks (no revenue/vote_average)
    const benchmark = {
      budget: 150000000, // $150M
      runtime: 130,      // 130 mins
      genres: 3,         // 3 genres  
      releaseMonth: 6    // Summer release
    };

    // Calculate scores for Pre-Release features
    const budgetScore = Math.min((inputData.budget / benchmark.budget) * 100, 100);
    const runtimeScore = Math.min((inputData.runtime / benchmark.runtime) * 100, 100);
    const genreScore = Math.min(((inputData.genres?.length || 1) / benchmark.genres) * 100, 100);

    // Release timing score (summer/holiday months get higher scores)
    const releaseMonth = inputData.release_month || inputData.releaseMonth || 6;
    const isGoodMonth = [6, 7, 11, 12].includes(releaseMonth);
    const timingScore = isGoodMonth ? 90 : 60;

    // Overall potential based on budget and genres
    const potentialScore = (budgetScore * 0.5 + genreScore * 0.3 + timingScore * 0.2);

    const dataValues = [
      budgetScore,
      runtimeScore,
      genreScore,
      timingScore,
      potentialScore
    ];

    this.radarChartInstance = new Chart(ctx, {
      type: 'radar',
      data: {
        labels: ['Budget Power', 'Runtime Fit', 'Genre Diversity', 'Release Timing', 'Overall Potential'],
        datasets: [{
          label: 'Your Movie',
          data: dataValues,
          backgroundColor: 'rgba(0, 229, 255, 0.2)',
          borderColor: '#00E5FF',
          pointBackgroundColor: '#00E5FF',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: '#00E5FF'
        }, {
          label: 'Blockbuster Avg',
          data: [80, 85, 70, 90, 80], // Ideal shape
          backgroundColor: 'rgba(255, 215, 0, 0.1)',
          borderColor: '#FFD700',
          pointBackgroundColor: '#FFD700',
          pointBorderColor: '#fff',
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: '#FFD700',
          borderDash: [5, 5]
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
            grid: { color: 'rgba(255, 255, 255, 0.1)' },
            pointLabels: {
              color: '#fff',
              font: { family: "'Outfit', sans-serif", size: 12 }
            },
            ticks: { display: false, backdropColor: 'transparent' },
            suggestedMin: 0,
            suggestedMax: 100
          }
        },
        plugins: {
          legend: {
            labels: { color: '#fff', font: { family: "'Outfit', sans-serif" } }
          }
        }
      }
    });
  },

  renderGauge(probability) {
    const ctx = document.getElementById('predictionGauge').getContext('2d');

    if (this.gaugeChart) {
      this.gaugeChart.destroy();
    }

    const value = probability * 100;
    const color = probability >= 0.6 ? '#00f2ea' : (probability <= 0.4 ? '#ff4d4d' : '#e2b714');

    this.gaugeChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Success', 'Remaining'],
        datasets: [{
          data: [value, 100 - value],
          backgroundColor: [
            color,
            'rgba(255, 255, 255, 0.05)'
          ],
          borderWidth: 0,
          borderRadius: 20,
          cutout: '85%',
          circumference: 360,
          rotation: 0,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false }
        },
        animation: {
          animateScale: true,
          animateRotate: true,
          duration: 1500,
          easing: 'easeOutQuart'
        }
      }
    });
  },

  renderFeatureChart(featureData) {
    const ctx = document.getElementById('featureChart').getContext('2d');

    if (this.featureChart) {
      this.featureChart.destroy();
    }

    // Handle both formats: array directly or object with top_features
    let features = [];
    if (Array.isArray(featureData)) {
      // New Pre-Release format: array of {feature, importance}
      features = featureData;
    } else if (featureData && featureData.top_features) {
      // Old format: object with top_features
      features = featureData.top_features;
    }

    if (!features || features.length === 0) {
      console.warn('No feature data available');
      return;
    }

    // Process data - handle both naming conventions
    const labels = features.map(f => f.feature || f.name || 'Unknown');
    const values = features.map(f => f.importance || 0);

    this.featureChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Importance (%)',
          data: values,
          backgroundColor: 'rgba(0, 242, 234, 0.6)',
          borderColor: '#00f2ea',
          borderWidth: 1,
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(10, 14, 23, 0.9)',
            titleColor: '#fff',
            bodyColor: '#94a3b8',
            borderColor: 'rgba(255, 255, 255, 0.1)',
            borderWidth: 1
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8', font: { family: "'JetBrains Mono', monospace" } }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#ffffff', font: { family: "'JetBrains Mono', monospace" } }
          }
        }
      }
    });
  },

  setupCharts() {
    // Global Chart Defaults for Dark Theme
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
    Chart.defaults.font.family = "'JetBrains Mono', monospace";
  },

  fillRandomData() {
    // Pre-Release random data (no revenue or vote_average)
    document.getElementById('title').value = 'Project Alpha ' + Math.floor(Math.random() * 100);
    document.getElementById('budget').value = Math.floor(Math.random() * 50000000) + 1000000;
    document.getElementById('runtime').value = Math.floor(Math.random() * 60) + 90;
    document.getElementById('releaseMonth').value = Math.floor(Math.random() * 12) + 1;

    // Random genres
    const allGenres = ['Action', 'Adventure', 'Comedy', 'Drama', 'Fantasy', 'Horror', 'Romance', 'Sci-Fi', 'Thriller'];
    const numGenres = Math.floor(Math.random() * 3) + 1;
    const selectedGenres = allGenres.sort(() => 0.5 - Math.random()).slice(0, numGenres);

    // Update genre chips
    document.querySelectorAll('.genre-chip').forEach(chip => {
      chip.classList.remove('active');
      if (selectedGenres.includes(chip.textContent)) {
        chip.classList.add('active');
      }
    });
    document.getElementById('genres').value = selectedGenres.join(',');
  },

  // === NEW: 3D Tilt Effect cho Movie Cards ===
  setup3DTilt() {
    const cards = document.querySelectorAll('.movie-card');
    cards.forEach(card => {
      card.addEventListener('mousemove', (e) => {
        const rect = card.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const centerX = rect.width / 2;
        const centerY = rect.height / 2;
        const tiltX = (x - centerX) / centerX * 10;
        const tiltY = (centerY - y) / centerY * 10;

        card.style.setProperty('--tilt-x', `${tiltX}deg`);
        card.style.setProperty('--tilt-y', `${tiltY}deg`);
        card.classList.add('tilt-active');
      });

      card.addEventListener('mouseleave', () => {
        card.classList.remove('tilt-active');
        card.style.setProperty('--tilt-x', '0deg');
        card.style.setProperty('--tilt-y', '0deg');
      });
    });
  },

  // === NEW: Animated Number Counters ===
  animateCounter(element, target, duration = 1000, prefix = '', suffix = '') {
    const start = 0;
    const startTime = performance.now();

    const update = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const easeProgress = 1 - Math.pow(1 - progress, 3); // easeOutCubic
      const current = Math.floor(start + (target - start) * easeProgress);

      element.textContent = prefix + current.toLocaleString() + suffix;
      element.classList.add('counting');

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        element.classList.remove('counting');
      }
    };

    requestAnimationFrame(update);
  },

  // === NEW: Parallax Scrolling ===
  setupParallax() {
    const layers = document.querySelectorAll('.parallax-layer');
    if (layers.length === 0) return;

    window.addEventListener('scroll', () => {
      const scrollY = window.pageYOffset;

      layers.forEach((layer, index) => {
        const speed = 0.1 + (index * 0.05);
        const yPos = scrollY * speed;
        layer.style.transform = `translateY(${yPos}px)`;
      });
    });
  },

  // === NEW: Init Data Viz Charts ===
  initDataVizCharts() {
    this.setupBudgetScatterChart();
    this.setupGenreHeatmap();
    this.setupMonthlyTimeline();
  },

  // Budget vs Success Scatter Chart
  setupBudgetScatterChart() {
    const ctx = document.getElementById('budgetScatterChart');
    if (!ctx) return;

    // Sample data simulating budget vs success
    const data = [
      { x: 5, y: 30, success: false },
      { x: 15, y: 45, success: false },
      { x: 25, y: 55, success: true },
      { x: 50, y: 65, success: true },
      { x: 80, y: 70, success: true },
      { x: 100, y: 75, success: true },
      { x: 150, y: 78, success: true },
      { x: 200, y: 82, success: true },
      { x: 10, y: 35, success: false },
      { x: 30, y: 50, success: false },
      { x: 45, y: 60, success: true },
      { x: 70, y: 72, success: true },
      { x: 120, y: 68, success: true },
      { x: 180, y: 85, success: true },
    ];

    new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'Thành công',
          data: data.filter(d => d.success),
          backgroundColor: 'rgba(0, 245, 212, 0.6)',
          borderColor: '#00f5d4',
          borderWidth: 2,
          pointRadius: 8,
          pointHoverRadius: 12
        }, {
          label: 'Thất bại',
          data: data.filter(d => !d.success),
          backgroundColor: 'rgba(255, 71, 87, 0.6)',
          borderColor: '#ff4757',
          borderWidth: 2,
          pointRadius: 8,
          pointHoverRadius: 12
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: '#94a3b8',
              font: { size: 10 },
              boxWidth: 10
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Budget (M$)', color: '#6b7c93' },
            grid: { color: 'rgba(255,255,255,0.05)' }
          },
          y: {
            title: { display: true, text: 'Success Score', color: '#6b7c93' },
            grid: { color: 'rgba(255,255,255,0.05)' },
            min: 0,
            max: 100
          }
        }
      }
    });
  },

  // Genre Heatmap
  setupGenreHeatmap() {
    const container = document.getElementById('genreHeatmap');
    if (!container) return;

    const genres = [
      { name: 'Action', rate: 72 },
      { name: 'Adventure', rate: 68 },
      { name: 'Comedy', rate: 55 },
      { name: 'Drama', rate: 48 },
      { name: 'Fantasy', rate: 65 },
      { name: 'Horror', rate: 52 },
      { name: 'Sci-Fi', rate: 70 },
      { name: 'Thriller', rate: 58 },
      { name: 'Romance', rate: 45 },
      { name: 'Animation', rate: 75 },
      { name: 'Crime', rate: 50 },
      { name: 'Family', rate: 62 }
    ];

    container.innerHTML = genres.map(g => {
      const hue = (g.rate / 100) * 120; // 0=red, 120=green
      const color = `hsl(${hue}, 70%, 50%)`;
      return `
        <div class="heatmap-cell" style="background: ${color}">
          <span>${g.rate}%</span>
          <div class="heatmap-cell-tooltip">${g.name}: ${g.rate}% thành công</div>
        </div>
      `;
    }).join('');
  },

  // Monthly Timeline
  setupMonthlyTimeline() {
    const container = document.getElementById('monthlyTimeline');
    if (!container) return;

    const months = [
      { name: 'Jan', value: 45, golden: false },
      { name: 'Feb', value: 52, golden: false },
      { name: 'Mar', value: 58, golden: false },
      { name: 'Apr', value: 50, golden: false },
      { name: 'May', value: 65, golden: false },
      { name: 'Jun', value: 82, golden: true },
      { name: 'Jul', value: 85, golden: true },
      { name: 'Aug', value: 68, golden: false },
      { name: 'Sep', value: 55, golden: false },
      { name: 'Oct', value: 60, golden: false },
      { name: 'Nov', value: 78, golden: true },
      { name: 'Dec', value: 80, golden: true }
    ];

    container.innerHTML = months.map(m => `
      <div class="month-bar ${m.golden ? 'golden' : ''}" style="height: 100%">
        <div class="month-bar-fill" style="height: ${m.value}%"></div>
        <span class="month-label">${m.name}</span>
        <span class="month-value">${m.value}%</span>
      </div>
    `).join('');
  },

  // === NEW: Update Comparison Bars khi predict ===
  updateComparisonBars(budget, runtime, genreCount) {
    // Average values (based on dataset)
    const avgBudget = 40000000;
    const avgRuntime = 110;
    const avgGenres = 2;

    const budgetPercent = Math.min((budget / avgBudget) * 60, 100);
    const runtimePercent = Math.min((runtime / avgRuntime) * 70, 100);
    const genrePercent = Math.min((genreCount / avgGenres) * 50, 100);

    const cmpBudget = document.getElementById('cmp-budget');
    const cmpRuntime = document.getElementById('cmp-runtime');
    const cmpGenres = document.getElementById('cmp-genres');

    if (cmpBudget) cmpBudget.style.width = `${budgetPercent}%`;
    if (cmpRuntime) cmpRuntime.style.width = `${runtimePercent}%`;
    if (cmpGenres) cmpGenres.style.width = `${genrePercent}%`;
  }
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  App.init();
  App.setup3DTilt();
  App.setupParallax();
  App.initDataVizCharts();
});
