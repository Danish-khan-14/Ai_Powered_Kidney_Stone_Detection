// Wait for DOM to load
document.addEventListener('DOMContentLoaded', function() {
    initializeTheme();
    initializeMobileMenu();
    initializeScrollAnimations();
    initializeUploadArea();
    checkLoginStatus();
});

// ===== Theme Toggle =====
function initializeTheme() {
    const themeToggle = document.getElementById('themeToggle');
    if (!themeToggle) return;

    // Set initial theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);

    // Toggle theme on click
    themeToggle.addEventListener('click', function(e) {
        e.preventDefault();
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        setTheme(newTheme);
    });
}

function setTheme(theme) {
    const themeToggle = document.getElementById('themeToggle');
    if (!themeToggle) return;

    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
    
    // Update icon
    themeToggle.innerHTML = theme === 'dark' 
        ? '<i class="fas fa-sun"></i>' 
        : '<i class="fas fa-moon"></i>';
}

// ===== Mobile Menu =====
function initializeMobileMenu() {
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const mobileMenu = document.getElementById('mobileMenu');

    if (mobileMenuBtn && mobileMenu) {
        mobileMenuBtn.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            mobileMenu.classList.toggle('active');
        });

        // Close menu when clicking a link
        mobileMenu.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', function() {
                mobileMenu.classList.remove('active');
            });
        });

        // Close menu when clicking outside
        document.addEventListener('click', function(e) {
            if (!mobileMenu.contains(e.target) && !mobileMenuBtn.contains(e.target)) {
                mobileMenu.classList.remove('active');
            }
        });
    }
}

// ===== Scroll Animations =====
function initializeScrollAnimations() {
    const animatedElements = document.querySelectorAll(
        '.feature-card, .stat-large-card, .contact-item, .contact-form, .about-content, .step-card'
    );

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                // Optional: unobserve after animation
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    animatedElements.forEach(el => observer.observe(el));
}

// ===== Upload Area =====
function initializeUploadArea() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    if (!uploadArea || !fileInput) return;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight() {
        uploadArea.classList.add('highlight');
        uploadArea.style.background = 'var(--bg)';
    }

    function unhighlight() {
        uploadArea.classList.remove('highlight');
        uploadArea.style.background = 'none';
    }

    // Handle dropped files
    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        handleFile(file);
    }

    // Handle click on upload area
    uploadArea.addEventListener('click', function() {
        fileInput.click();
    });

    // Handle file selection
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        handleFile(file);
    });
}

// ===== Handle File Upload =====
function handleFile(file) {
    if (!file) return;

    // Check file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (JPG, PNG)');
        return;
    }

    // Check file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewImage = document.getElementById('previewImage');
        const uploadArea = document.getElementById('uploadArea');
        const previewArea = document.getElementById('previewArea');
        
        if (previewImage && uploadArea && previewArea) {
            previewImage.src = e.target.result;
            uploadArea.style.display = 'none';
            previewArea.style.display = 'block';
        }
    };
    reader.readAsDataURL(file);
}

// ===== Analyze Image =====
async function analyzeImage() {
    const analyzeBtn = document.querySelector('.analyze-btn');
    const fileInput = document.getElementById('fileInput');
    
    if (!analyzeBtn || !fileInput || !fileInput.files[0]) return;

    const originalText = analyzeBtn.innerHTML;
    analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
    analyzeBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('image', fileInput.files[0]);

        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (response.status === 401) {
            window.location.href = '/login';
            return;
        }

        const data = await response.json();

        if (!data.success) {
            alert('Error: ' + (data.error || 'Failed to analyze image'));
            return;
        }

        // Display results
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing image. Please try again.');
    } finally {
        analyzeBtn.innerHTML = originalText;
        analyzeBtn.disabled = false;
    }
}

// ===== Display Results =====
function displayResults(data) {
    const previewArea = document.getElementById('previewArea');
    const resultsArea = document.getElementById('resultsArea');
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultDescription = document.getElementById('resultDescription');
    const confidenceValue = document.getElementById('confidenceValue');
    const processingTime = document.getElementById('processingTime');

    if (!resultsArea) return;

    // Update result icon and title
    if (data.result === 1) {
        resultIcon.innerHTML = '<i class="fas fa-exclamation-triangle" style="color: #ef4444;"></i>';
        resultTitle.textContent = '⚠️ ' + data.result_text;
        resultTitle.style.color = '#ef4444';
    } else {
        resultIcon.innerHTML = '<i class="fas fa-check-circle" style="color: #10b981;"></i>';
        resultTitle.textContent = '✅ ' + data.result_text;
        resultTitle.style.color = '#10b981';
    }

    // Update metrics
    confidenceValue.textContent = data.confidence_percent.toFixed(1) + '%';
    processingTime.textContent = data.processing_time + 's';
    
    // Update description
    resultDescription.textContent = data.description;

    // Update progress ring if it exists
    updateProgressRing(data.confidence_percent);

    // Show results
    if (previewArea) previewArea.style.display = 'none';
    resultsArea.style.display = 'block';
}

// ===== Update Progress Ring =====
function updateProgressRing(confidence) {
    const circle = document.getElementById('confidenceProgress');
    if (!circle) return;

    const radius = 54;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (confidence / 100) * circumference;
    
    circle.style.strokeDasharray = `${circumference}`;
    circle.style.strokeDashoffset = offset;
}

// ===== Reset Functions =====
function resetUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const previewArea = document.getElementById('previewArea');
    const fileInput = document.getElementById('fileInput');
    
    if (uploadArea) uploadArea.style.display = 'block';
    if (previewArea) previewArea.style.display = 'none';
    if (fileInput) fileInput.value = '';
}

function resetAll() {
    resetUpload();
    const resultsArea = document.getElementById('resultsArea');
    if (resultsArea) resultsArea.style.display = 'none';
}

// ===== Check Login Status =====
// NOTE: Your Flask /health endpoint must return these fields when authenticated:
//   { authenticated: true, user_name: "Danish", user_email: "...", stats: { total_detections, stones_detected, normal_results } }
async function checkLoginStatus() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        updateNavbarForAuth(data.authenticated, data.user_name || '', data.user_email || '', data.stats || {});
    } catch (error) {
        console.error('Error checking login status:', error);
    }
}

function updateNavbarForAuth(isAuthenticated, userName, userEmail, stats) {
    const authButtons   = document.getElementById('authButtons');
    const userMenu      = document.getElementById('userMenu');
    const mobileAuth    = document.getElementById('mobileAuth');
    const mobileUser    = document.getElementById('mobileUser');
    const heroStartBtn  = document.getElementById('heroStartBtn');
    const heroPlayBtn   = document.getElementById('heroPlayBtn');

    // Greeting elements
    const navGreeting       = document.getElementById('navGreeting');
    const dropdownName      = document.getElementById('dropdownName');
    const dropdownEmail     = document.getElementById('dropdownEmail');
    const heroGreeting      = document.getElementById('heroGreeting');
    const heroGreetingText  = document.getElementById('heroGreetingText');
    const heroBadge         = document.getElementById('heroBadge');

    // Profile dropdown stats
    const pdTotal   = document.getElementById('pdTotal');
    const pdStones  = document.getElementById('pdStones');
    const pdNormal  = document.getElementById('pdNormal');

    const displayName = userName || 'User';

    if (isAuthenticated) {
        // --- Navbar ---
        if (authButtons) authButtons.style.display = 'none';
        if (userMenu)    userMenu.style.display = 'flex';
        if (mobileAuth)  mobileAuth.style.display = 'none';
        if (mobileUser)  mobileUser.style.display = 'block';

        // Greeting text in navbar
        if (navGreeting)  navGreeting.textContent = 'Hello, ' + displayName;

        // Profile dropdown info
        if (dropdownName)  dropdownName.textContent = displayName;
        if (dropdownEmail) dropdownEmail.textContent = userEmail || '—';
        if (pdTotal)  pdTotal.textContent  = stats.total_detections  ?? '—';
        if (pdStones) pdStones.textContent = stats.stones_detected   ?? '—';
        if (pdNormal) pdNormal.textContent = stats.normal_results    ?? '—';

        // --- Hero personalised greeting ---
        if (heroGreeting && heroGreetingText && heroBadge) {
            heroGreetingText.textContent = '👋 Hello, ' + displayName + '! Welcome back';
            heroGreeting.style.display = 'inline-flex';
            heroBadge.style.display = 'none';
        }

        // CTA buttons all go to /upload
        if (heroStartBtn) {
            heroStartBtn.href = '/upload';
            heroStartBtn.innerHTML = '<i class="fas fa-microscope"></i> Start Detection';
        }
        if (heroPlayBtn) heroPlayBtn.onclick = () => window.location.href = '/upload';

        document.querySelectorAll('.feature-link').forEach(link => { link.href = '/upload'; });

    } else {
        // --- Logged out ---
        if (authButtons) authButtons.style.display = 'flex';
        if (userMenu)    userMenu.style.display = 'none';
        if (mobileAuth)  mobileAuth.style.display = 'block';
        if (mobileUser)  mobileUser.style.display = 'none';

        if (heroBadge)    heroBadge.style.display = 'inline-flex';
        if (heroGreeting) heroGreeting.style.display = 'none';

        if (heroStartBtn) {
            heroStartBtn.href = '/login';
            heroStartBtn.innerHTML = '<i class="fas fa-microscope"></i> Start Detection';
        }
        if (heroPlayBtn) heroPlayBtn.onclick = () => window.location.href = '/login';

        document.querySelectorAll('.feature-link').forEach(link => { link.href = '/login'; });
    }
}

// ===== Profile Dropdown Toggle =====
document.addEventListener('click', function(e) {
    const btn      = document.getElementById('profileAvatarBtn');
    const dropdown = document.getElementById('profileDropdown');
    if (!btn || !dropdown) return;

    if (btn.contains(e.target)) {
        dropdown.classList.toggle('open');
    } else if (!dropdown.contains(e.target)) {
        dropdown.classList.remove('open');
    }
});
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        if (href === '#') return;
        
        e.preventDefault();
        const target = document.querySelector(href);
        if (target) {
            target.scrollIntoView({ 
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});