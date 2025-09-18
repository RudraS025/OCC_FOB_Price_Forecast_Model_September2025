/**
 * OCC Price Forecasting Application
 * Custom JavaScript functionality
 */

// Global application object
const OCCApp = {
    // Configuration
    config: {
        apiTimeout: 30000,
        refreshInterval: 300000, // 5 minutes
        chartColors: {
            primary: '#2E86AB',
            secondary: '#A23B72',
            accent: '#F18F01',
            success: '#C73E1D',
            neutral: '#6C757D'
        }
    },
    
    // Application state
    state: {
        isLoading: false,
        lastUpdate: null,
        currentPage: null
    },
    
    // Initialize the application
    init: function() {
        this.setupEventListeners();
        this.setupAutoRefresh();
        this.detectCurrentPage();
        this.addAnimations();
        console.log('OCC Price Forecasting App initialized');
    },
    
    // Setup global event listeners
    setupEventListeners: function() {
        // Handle form submissions
        document.addEventListener('submit', this.handleFormSubmit.bind(this));
        
        // Handle navigation
        document.addEventListener('click', this.handleNavigation.bind(this));
        
        // Handle keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this));
        
        // Handle window resize
        window.addEventListener('resize', this.handleWindowResize.bind(this));
        
        // Handle visibility change for auto-refresh
        document.addEventListener('visibilitychange', this.handleVisibilityChange.bind(this));
    },
    
    // Detect current page for specific functionality
    detectCurrentPage: function() {
        const path = window.location.pathname;
        if (path === '/' || path === '/index') {
            this.state.currentPage = 'dashboard';
        } else if (path === '/forecast') {
            this.state.currentPage = 'forecast';
        } else if (path === '/update_data') {
            this.state.currentPage = 'update_data';
        } else if (path === '/analytics') {
            this.state.currentPage = 'analytics';
        }
    },
    
    // Add CSS animations to elements
    addAnimations: function() {
        // Add fade-in animation to cards
        const cards = document.querySelectorAll('.card');
        cards.forEach((card, index) => {
            setTimeout(() => {
                card.classList.add('fade-in');
            }, index * 100);
        });
        
        // Add slide-in animations to navigation items
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach((item, index) => {
            setTimeout(() => {
                item.classList.add('slide-in-right');
            }, index * 50);
        });
    },
    
    // Handle form submissions
    handleFormSubmit: function(event) {
        const form = event.target;
        if (form.classList.contains('needs-validation')) {
            event.preventDefault();
            if (this.validateForm(form)) {
                this.submitForm(form);
            }
        }
    },
    
    // Handle navigation clicks
    handleNavigation: function(event) {
        const link = event.target.closest('a[data-navigate]');
        if (link) {
            event.preventDefault();
            this.navigateTo(link.href);
        }
    },
    
    // Handle keyboard shortcuts
    handleKeyboardShortcuts: function(event) {
        // Ctrl/Cmd + Enter for quick forecast
        if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
            if (this.state.currentPage === 'forecast') {
                const forecastForm = document.getElementById('forecast-form');
                if (forecastForm) {
                    forecastForm.dispatchEvent(new Event('submit'));
                }
            }
        }
        
        // F5 for refresh (override default)
        if (event.key === 'F5') {
            event.preventDefault();
            this.refreshCurrentPage();
        }
        
        // Escape to close modals
        if (event.key === 'Escape') {
            this.closeAllModals();
        }
    },
    
    // Handle window resize
    handleWindowResize: function() {
        // Redraw charts if they exist
        if (window.Chart) {
            Chart.helpers.each(Chart.instances, function(instance) {
                instance.resize();
            });
        }
    },
    
    // Handle visibility change
    handleVisibilityChange: function() {
        if (document.hidden) {
            this.pauseAutoRefresh();
        } else {
            this.resumeAutoRefresh();
        }
    },
    
    // Form validation
    validateForm: function(form) {
        let isValid = true;
        const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
        
        inputs.forEach(input => {
            if (!input.value.trim()) {
                this.showFieldError(input, 'This field is required');
                isValid = false;
            } else {
                this.clearFieldError(input);
            }
        });
        
        return isValid;
    },
    
    // Show field error
    showFieldError: function(field, message) {
        field.classList.add('is-invalid');
        let errorDiv = field.parentNode.querySelector('.invalid-feedback');
        if (!errorDiv) {
            errorDiv = document.createElement('div');
            errorDiv.className = 'invalid-feedback';
            field.parentNode.appendChild(errorDiv);
        }
        errorDiv.textContent = message;
    },
    
    // Clear field error
    clearFieldError: function(field) {
        field.classList.remove('is-invalid');
        const errorDiv = field.parentNode.querySelector('.invalid-feedback');
        if (errorDiv) {
            errorDiv.remove();
        }
    },
    
    // Submit form via AJAX
    submitForm: function(form) {
        this.showLoading();
        
        const formData = new FormData(form);
        const url = form.action || window.location.pathname;
        
        fetch(url, {
            method: 'POST',
            body: formData,
            headers: {
                'X-Requested-With': 'XMLHttpRequest'
            }
        })
        .then(response => response.json())
        .then(data => {
            this.hideLoading();
            this.handleFormResponse(data);
        })
        .catch(error => {
            this.hideLoading();
            this.showError('Network error: ' + error.message);
        });
    },
    
    // Handle form response
    handleFormResponse: function(data) {
        if (data.success) {
            this.showSuccess(data.message || 'Operation completed successfully');
            if (data.redirect) {
                setTimeout(() => {
                    window.location.href = data.redirect;
                }, 1500);
            }
        } else {
            this.showError(data.error || 'An error occurred');
        }
    },
    
    // Navigation with loading state
    navigateTo: function(url) {
        this.showLoading();
        window.location.href = url;
    },
    
    // Refresh current page
    refreshCurrentPage: function() {
        this.showLoading();
        window.location.reload();
    },
    
    // Auto-refresh functionality
    setupAutoRefresh: function() {
        this.autoRefreshTimer = setInterval(() => {
            if (!document.hidden && !this.state.isLoading) {
                this.checkForUpdates();
            }
        }, this.config.refreshInterval);
    },
    
    pauseAutoRefresh: function() {
        if (this.autoRefreshTimer) {
            clearInterval(this.autoRefreshTimer);
        }
    },
    
    resumeAutoRefresh: function() {
        this.setupAutoRefresh();
    },
    
    // Check for data updates
    checkForUpdates: function() {
        fetch('/api/data_summary')
        .then(response => response.json())
        .then(data => {
            const lastUpdate = data.metadata?.last_updated;
            if (lastUpdate && lastUpdate !== this.state.lastUpdate) {
                this.state.lastUpdate = lastUpdate;
                this.showUpdateNotification();
            }
        })
        .catch(error => {
            console.log('Background update check failed:', error);
        });
    },
    
    // Show update notification
    showUpdateNotification: function() {
        const notification = document.createElement('div');
        notification.className = 'alert alert-info alert-dismissible fade show position-fixed';
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 300px;';
        notification.innerHTML = `
            <i class="fas fa-info-circle me-2"></i>
            New data available. <a href="#" onclick="OCCApp.refreshCurrentPage()">Refresh page</a>
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        document.body.appendChild(notification);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 10000);
    },
    
    // Loading state management
    showLoading: function(message = 'Loading...') {
        this.state.isLoading = true;
        
        // Create loading overlay
        let overlay = document.getElementById('loading-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'loading-overlay';
            overlay.className = 'position-fixed top-0 start-0 w-100 h-100 d-flex align-items-center justify-content-center';
            overlay.style.cssText = 'background: rgba(0,0,0,0.7); z-index: 9999; backdrop-filter: blur(5px);';
            overlay.innerHTML = `
                <div class="text-center text-white">
                    <div class="spinner-border mb-3" style="width: 3rem; height: 3rem;" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <h5 id="loading-message">${message}</h5>
                </div>
            `;
            document.body.appendChild(overlay);
        } else {
            overlay.style.display = 'flex';
            document.getElementById('loading-message').textContent = message;
        }
    },
    
    hideLoading: function() {
        this.state.isLoading = false;
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.style.display = 'none';
        }
    },
    
    // Notification management
    showSuccess: function(message) {
        this.showNotification(message, 'success');
    },
    
    showError: function(message) {
        this.showNotification(message, 'danger');
    },
    
    showWarning: function(message) {
        this.showNotification(message, 'warning');
    },
    
    showInfo: function(message) {
        this.showNotification(message, 'info');
    },
    
    showNotification: function(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; max-width: 400px;';
        
        const icon = {
            success: 'check-circle',
            danger: 'exclamation-triangle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        }[type] || 'info-circle';
        
        notification.innerHTML = `
            <i class="fas fa-${icon} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                const bsAlert = new bootstrap.Alert(notification);
                bsAlert.close();
            }
        }, 5000);
    },
    
    // Modal management
    closeAllModals: function() {
        const modals = document.querySelectorAll('.modal.show');
        modals.forEach(modal => {
            const bsModal = bootstrap.Modal.getInstance(modal);
            if (bsModal) {
                bsModal.hide();
            }
        });
    },
    
    // Utility functions
    formatCurrency: function(value, currency = 'USD') {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: currency
        }).format(value);
    },
    
    formatDate: function(date, options = {}) {
        const defaultOptions = {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        };
        return new Intl.DateTimeFormat('en-US', { ...defaultOptions, ...options }).format(new Date(date));
    },
    
    formatNumber: function(value, decimals = 2) {
        return parseFloat(value).toFixed(decimals);
    },
    
    // Chart utilities
    createGradient: function(ctx, color1, color2) {
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, color1);
        gradient.addColorStop(1, color2);
        return gradient;
    },
    
    // Data export utilities
    exportToCSV: function(data, filename = 'data.csv') {
        if (!data || data.length === 0) return;
        
        const headers = Object.keys(data[0]);
        const csvContent = [
            headers.join(','),
            ...data.map(row => 
                headers.map(header => {
                    const value = row[header];
                    return typeof value === 'string' && value.includes(',') ? `"${value}"` : value;
                }).join(',')
            )
        ].join('\n');
        
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    },
    
    exportToJSON: function(data, filename = 'data.json') {
        const jsonContent = JSON.stringify(data, null, 2);
        const blob = new Blob([jsonContent], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        window.URL.revokeObjectURL(url);
    },
    
    // API utilities
    apiCall: function(endpoint, options = {}) {
        const defaultOptions = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            }
        };
        
        return fetch(endpoint, { ...defaultOptions, ...options })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            });
    },
    
    // Cleanup function
    destroy: function() {
        this.pauseAutoRefresh();
        document.removeEventListener('submit', this.handleFormSubmit);
        document.removeEventListener('click', this.handleNavigation);
        document.removeEventListener('keydown', this.handleKeyboardShortcuts);
        window.removeEventListener('resize', this.handleWindowResize);
        document.removeEventListener('visibilitychange', this.handleVisibilityChange);
    }
};

// Page-specific functionality
const PageModules = {
    dashboard: {
        init: function() {
            this.setupQuickActions();
            this.loadRecentData();
        },
        
        setupQuickActions: function() {
            // Quick forecast button
            const quickForecastBtn = document.querySelector('[onclick*="generateQuickForecast"]');
            if (quickForecastBtn) {
                quickForecastBtn.addEventListener('click', this.generateQuickForecast);
            }
        },
        
        loadRecentData: function() {
            OCCApp.apiCall('/api/historical_data')
                .then(data => {
                    this.updateRecentDataTable(data.slice(-12));
                })
                .catch(error => {
                    console.error('Error loading recent data:', error);
                });
        },
        
        updateRecentDataTable: function(data) {
            const tableBody = document.getElementById('recent-data-table');
            if (!tableBody) return;
            
            const html = data.map((row, index) => {
                const prevRow = index > 0 ? data[index - 1] : null;
                const change = prevRow ? row['Price(USD/ton)'] - prevRow['Price(USD/ton)'] : 0;
                const changeClass = change >= 0 ? 'success' : 'danger';
                
                return `
                    <tr>
                        <td>${OCCApp.formatDate(row.Month)}</td>
                        <td>${OCCApp.formatCurrency(row['Price(USD/ton)'])}</td>
                        <td>
                            ${prevRow ? `
                                <span class="badge bg-${changeClass}">
                                    ${change >= 0 ? '+' : ''}${OCCApp.formatNumber(change)}
                                </span>
                            ` : '<span class="text-muted">-</span>'}
                        </td>
                    </tr>
                `;
            }).join('');
            
            tableBody.innerHTML = html;
        },
        
        generateQuickForecast: function() {
            // This function is called from the template
            // Implementation is in the template's script section
        }
    },
    
    forecast: {
        init: function() {
            this.setupForecastForm();
            this.setDefaultValues();
        },
        
        setupForecastForm: function() {
            // Form handling is done in the template's script section
            // This is to avoid conflicts between multiple event listeners
            const form = document.getElementById('forecast-form');
            if (form) {
                console.log('Forecast form found - handling in template script');
            }
        },
        
        setDefaultValues: function() {
            const monthsSelect = document.getElementById('n_months');
            if (monthsSelect && !monthsSelect.value) {
                monthsSelect.value = '6';
            }
        },
        
        handleForecastSubmit: function(event) {
            event.preventDefault();
            const form = event.target;
            const formData = new FormData(form);
            
            OCCApp.showLoading('Generating forecast...');
            
            fetch('/forecast', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                OCCApp.hideLoading();
                if (data.success) {
                    this.displayForecastResults(data);
                } else {
                    OCCApp.showError(data.error || 'Forecast generation failed');
                }
            })
            .catch(error => {
                OCCApp.hideLoading();
                OCCApp.showError('Network error: ' + error.message);
            });
        },
        
        displayForecastResults: function(data) {
            // Implementation is in the template's script section
            // This is a placeholder for the forecast display logic
        }
    },
    
    analytics: {
        init: function() {
            this.setupTabNavigation();
            this.loadChartData();
        },
        
        setupTabNavigation: function() {
            const tabs = document.querySelectorAll('#chartTabs button[data-bs-toggle="tab"]');
            tabs.forEach(tab => {
                tab.addEventListener('shown.bs.tab', this.handleTabChange.bind(this));
            });
        },
        
        handleTabChange: function(event) {
            const target = event.target.getAttribute('data-bs-target');
            console.log('Tab changed to:', target);
            // Refresh charts when tab becomes visible
            setTimeout(() => {
                if (window.Chart) {
                    Chart.helpers.each(Chart.instances, function(instance) {
                        instance.resize();
                    });
                }
            }, 100);
        },
        
        loadChartData: function() {
            // Load data for analytics charts
            OCCApp.apiCall('/api/historical_data')
                .then(data => {
                    this.initializeAnalyticsCharts(data);
                })
                .catch(error => {
                    console.error('Error loading analytics data:', error);
                });
        },
        
        initializeAnalyticsCharts: function(data) {
            // Implementation for analytics-specific charts
            console.log('Initializing analytics charts with data:', data.length, 'records');
        }
    }
};

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    OCCApp.init();
    
    // Initialize page-specific modules
    if (OCCApp.state.currentPage && PageModules[OCCApp.state.currentPage]) {
        PageModules[OCCApp.state.currentPage].init();
    }
});

// Cleanup when page is unloaded
window.addEventListener('beforeunload', function() {
    OCCApp.destroy();
});

// Export for global access
window.OCCApp = OCCApp;
window.PageModules = PageModules;
