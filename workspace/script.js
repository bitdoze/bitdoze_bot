/**
 * Simple HTML Website - JavaScript
 * Version: 1.0
 * 
 * Features:
 * - Mobile menu toggle
 * - Smooth scrolling for anchor links
 * - Header shadow on scroll
 * - Keyboard navigation support
 */

(function() {
  'use strict';

  // ========================================
  // Element References
  // ========================================

  const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
  const mainNav = document.querySelector('.main-nav');
  const siteHeader = document.querySelector('.site-header');
  const body = document.body;

  // Check if elements exist
  if (!mobileMenuToggle || !mainNav || !siteHeader) {
    console.warn('Required elements not found');
    return;
  }

  // ========================================
  // Mobile Menu Toggle
  // ========================================

  /**
   * Toggle the mobile menu open/closed state
   */
  function toggleMobileMenu() {
    const isOpen = mainNav.classList.toggle('open');
    
    // Update ARIA attribute for accessibility
    mobileMenuToggle.setAttribute('aria-expanded', isOpen);
    
    // Prevent body scroll when menu is open
    body.style.overflow = isOpen ? 'hidden' : '';
    
    // Return the new state
    return isOpen;
  }

  // Click handler for mobile menu toggle
  mobileMenuToggle.addEventListener('click', function(e) {
    e.preventDefault();
    toggleMobileMenu();
  });

  // ========================================
  // Smooth Scrolling
  // ========================================

  /**
   * Smooth scroll to anchor target
   * @param {Event} e - Click event
   */
  function smoothScroll(e) {
    const href = this.getAttribute('href');
    
    // Only handle anchor links
    if (!href || !href.startsWith('#')) {
      return;
    }
    
    const targetId = href.substring(1);
    const targetElement = document.getElementById(targetId);
    
    if (targetElement) {
      e.preventDefault();
      
      // Close mobile menu if open
      if (mainNav.classList.contains('open')) {
        toggleMobileMenu();
      }
      
      // Smooth scroll to target
      targetElement.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
      
      // Set focus on target for accessibility
      targetElement.setAttribute('tabindex', '-1');
      targetElement.focus();
      
      // Remove tabindex after blur (to avoid keeping focusable)
      targetElement.addEventListener('blur', function() {
        targetElement.removeAttribute('tabindex');
      }, { once: true });
    }
  }

  // Add smooth scrolling to all anchor links
  document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
    anchor.addEventListener('click', smoothScroll);
  });

  // ========================================
  // Header Shadow on Scroll
  // ========================================

  /**
   * Add/remove shadow to header based on scroll position
   */
  function updateHeaderShadow() {
    if (window.pageYOffset > 10) {
      siteHeader.classList.add('scrolled');
    } else {
      siteHeader.classList.remove('scrolled');
    }
  }

  // Throttle scroll events for better performance
  let scrollTimeout;
  window.addEventListener('scroll', function() {
    if (scrollTimeout) {
      window.cancelAnimationFrame(scrollTimeout);
    }
    scrollTimeout = window.requestAnimationFrame(updateHeaderShadow);
  });

  // Initial check
  updateHeaderShadow();

  // ========================================
  // Keyboard Event Handlers
  // ========================================

  /**
   * Handle keyboard events for accessibility
   * @param {KeyboardEvent} e - Keyboard event
   */
  function handleKeyboard(e) {
    // Close menu with Escape key
    if (e.key === 'Escape' || e.key === 'Esc') {
      if (mainNav.classList.contains('open')) {
        toggleMobileMenu();
        // Return focus to toggle button
        mobileMenuToggle.focus();
      }
    }
  }

  // Keyboard event listener
  document.addEventListener('keydown', handleKeyboard);

  // ========================================
  // Close Menu on Outside Click
  // ========================================

  /**
   * Close mobile menu when clicking outside
   * @param {MouseEvent} e - Click event
   */
  function handleOutsideClick(e) {
    if (mainNav.classList.contains('open')) {
      // Check if click is outside the nav
      if (!mainNav.contains(e.target) && !mobileMenuToggle.contains(e.target)) {
        toggleMobileMenu();
      }
    }
  }

  document.addEventListener('click', handleOutsideClick);

  // ========================================
  // Handle Resize - Reset Mobile Menu
  // ========================================

  /**
   * Reset mobile menu state when switching to desktop
   */
  function handleResize() {
    if (window.innerWidth >= 768 && mainNav.classList.contains('open')) {
      mainNav.classList.remove('open');
      mobileMenuToggle.setAttribute('aria-expanded', 'false');
      body.style.overflow = '';
    }
  }

  // Debounce resize events
  let resizeTimeout;
  window.addEventListener('resize', function() {
    if (resizeTimeout) {
      clearTimeout(resizeTimeout);
    }
    resizeTimeout = setTimeout(handleResize, 150);
  });

  // ========================================
  // Active Link Highlighting on Scroll
  // ========================================

  /**
   * Highlight the current section in navigation
   */
  function updateActiveLink() {
    const scrollPosition = window.pageYOffset + 80; // Offset for header
    const sections = document.querySelectorAll('section[id]');
    
    sections.forEach(function(section) {
      const sectionTop = section.offsetTop;
      const sectionHeight = section.offsetHeight;
      const sectionId = section.getAttribute('id');
      
      if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
        // Remove active class from all links
        document.querySelectorAll('.main-nav a').forEach(function(link) {
          link.removeAttribute('aria-current');
        });
        
        // Add active class to current link
        const activeLink = document.querySelector('.main-nav a[href="#' + sectionId + '"]');
        if (activeLink) {
          activeLink.setAttribute('aria-current', 'page');
        }
      }
    });
  }

  // Throttle scroll events for active link updates
  let scrollTimeout2;
  window.addEventListener('scroll', function() {
    if (scrollTimeout2) {
      window.cancelAnimationFrame(scrollTimeout2);
    }
    scrollTimeout2 = window.requestAnimationFrame(updateActiveLink);
  });

  // ========================================
  // Service Cards Keyboard Support
  // ========================================

  const serviceCards = document.querySelectorAll('.service-card[tabindex="0"]');

  serviceCards.forEach(function(card) {
    card.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        // Simulate click behavior
        card.click();
      }
    });
  });

  // ========================================
  // Console Info
  // ========================================

  console.log('Simple HTML Website initialized successfully');
  console.log('Features: Mobile menu, smooth scroll, keyboard navigation');

})();