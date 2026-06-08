/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono:    ['"JetBrains Mono"', 'monospace'],
        display: ['"Space Grotesk"', 'sans-serif'],
        body:    ['"DM Sans"', 'sans-serif'],
      },
      colors: {
        hex: {
          bg:      '#060b14',
          surface: '#0c1220',
          border:  '#1a2540',
          muted:   '#1e2d4a',
          text:    '#e2eaf6',
          sub:     '#6b7fa3',
          dim:     '#3a4a6b',
          accent:  '#00d4ff',
          green:   '#00ff9d',
          red:     '#ff3366',
          yellow:  '#ffd166',
          purple:  '#b57bee',
          orange:  '#ff8c42',
        }
      },
      animation: {
        'fade-in':  'fadeIn 0.4s ease forwards',
        'slide-up': 'slideUp 0.4s ease forwards',
        'glow':     'glow 2s ease-in-out infinite alternate',
        'scan':     'scan 4s linear infinite',
      },
      keyframes: {
        fadeIn:  { from: { opacity: 0 }, to: { opacity: 1 } },
        slideUp: { from: { opacity: 0, transform: 'translateY(12px)' },
                   to:   { opacity: 1, transform: 'translateY(0)' } },
        glow:    { from: { boxShadow: '0 0 5px #00d4ff33' },
                   to:   { boxShadow: '0 0 20px #00d4ff66, 0 0 40px #00d4ff22' } },
        scan:    { from: { transform: 'translateY(-100%)' },
                   to:   { transform: 'translateY(100vh)' } },
      }
    }
  },
  plugins: []
}