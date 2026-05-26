/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        background: 'rgb(var(--background) / <alpha-value>)',
        foreground: 'rgb(var(--foreground) / <alpha-value>)',
        panel: 'rgb(var(--panel) / <alpha-value>)',
        panelStrong: 'rgb(var(--panel-strong) / <alpha-value>)',
        line: 'rgb(var(--line) / <alpha-value>)',
        muted: 'rgb(var(--muted) / <alpha-value>)',
        cyan: 'rgb(var(--cyan) / <alpha-value>)',
        green: 'rgb(var(--green) / <alpha-value>)',
        amber: 'rgb(var(--amber) / <alpha-value>)',
        red: 'rgb(var(--red) / <alpha-value>)',
      },
      fontFamily: {
        display: ['IBM Plex Sans', 'Aptos Display', 'Segoe UI', 'sans-serif'],
        body: ['IBM Plex Sans', 'Aptos', 'Segoe UI', 'sans-serif'],
        mono: ['IBM Plex Mono', 'Cascadia Mono', 'Consolas', 'monospace'],
      },
      boxShadow: {
        terminal: 'inset 0 1px 0 rgb(255 255 255 / 0.035)',
        bloom: 'none',
      },
    },
  },
  plugins: [],
}
