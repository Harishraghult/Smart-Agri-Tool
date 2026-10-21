import React from 'react';

// Decorative leafy vines drawn as SVG (no external images, no licensing concerns).
// Fixed behind all content; purely visual, so hidden from screen readers and clicks pass through.

function Leaf({ x, y, angle, size = 1, tone }) {
  return (
    <g transform={`translate(${x} ${y}) rotate(${angle}) scale(${size})`}>
      <path d="M0 0 C 18 -26, 58 -28, 84 0 C 58 28, 18 26, 0 0 Z" fill={tone} />
      <path d="M2 0 C 30 -2, 56 -1, 80 0" stroke="#ffffff" strokeOpacity="0.55" strokeWidth="1.6" fill="none" />
      <path d="M26 0 L 38 -10 M44 0 L 56 -9 M26 0 L 38 10 M44 0 L 56 9" stroke="#ffffff" strokeOpacity="0.4" strokeWidth="1.1" fill="none" />
    </g>
  );
}

function Vine({ leaves, stem }) {
  return (
    <svg viewBox="0 0 420 420" width="100%" height="100%" aria-hidden="true">
      <path d={stem} stroke="#5b8c3a" strokeWidth="4" fill="none" strokeLinecap="round" />
      {leaves.map((l, i) => <Leaf key={i} {...l} />)}
    </svg>
  );
}

const TOP_LEFT = {
  stem: 'M-10 40 C 90 60, 150 120, 190 210 S 250 360, 330 430',
  leaves: [
    { x: 40, y: 48, angle: -20, size: 1.1, tone: '#7cb342' },
    { x: 95, y: 70, angle: 35, size: 0.95, tone: '#558b2f' },
    { x: 140, y: 120, angle: -35, size: 1.05, tone: '#8bc34a' },
    { x: 172, y: 175, angle: 30, size: 0.9, tone: '#689f38' },
    { x: 195, y: 235, angle: -50, size: 1.0, tone: '#7cb342' },
    { x: 230, y: 300, angle: 25, size: 0.85, tone: '#558b2f' },
    { x: 270, y: 360, angle: -40, size: 0.9, tone: '#8bc34a' },
  ],
};

const BOTTOM_RIGHT = {
  stem: 'M430 380 C 330 360, 270 300, 230 220 S 170 70, 90 -10',
  leaves: [
    { x: 380, y: 372, angle: 200, size: 1.1, tone: '#689f38' },
    { x: 320, y: 345, angle: 150, size: 0.95, tone: '#8bc34a' },
    { x: 275, y: 295, angle: 215, size: 1.0, tone: '#558b2f' },
    { x: 245, y: 235, angle: 160, size: 0.9, tone: '#7cb342' },
    { x: 220, y: 170, angle: 230, size: 1.0, tone: '#689f38' },
    { x: 185, y: 105, angle: 150, size: 0.85, tone: '#8bc34a' },
    { x: 145, y: 50, angle: 220, size: 0.9, tone: '#558b2f' },
  ],
};

export default function PlantBackdrop() {
  return (
    <div className="plant-backdrop" aria-hidden="true">
      <div className="plant-vine plant-vine-top-left"><Vine {...TOP_LEFT} /></div>
      <div className="plant-vine plant-vine-bottom-right"><Vine {...BOTTOM_RIGHT} /></div>
    </div>
  );
}
