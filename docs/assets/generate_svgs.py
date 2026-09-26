import os

svg_dir = r"D:\VLM\docs\assets"
os.makedirs(svg_dir, exist_ok=True)

# Generate Architecture SVG
arch_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 600">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#1e293b" />
    </linearGradient>
    <linearGradient id="node" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#3b82f6" />
      <stop offset="100%" stop-color="#2563eb" />
    </linearGradient>
    <linearGradient id="node_ai" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#8b5cf6" />
      <stop offset="100%" stop-color="#6d28d9" />
    </linearGradient>
    <linearGradient id="node_db" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#10b981" />
      <stop offset="100%" stop-color="#059669" />
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  
  <rect width="1000" height="600" fill="url(#bg)" rx="10" />
  
  <g transform="translate(50, 50)" font-family="Arial, sans-serif" font-size="14" fill="#ffffff" filter="url(#glow)">
    <!-- Hardware Layer -->
    <rect x="50" y="50" width="150" height="60" fill="url(#node)" rx="8" />
    <text x="125" y="85" text-anchor="middle">CCTV / Camera</text>
    
    <rect x="50" y="150" width="150" height="60" fill="url(#node)" rx="8" />
    <text x="125" y="185" text-anchor="middle">ESP32 IoT Sensors</text>

    <!-- AI Pipeline Layer -->
    <rect x="300" y="50" width="200" height="60" fill="url(#node_ai)" rx="8" />
    <text x="400" y="85" text-anchor="middle">YOLO Detection + Tracking</text>
    
    <rect x="300" y="150" width="200" height="60" fill="url(#node_ai)" rx="8" />
    <text x="400" y="185" text-anchor="middle">Hierarchical Classification</text>
    
    <rect x="300" y="250" width="200" height="60" fill="url(#node_ai)" rx="8" />
    <text x="400" y="285" text-anchor="middle">Facial Recognition (L5/L6)</text>
    
    <rect x="300" y="350" width="200" height="60" fill="url(#node_ai)" rx="8" />
    <text x="400" y="385" text-anchor="middle">VLM Engine + Behaviour</text>

    <!-- Backend Layer -->
    <rect x="600" y="150" width="150" height="100" fill="url(#node)" rx="8" />
    <text x="675" y="200" text-anchor="middle">FastAPI Backend</text>
    
    <!-- DB / Audit -->
    <rect x="600" y="300" width="150" height="60" fill="url(#node_db)" rx="8" />
    <text x="675" y="335" text-anchor="middle">PostgreSQL DB</text>

    <rect x="600" y="400" width="150" height="60" fill="url(#node_db)" rx="8" />
    <text x="675" y="435" text-anchor="middle">Blockchain Ledger</text>

    <!-- Frontend Layer -->
    <rect x="800" y="150" width="150" height="100" fill="url(#node)" rx="8" />
    <text x="875" y="200" text-anchor="middle">React / Vite</text>
    <text x="875" y="220" text-anchor="middle" font-size="12">Command Center</text>
    
    <!-- Connectors -->
    <path d="M 200 80 C 250 80, 250 80, 300 80" stroke="#94a3b8" stroke-width="2" fill="none" />
    <path d="M 200 180 C 250 180, 250 180, 300 180" stroke="#94a3b8" stroke-width="2" fill="none" />
    <path d="M 500 80 C 550 80, 550 200, 600 200" stroke="#94a3b8" stroke-width="2" fill="none" />
    <path d="M 500 180 C 550 180, 550 200, 600 200" stroke="#94a3b8" stroke-width="2" fill="none" />
    <path d="M 500 280 C 550 280, 550 200, 600 200" stroke="#94a3b8" stroke-width="2" fill="none" />
    <path d="M 500 380 C 550 380, 550 200, 600 200" stroke="#94a3b8" stroke-width="2" fill="none" />
    
    <path d="M 750 200 L 800 200" stroke="#94a3b8" stroke-width="2" fill="none" />
    <path d="M 675 250 L 675 300" stroke="#94a3b8" stroke-width="2" fill="none" />
    <path d="M 675 360 L 675 400" stroke="#94a3b8" stroke-width="2" fill="none" />
  </g>
</svg>"""

with open(os.path.join(svg_dir, "architecture.svg"), "w") as f:
    f.write(arch_svg)

# Generate AI Pipeline SVG
ai_pipeline_svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 400">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0f172a" />
      <stop offset="100%" stop-color="#1e293b" />
    </linearGradient>
    <linearGradient id="node" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#8b5cf6" />
      <stop offset="100%" stop-color="#4c1d95" />
    </linearGradient>
  </defs>
  <rect width="1000" height="400" fill="url(#bg)" rx="10" />
  
  <g transform="translate(50, 150)" font-family="Arial, sans-serif" font-size="14" fill="#ffffff">
    <rect x="0" y="0" width="120" height="50" fill="url(#node)" rx="5" />
    <text x="60" y="30" text-anchor="middle">Input Frame</text>
    
    <rect x="150" y="-50" width="120" height="50" fill="url(#node)" rx="5" />
    <text x="210" y="-20" text-anchor="middle">YOLO Det</text>

    <rect x="150" y="50" width="120" height="50" fill="url(#node)" rx="5" />
    <text x="210" y="80" text-anchor="middle">ByteTrack</text>

    <rect x="300" y="-50" width="120" height="50" fill="url(#node)" rx="5" />
    <text x="360" y="-20" text-anchor="middle">Veh. Classify</text>
    
    <rect x="300" y="50" width="120" height="50" fill="url(#node)" rx="5" />
    <text x="360" y="80" text-anchor="middle">Face Quality</text>

    <rect x="450" y="-50" width="120" height="50" fill="url(#node)" rx="5" />
    <text x="510" y="-20" text-anchor="middle">ANPR / OCR</text>
    
    <rect x="450" y="50" width="120" height="50" fill="url(#node)" rx="5" />
    <text x="510" y="80" text-anchor="middle">Face Recog</text>

    <rect x="600" y="0" width="120" height="50" fill="url(#node)" rx="5" />
    <text x="660" y="30" text-anchor="middle">Behaviour</text>
    
    <rect x="750" y="0" width="120" height="50" fill="url(#node)" rx="5" />
    <text x="810" y="30" text-anchor="middle">VLM Layer</text>

    <!-- Connectors -->
    <path d="M 120 25 C 135 25, 135 -25, 150 -25" stroke="#94a3b8" stroke-width="2" fill="none" />
    <path d="M 120 25 C 135 25, 135 75, 150 75" stroke="#94a3b8" stroke-width="2" fill="none" />
    
    <path d="M 270 -25 L 300 -25" stroke="#94a3b8" stroke-width="2" fill="none" />
    <path d="M 270 75 L 300 75" stroke="#94a3b8" stroke-width="2" fill="none" />
    
    <path d="M 420 -25 L 450 -25" stroke="#94a3b8" stroke-width="2" fill="none" />
    <path d="M 420 75 L 450 75" stroke="#94a3b8" stroke-width="2" fill="none" />
    
    <path d="M 570 -25 C 585 -25, 585 25, 600 25" stroke="#94a3b8" stroke-width="2" fill="none" />
    <path d="M 570 75 C 585 75, 585 25, 600 25" stroke="#94a3b8" stroke-width="2" fill="none" />
    
    <path d="M 720 25 L 750 25" stroke="#94a3b8" stroke-width="2" fill="none" />
  </g>
</svg>"""

with open(os.path.join(svg_dir, "ai_pipeline.svg"), "w") as f:
    f.write(ai_pipeline_svg)
    
print("SVG generation successful.")
