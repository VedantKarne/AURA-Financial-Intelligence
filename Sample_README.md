# 🗺️ RouteRush Island – Gamified Pathfinding Simulator

[![Release](https://img.shields.io/github/v/release/VedantKarne/routerush-island)](https://github.com/VedantKarne/routerush-island/releases)
[![Downloads](https://img.shields.io/github/downloads/VedantKarne/routerush-island/total.svg)](https://github.com/VedantKarne/routerush-island/releases)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-blue)]()

A beautiful, gamified desktop application for visualizing and comparing pathfinding algorithms in real-time on an interactive tropical island.

![RouteRush Island Preview](.github/images/preview.png)

---

## 📥 Quick Start

**System Requirements:** Windows 10+, macOS 10.13+, or Linux (64-bit)

**Download the latest release:**  
👉 **[Windows](https://github.com/VedantKarne/routerush-island/releases)** • **[macOS](https://github.com/VedantKarne/routerush-island/releases)** • **[Linux](https://github.com/VedantKarne/routerush-island/releases)**

🎯 **No installation complexity—just download and run!**

---

## Table of Contents
- [Demo & Tutorial](#-demo--tutorial)
- [Features](#-features)
- [What Makes This Special](#-what-makes-this-special)
- [The Story Behind RouteRush Island](#-the-story-behind-routerush-island)
- [Installation](#-installation)
- [Usage](#-usage)
- [Keyboard Shortcuts](#️-keyboard-shortcuts)
- [Algorithms](#-algorithms)
- [For Developers](#-for-developers)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📺 Demo & Tutorial
<div align="center">
  <a href="https://www.youtube.com/watch?v=kxpqzrJj5qg">
    <img src="https://img.youtube.com/vi/kxpqzrJj5qg/0.jpg" alt="RouteRush Island - Complete Demo and Tutorial" width="100%">
  </a>
</div>

---

## ✨ Features

### 🌍 Interactive Tropical Island
- Pan, zoom, and explore a beautifully rendered map
- Click to set source and destination points
- Place dynamic obstacles (accidents, construction zones)

### 🧠 Compare Algorithms Side-by-Side
Visualize and compare performance of:
- **[Dijkstra's Algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)** - Classic shortest path
- **[Bidirectional Dijkstra](https://en.wikipedia.org/wiki/Bidirectional_search)** - Faster point-to-point search
- **[Bellman-Ford](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm)** - Handles negative weights
- **BMSSP-Inspired** - Educational multi-source visualization

### 🎮 Dynamic Scenarios
- **Normal Traffic**: Standard pathfinding
- **Emergency Routing**: Priority routes
- **Incident Management**: Navigate around obstacles
- **Random Generation**: Unpredictable challenges

### 🎨 Rich Visual Experience
- Real-time algorithm visualization with color-coded paths
- Performance metrics and comparison charts
- Smooth animations and transitions
- Immersive sound effects powered by Tone.js

### 🟡 BMSSP Performance Simulator

> [!WARNING]
> **Educational Disclaimer:** This is a performance simulator based on BMSSP concepts, not a full implementation of the research algorithm. [Learn more →](docs/bmssp_disclaimer.md)

---

## 🎯 What Makes This Special

RouteRush Island isn't just another algorithm visualizer—it's an **immersive learning experience** that makes complex computer science concepts tangible and fun.

✅ **Educational** - Perfect for students learning graph algorithms  
✅ **Visual** - See exactly how each algorithm explores the graph  
✅ **Comparative** - Run multiple algorithms simultaneously  
✅ **Interactive** - Experiment with different scenarios instantly  
✅ **Beautiful** - Modern design with glassmorphism and tropical aesthetics

<div align="center">

### 🚀 Ready to Explore?

[![Download for Windows](https://img.shields.io/badge/Download-Windows-blue?style=for-the-badge&logo=windows)](https://github.com/VedantKarne/routerush-island/releases)
[![Download for macOS](https://img.shields.io/badge/Download-macOS-black?style=for-the-badge&logo=apple)](https://github.com/VedantKarne/routerush-island/releases)
[![Download for Linux](https://img.shields.io/badge/Download-Linux-yellow?style=for-the-badge&logo=linux)](https://github.com/VedantKarne/routerush-island/releases)

</div>

---

## 🌟 The Story Behind RouteRush Island

RouteRush Island started as a DSA mini-project but evolved into something much more immersive. Instead of submitting just another console application, I built an interactive tropical island where algorithms come to life. 

[Read the full journey →](https://medium.com/@vedantkarne15/%EF%B8%8F-building-routerush-island-how-i-turned-textbook-dsa-into-a-gamified-pathfinding-simulator-b3da8f279064)  

---

## 📦 Installation

### For Users (Recommended)

**Download the installer for your platform:**

| Platform | Download | Instructions |
|----------|----------|--------------|
| 🪟 **Windows** | [RouteRush Island Setup.exe](https://github.com/VedantKarne/routerush-island/releases) | Double-click and follow installer |
| 🍎 **macOS (Apple Silicon)** | [RouteRush Island-arm64.dmg](https://github.com/VedantKarne/routerush-island/releases) | Open and drag to Applications |
| 🐧 **Linux** | [RouteRush Island.AppImage](https://github.com/VedantKarne/routerush-island/releases) | `chmod +x` and run |

### For Developers

See [Developer Guide](#️-for-developers) below.

---

## 🎮 Usage

1. **Launch the application** from your applications menu
2. **Click on the map** to set your source (start) point
3. **Click again** to set your destination (end) point
4. **Choose a scenario** (Normal, Emergency, or Incident)
5. **Select algorithms** to compare
6. **Click "Find Best Route"** and watch the magic happen!

**Pro Tips:**
- 🖱️ Click and drag to pan the map
- 🔍 Use mouse wheel or zoom controls to zoom
- 🚧 Switch to Incident mode to place obstacles
- 📊 Compare performance metrics in real-time

---

## ⌨️ Keyboard Shortcuts

RouteRush Island includes comprehensive keyboard shortcuts for navigation, camera control, simulation management, route selection, and more.

📖 **[View all shortcuts →](docs/shortcuts.md)**

---

## 🎨 Algorithms

RouteRush Island implements four pathfinding algorithms, each with unique strengths:

| Algorithm | Speed | Handles Negatives | Best For |
|-----------|-------|-------------------|----------|
| **Dijkstra** | Fast | ❌ No | General shortest path |
| **Bidirectional** | Faster | ❌ No | Point-to-point queries |
| **Bellman-Ford** | Slower | ✅ Yes | Negative weights |
| **BMSSP-Inspired** | Fastest | Educational | Learning concepts |

📖 **[Read detailed algorithm explanations →](docs/algorithms.md)**

---

## 🛠️ For Developers

### Build from Source

**Prerequisites:**
- Node.js (v14 or higher)
- npm or yarn

**Quick Start:**
```bash
# Clone the repository
git clone https://github.com/VedantKarne/routerush-island.git
cd routerush-island

# Install dependencies
npm install

# Run in development mode
npm start
```

### Build for Production

```bash
# Build for all platforms
npm run build

# Or build for specific platform
npm run build:win    # Windows
npm run build:mac    # macOS
npm run build:linux  # Linux
```

Built installers will be in the `release/` directory.

### Project Structure

```text
traffic-simulator-desktop/
├── src/           # Source code
├── docs/          # Documentation
├── dist/          # Production build
├── build/         # Build assets
└── main.js        # Electron entry point
```

📖 **[View detailed project structure →](docs/project_structure.md)**

### Tech Stack

- **Desktop Framework:** [Electron](https://electronjs.org) v28.0
- **Rendering:** HTML5 Canvas API
- **Audio:** [Tone.js](https://tonejs.github.io) v15.0
- **Build Tool:** [Electron Builder](https://electron.build)
- **Language:** JavaScript (ES6+)

---

## 🤝 Contributing

We welcome contributions! Whether it's:

- 🐛 Bug fixes
- ✨ New features
- 📝 Documentation improvements
- 🎨 UI/UX enhancements

**Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a PR.**

### Quick Contribution Steps

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Algorithm implementations inspired by classic CS textbooks
- Visual design inspired by modern web aesthetics
- Tropical theme for an engaging learning experience

---

## 📧 Support

- 🐛 **Found a bug?** [Open an issue](https://github.com/VedantKarne/routerush-island/issues/new?template=bug_report.md)
- 💡 **Have an idea?** [Request a feature](https://github.com/VedantKarne/routerush-island/issues/new?template=feature_request.md)
- 💬 **Questions?** [Start a discussion](https://github.com/VedantKarne/routerush-island/discussions)

---

## 🔍 Keywords

Pathfinding • Dijkstra • Bellman-Ford • Visualization • Electron • Desktop App • BMSSP • Algorithm Simulator • Traffic Simulation • Shortest Path • Graph Algorithms • Educational Tool

---

<div align="center">

**Enjoy exploring pathfinding algorithms with RouteRush Island!** 🏝️🚗

[Download Now](https://github.com/VedantKarne/routerush-island/releases) • [View Documentation](docs/) • [Report Issue](https://github.com/VedantKarne/routerush-island/issues)

</div>
