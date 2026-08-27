# 📦 Installation Guide

<div class="feature-header">
  <div class="feature-title">
    <h2>Installation Guide</h2>
    <p>Get KDP up and running in your environment quickly and easily</p>
  </div>
</div>

## 📋 Overview

<div class="overview-card">
  <p>KDP can be installed through various methods, from simple pip installation to building from source. Choose the method that best fits your needs and environment.</p>
</div>

<div class="key-benefits">
  <div class="benefit-card">
    <span class="benefit-icon">🚀</span>
    <h3>Quick Installation</h3>
    <p>Simple pip install for most users</p>
  </div>
  <div class="benefit-card">
    <span class="benefit-icon">🛠️</span>
    <h3>Multiple Methods</h3>
    <p>pip, Poetry, or source installation</p>
  </div>
  <div class="benefit-card">
    <span class="benefit-icon">🧩</span>
    <h3>Optional Dependencies</h3>
    <p>Install only what you need</p>
  </div>
  <div class="benefit-card">
    <span class="benefit-icon">⚡</span>
    <h3>GPU Support</h3>
    <p>Leverage GPU acceleration</p>
  </div>
</div>

## 🚀 Quick Installation

<div class="code-container">

```bash
pip install kdp
```

</div>

## 🛠️ Installation Methods

<div class="installation-methods">
  <div class="method-card">
    <h3>Using pip (Recommended)</h3>
    <div class="code-container">

```bash
# Basic installation
pip install kdp
```

    </div>
  </div>

  <div class="method-card">
    <h3>Using Poetry</h3>
    <div class="code-container">

```bash
# Add to your project
poetry add kdp
```

    </div>
  </div>

  <div class="method-card">
    <h3>From Source</h3>
    <div class="code-container">

```bash
# Clone the repository
git clone https://github.com/UnicoLab/keras-data-processor.git
cd keras-data-processor

# Install using pip
pip install -e .

# Or using poetry
poetry install
```

    </div>
  </div>
</div>

## 🧩 Dependencies

<div class="dependencies-container">
  <div class="dependencies-card">
    <h3>Core Dependencies</h3>
    <ul class="dependency-list">
      <li>🐍 Python 3.10+</li>
      <li>🔄 TensorFlow 2.18+ (Keras 3)</li>
      <li>🔢 NumPy 1.19+</li>
      <li>📊 Pandas 1.2+</li>
    </ul>
  </div>

  <div class="dependencies-card">
    <h3>Optional Dependencies</h3>
    <div class="table-container">
      <table class="config-table">
        <thead>
          <tr>
            <th>Package</th>
            <th>Purpose</th>
            <th>Install Command</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>scipy</td>
            <td>🧪 Scientific computing and statistical functions</td>
            <td><code>pip install "kdp[dev]"</code></td>
          </tr>
          <tr>
            <td>ipython</td>
            <td>🔍 Interactive Python shell and notebook support</td>
            <td><code>pip install "kdp[dev]"</code></td>
          </tr>
          <tr>
            <td>pytest</td>
            <td>✅ Testing framework and utilities</td>
            <td><code>pip install "kdp[dev]"</code></td>
          </tr>
          <tr>
            <td>pydot</td>
            <td>📊 Graph visualization for model architecture</td>
            <td><code>pip install "kdp[dev]"</code></td>
          </tr>
          <tr>
            <td>Development Tools</td>
            <td>🛠️ All development dependencies</td>
            <td><code>pip install "kdp[dev]"</code></td>
          </tr>
          <tr>
            <td>Documentation Tools</td>
            <td>📚 Documentation generation tools</td>
            <td><code>pip install "kdp[doc]"</code></td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

## 🖥️ GPU Support

<div class="gpu-support-card">
  <h3>Enable GPU Acceleration</h3>
  <div class="code-container">

```bash
# Install TensorFlow with GPU support
pip install tensorflow-gpu
```

  </div>
  <p>Ensure you have the appropriate CUDA and cuDNN versions installed.</p>
</div>

## ✅ Verifying Your Installation

<div class="verification-card">
  <div class="code-container">

```python
import kdp

# Check version
print(f"KDP version: {kdp.__version__}")

# Basic functionality test
from kdp import PreprocessingModel, FeatureType
features = {"test": FeatureType.FLOAT}
model = PreprocessingModel(features_specs=features)
print("Installation successful!")
```

  </div>
</div>

## 👣 Next Steps

<div class="next-steps-container">
  <div class="next-step-card">
    <span class="next-step-icon">🏁</span>
    <h3>Quick Start Guide</h3>
    <p>Learn the basics of KDP</p>
    <a href="quick-start.md" class="next-step-link">Get Started →</a>
  </div>

  <div class="next-step-card">
    <span class="next-step-icon">🏗️</span>
    <h3>Architecture Overview</h3>
    <p>Understand KDP's components</p>
    <a href="architecture.md" class="next-step-link">Learn More →</a>
  </div>

  <div class="next-step-card">
    <span class="next-step-icon">🔍</span>
    <h3>Feature Processing</h3>
    <p>Explore KDP's capabilities</p>
    <a href="../features/overview.md" class="next-step-link">Explore →</a>
  </div>
</div>

<style>
/* Base styling */
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.6;
  color: #333;
  margin: 0;
  padding: 0;
}

/* Feature header */
.feature-header {
  background: linear-gradient(135deg, #FF5722 0%, #FF7043 100%);
  border-radius: 10px;
  padding: 30px;
  margin: 30px 0;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  color: white;
}

.feature-title h2 {
  margin-top: 0;
  font-size: 28px;
}

.feature-title p {
  font-size: 18px;
  margin-bottom: 0;
  opacity: 0.9;
}

/* Overview card */
.overview-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px 25px;
  margin: 20px 0;
  box-shadow: 0 2px 5px rgba(0,0,0,0.05);
  border-left: 4px solid #FF5722;
}

.overview-card p {
  margin: 0;
  font-size: 16px;
}

/* Key benefits */
.key-benefits {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.benefit-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.benefit-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.benefit-icon {
  font-size: 2.5em;
  margin-bottom: 15px;
}

.benefit-card h3 {
  margin: 0 0 10px 0;
  color: #FF5722;
}

.benefit-card p {
  margin: 0;
}

/* Installation methods */
.installation-methods {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.method-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.method-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.method-card h3 {
  margin-top: 0;
  color: #FF5722;
}

/* Dependencies */
.dependencies-container {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.dependencies-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}

.dependencies-card h3 {
  margin-top: 0;
  color: #FF5722;
}

.dependency-list {
  list-style: none;
  padding: 0;
}

.dependency-list li {
  margin: 10px 0;
  display: flex;
  align-items: center;
}

/* GPU support */
.gpu-support-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  margin: 30px 0;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}

.gpu-support-card h3 {
  margin-top: 0;
  color: #FF5722;
}

/* Verification */
.verification-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  margin: 30px 0;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}

/* Next steps */
.next-steps-container {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: 20px;
  margin: 30px 0;
}

.next-step-card {
  background-color: #fff;
  border-radius: 10px;
  padding: 20px;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  text-align: center;
}

.next-step-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0,0,0,0.1);
}

.next-step-icon {
  font-size: 2.5em;
  margin-bottom: 15px;
}

.next-step-card h3 {
  margin: 0 0 10px 0;
  color: #FF5722;
}

.next-step-card p {
  margin: 0 0 15px 0;
}

.next-step-link {
  display: inline-block;
  padding: 8px 16px;
  background-color: #FF5722;
  color: white;
  text-decoration: none;
  border-radius: 5px;
  transition: background-color 0.3s ease;
}

.next-step-link:hover {
  background-color: #F4511E;
}

/* Code containers */
.code-container {
  background-color: #f8f9fa;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 5px rgba(0,0,0,0.1);
  margin: 20px 0;
}

.code-container pre {
  margin: 0;
  padding: 20px;
}

/* Tables */
.table-container {
  margin: 30px 0;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}

.config-table {
  width: 100%;
  border-collapse: collapse;
}

.config-table th {
  background-color: #FBE9E7;
  padding: 15px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid #FF5722;
}

.config-table td {
  padding: 12px 15px;
  border-bottom: 1px solid #eaecef;
}

.config-table tr:nth-child(even) {
  background-color: #f8f9fa;
}

.config-table tr:hover {
  background-color: #FBE9E7;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .key-benefits,
  .installation-methods,
  .dependencies-container,
  .next-steps-container {
    grid-template-columns: 1fr;
  }
}
</style>
