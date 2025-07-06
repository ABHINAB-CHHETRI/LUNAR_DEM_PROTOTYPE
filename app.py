import os
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import numpy.fft as fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

def frankot_chellappa(p, q):
    H, W = p.shape
    fx = fft.fftfreq(W).reshape(1, -1)
    fy = fft.fftfreq(H).reshape(-1, 1)
    FX, FY = np.tile(fx, (H, 1)), np.tile(fy, (1, W))
    FX[0, 0] = 1e-6; FY[0, 0] = 1e-6
    denom = (2j * np.pi * FX)**2 + (2j * np.pi * FY)**2
    Z = (fft.fft2(p) * 2j * np.pi * FX + fft.fft2(q) * 2j * np.pi * FY) / denom
    return fft.ifft2(Z).real

def process_and_plot(img_path, out_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0

    gy, gx = np.gradient(img)
    gx_s, gy_s = gaussian_filter(gx, 2), gaussian_filter(gy, 2)

    sun_azimuth = np.pi / 4
    sun_elevation = np.pi / 6
    L = np.array([
        np.cos(sun_elevation) * np.cos(sun_azimuth),
        np.cos(sun_elevation) * np.sin(sun_azimuth),
        np.sin(sun_elevation)
    ])

    N = np.dstack([-gx, -gy, np.ones_like(img)])
    N_unit = N / (np.linalg.norm(N, axis=2, keepdims=True) + 1e-8)
    I_est = np.clip(np.sum(N_unit * L, axis=2), 0, 1)

    disparity = np.sqrt(gx_s**2 + gy_s**2)
    elevation = frankot_chellappa(-gx_s, -gy_s)

    plt.figure(figsize=(18, 8))
    plt.subplot(2, 3, 1)
    plt.title("Original Lunar Image")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Shading Estimate")
    plt.imshow(I_est, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Disparity Map (|∇z|)")
    plt.imshow(disparity, cmap="plasma")
    plt.colorbar(label="Disparity", shrink=0.7)
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("Elevation Map")
    plt.imshow(elevation, cmap="terrain")
    plt.colorbar(label="Elevation", shrink=0.7)
    plt.axis("off")

    ax = plt.subplot(2, 3, (5,6), projection='3d')
    X, Y = np.meshgrid(np.arange(elevation.shape[1]), np.arange(elevation.shape[0]))
    ax.plot_surface(X, Y, elevation, cmap='terrain', linewidth=0, antialiased=False)
    ax.set_title("3D Surface")
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            out_path = os.path.join('static', 'plot.png')
            process_and_plot(filepath, out_path)
            plot_url = url_for('static', filename='plot.png')
    return render_template('index.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
