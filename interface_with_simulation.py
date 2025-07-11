import gradio as gr
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

# Load trained model and encoder
model = joblib.load("outputs/model.pkl")
encoder = joblib.load("outputs/label_encoder.pkl")

# Constants for simulation
G = 6.67430e-11
M_earth = 5.972e24
R_earth = 6.371e6

def classify_and_simulate(mass, year, lat, long, simulate):
    # Input validation
    if not (0 < mass < 1000):
        return "❌ Invalid mass", None
    if mass >= 10:
        return "❌ Not a micrometeoroid (mass ≥ 10g)", None
    if not (1000 <= year <= 2030):
        return "❌ Invalid year", None
    if not (-90 <= lat <= 90):
        return "❌ Invalid latitude", None
    if not (-180 <= long <= 180):
        return "❌ Invalid longitude", None

    # Classification
    features = np.array([[mass, year, lat, long]])
    label = model.predict(features)[0]
    class_name = encoder.inverse_transform([label])[0]

    message = f"✅ Micrometeoroid Detected\nClass: **{class_name}**"

    if simulate:
        # Begin orbital simulation
        angle_deg = 45
        speed = 12000
        angle_rad = np.radians(angle_deg)

        x, y = 0, R_earth + 100000
        vx = speed * np.cos(angle_rad)
        vy = -speed * np.sin(angle_rad)

        dt = 0.1
        t_max = 1000
        pos_x, pos_y = [], []

        for _ in range(int(t_max / dt)):
            r = np.sqrt(x**2 + y**2)
            if r <= R_earth:
                break
            a = -G * M_earth / r**2
            ax = a * x / r
            ay = a * y / r
            vx += ax * dt
            vy += ay * dt
            x += vx * dt
            y += vy * dt
            pos_x.append(x / 1000)
            pos_y.append(y / 1000)

        # Plot
        plt.figure(figsize=(6, 5))
        plt.plot(pos_x, pos_y, label=class_name, color='orange' if class_name == "Chondrite" else ('gray' if class_name == "Iron" else 'green'))
        plt.xlabel("x (km)")
        plt.ylabel("y (km)")
        plt.title("Micrometeoroid Orbit Simulation")
        plt.grid(True)
        plt.axis('equal')
        earth = plt.Circle((0, 0), R_earth / 1000, color='blue', alpha=0.5)
        plt.gca().add_patch(earth)
        plt.legend()
        plot_path = "outputs/orbit_plot.png"
        plt.savefig(plot_path)
        plt.close()
        return message, plot_path

    return message, None

# Create UI
interface = gr.Interface(
    fn=classify_and_simulate,
    inputs=[
        gr.Number(label="Mass (g)", value=5),
        gr.Number(label="Year", value=2020),
        gr.Number(label="Latitude", value=0.0),
        gr.Number(label="Longitude", value=0.0),
        gr.Checkbox(label="Simulate Orbit")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Image(label="Orbit Simulation (if selected)")
    ],
    title="☄️ Micrometeoroid Classifier & Simulator",
    description="Enter meteorite info to classify it and simulate its orbital path. Built with ML + Physics!"
)

interface.launch()
