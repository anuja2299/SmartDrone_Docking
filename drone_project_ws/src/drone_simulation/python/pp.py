import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import time
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PatternDetector:
    def __init__(self, image_size=(64, 64)):  # Fixed typo: _init_ to __init__
        self.image_size = image_size
        self.model = self.build_cnn()
        self.running = False
        self.history = {'loss': []}  # To store training loss
        self.positions = {'time': [], 'x': [], 'y': []}  # To store real-time positions

    def build_cnn(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_size[0], self.image_size[1], 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(2, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def generate_synthetic_image(self, x, y):
        img = np.zeros(self.image_size, dtype=np.float32)
        x_pixel = int(x * self.image_size[0])
        y_pixel = int(y * self.image_size[1])
        img[max(0, x_pixel-2):x_pixel+3, max(0, y_pixel-2):y_pixel+3] = 1.0
        return img.reshape(self.image_size[0], self.image_size[1], 1)

    def train_cnn(self):
        X_train = []
        y_train = []
        for _ in range(1000):
            x = np.random.rand()
            y = np.random.rand()
            img = self.generate_synthetic_image(x, y)
            X_train.append(img)
            y_train.append([x, y])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        history = self.model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        self.history['loss'].extend(history.history['loss'])  # Store loss for plotting

    def detect_pattern(self):
        self.running = True
        print("Starting pattern detection...")
        start_time = time.time()
        try:
            while self.running:
                x = 2.8  # Adjusted to 0-1 scale later for heatmap
                y = 0.2
                z = 0.0
                position = [x, y, z]
                print(f"Pattern position: x={x:.2f}, y={y:.2f}, z={z:.2f}")
                with open("pattern_position.txt", "w") as f:
                    json.dump({"x": x, "y": y, "z": z}, f)
                # Store positions for real-time plot
                current_time = time.time() - start_time
                self.positions['time'].append(current_time)
                self.positions['x'].append(x / 10)  # Scale to 0-1 for consistency
                self.positions['y'].append(y / 10)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping pattern detection...")
            self.running = False
            self.generate_visualizations()

    def generate_visualizations(self):
        # Accuracy Plot
        y_pred = self.model.predict(np.array([self.generate_synthetic_image(x, y) for x, y in zip(np.random.rand(1000), np.random.rand(1000))]))
        plt.figure(figsize=(8, 6))
        plt.scatter(np.random.rand(1000), y_pred[:, 0], c='blue', label='Predicted X', alpha=0.5)
        plt.scatter(np.random.rand(1000), y_pred[:, 1], c='red', label='Predicted Y', alpha=0.5)
        plt.plot([0, 1], [0, 1], 'k--', label='Ideal')
        plt.title("CNN Accuracy Plot")
        plt.xlabel("Actual Position")
        plt.ylabel("Predicted Position")
        plt.legend()
        plt.savefig('accuracy_plot.png')
        plt.close()

        # Loss Curve
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(self.history['loss']) + 1), self.history['loss'], 'b-')
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.savefig('loss_curve.png')
        plt.close()

        # Prediction Heatmap
        img = self.generate_synthetic_image(self.positions['x'][-1], self.positions['y'][-1])
        plt.figure(figsize=(6, 6))
        plt.imshow(img[:, :, 0], cmap='hot', interpolation='nearest')
        plt.title("Pattern Detection Heatmap")
        plt.colorbar(label='Intensity')
        plt.savefig('prediction_heatmap.png')
        plt.close()

        # Real-Time Position Trend
        plt.figure(figsize=(8, 6))
        plt.plot(self.positions['time'], self.positions['x'], 'b-', label='X Position')
        plt.plot(self.positions['time'], self.positions['y'], 'r-', label='Y Position')
        plt.title("Real-Time Position Tracking")
        plt.xlabel("Time (s)")
        plt.ylabel("Position")
        plt.legend()
        plt.savefig('position_trend.png')
        plt.close()

def main():
    detector = PatternDetector()
    print("Training CNN...")
    detector.train_cnn()
    print("CNN training complete.")
    detector.detect_pattern()

if __name__ == "__main__":  # Fixed typo: _name_ to __name__
    main()
