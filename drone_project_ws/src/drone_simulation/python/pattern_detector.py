import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import time
import json

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PatternDetector:
    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size
        self.model = self.build_cnn()
        self.running = False

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
        self.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    def detect_pattern(self):
        self.running = True
        print("Starting pattern detection...")
        try:
            while self.running:
                x = 2.8
                y = 0.2
                z = 0.0
                position = [x, y, z]
                print(f"Pattern position: x={x:.2f}, y={y:.2f}, z={z:.2f}")
                with open("pattern_position.txt", "w") as f:
                    json.dump({"x": x, "y": y, "z": z}, f)
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping pattern detection...")
            self.running = False

def main():
    detector = PatternDetector()
    print("Training CNN...")
    detector.train_cnn()
    print("CNN training complete.")
    detector.detect_pattern()

if __name__ == "__main__":
    main()
