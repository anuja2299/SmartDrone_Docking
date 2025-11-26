import qrcode
import os

# Create a QR code
qr = qrcode.QRCode(version=1, box_size=10, border=1)
qr.add_data("Sample QR Code")
qr.make(fit=True)

# Generate the image
img = qr.make_image(fill='black', back_color='white')

# Expand the ~ to the home directory and save the image
file_path = os.path.expanduser("~/drone_project_ws/src/drone_simulation/worlds/qr_code.png")
img.save(file_path)
