from PIL import Image
import numpy as np
import imageio

# === Step 1: Load all three images ===
paths = ["Picture 1.png", "Picture 2.png", "Picture 3.png"]
images = [Image.open(p).convert("RGB") for p in paths]

# === Step 2: Resize to same size ===
base_size = images[0].size
images = [img.resize(base_size) for img in images]

# === Step 3: Define dispersion patch area ===
# (This must be adjusted to fit your specific layout if needed)
# Format: (left, top, right, bottom)
dispersion_box = (120, 100, 630, 470)  # You can fine-tune this if needed

# === Step 4: Use first image as static layout ===
static_frame = images[0].copy()

# === Step 5: Extract just the dynamic dispersion parts ===
patches = [img.crop(dispersion_box) for img in images]

# === Step 6: Interpolate between patches and paste into static layout ===
def blend_dispersion_patch(p1, p2, steps=60):
    a1, a2 = np.array(p1).astype(float), np.array(p2).astype(float)
    for alpha in np.linspace(0, 1, steps):
        blended = Image.fromarray(np.uint8((1 - alpha) * a1 + alpha * a2))
        new_frame = static_frame.copy()
        new_frame.paste(blended, dispersion_box[:2])
        yield new_frame

# === Step 7: Build the full animation sequence ===
frames = []
for i in range(len(patches) - 1):
    frames.extend(blend_dispersion_patch(patches[i], patches[i+1], steps=60))
for i in reversed(range(len(patches) - 1)):
    frames.extend(blend_dispersion_patch(patches[i+1], patches[i], steps=60))

# === Step 8: Save as animated GIF ===
output_path = "dispersion_simulation.gif"
imageio.mimsave(output_path, [np.array(f) for f in frames], duration=0.03)

print(f"✅ Done! Saved animation as: {output_path}")
