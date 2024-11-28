import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt

class ImageEnhancementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Enhancement App")

        self.image_path = tk.StringVar()
        self.operation = tk.StringVar(value="Grayscale")
        self.min_val = tk.IntVar(value=0)
        self.max_val = tk.IntVar(value=255)
        self.threshold = tk.IntVar(value=128)

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="Select Image:").grid(row=0, column=0, padx=10, pady=10)
        tk.Entry(self.root, textvariable=self.image_path, width=50).grid(row=0, column=1, padx=10, pady=10)
        tk.Button(self.root, text="Browse", command=self.browse_image).grid(row=0, column=2, padx=10, pady=10)

        tk.Label(self.root, text="Select Operation:").grid(row=1, column=0, padx=10, pady=10)
        operations = ["Grayscale", "Pseudo-color", "Negative", "Threshold", "Histogram Equalization"]
        self.operation_menu = ttk.Combobox(self.root, textvariable=self.operation, values=operations)
        self.operation_menu.grid(row=1, column=1, padx=10, pady=10)
        self.operation_menu.bind("<<ComboboxSelected>>", self.update_options)

        self.min_label = tk.Label(self.root, text="Min Value:")
        self.min_label.grid(row=2, column=0, padx=10, pady=10)
        self.min_entry = tk.Entry(self.root, textvariable=self.min_val)
        self.min_entry.grid(row=2, column=1, padx=10, pady=10)

        self.max_label = tk.Label(self.root, text="Max Value:")
        self.max_label.grid(row=3, column=0, padx=10, pady=10)
        self.max_entry = tk.Entry(self.root, textvariable=self.max_val)
        self.max_entry.grid(row=3, column=1, padx=10, pady=10)

        self.threshold_label = tk.Label(self.root, text="Threshold:")
        self.threshold_label.grid(row=4, column=0, padx=10, pady=10)
        self.threshold_entry = tk.Entry(self.root, textvariable=self.threshold)
        self.threshold_entry.grid(row=4, column=1, padx=10, pady=10)

        tk.Button(self.root, text="Apply", command=self.apply_operation).grid(row=5, column=0, columnspan=3, pady=20)

        self.update_options()

    def update_options(self, event=None):
        operation = self.operation.get()
        if operation in ["Grayscale", "Negative"]:
            self.min_label.grid_remove()
            self.min_entry.grid_remove()
            self.max_label.grid_remove()
            self.max_entry.grid_remove()
            self.threshold_label.grid_remove()
            self.threshold_entry.grid_remove()
        elif operation == "Pseudo-color":
            self.min_label.grid()
            self.min_entry.grid()
            self.max_label.grid()
            self.max_entry.grid()
            self.threshold_label.grid_remove()
            self.threshold_entry.grid_remove()
        elif operation in ["Threshold", "Histogram Equalization"]:
            self.threshold_label.grid()
            self.threshold_entry.grid()
            self.min_label.grid_remove()
            self.min_entry.grid_remove()
            self.max_label.grid_remove()
            self.max_entry.grid_remove()

    def browse_image(self):
        self.image_path.set(filedialog.askopenfilename())

    def apply_operation(self):
        image_path = self.image_path.get()
        operation = self.operation.get()
        min_val = self.min_val.get()
        max_val = self.max_val.get()
        threshold = self.threshold.get()

        if not image_path:
            messagebox.showerror("Error", "Please select an image.")
            return

        img = Image.open(image_path).convert('L')
        img_array = np.array(img)

        if operation == "Grayscale":
            result_img = img_array
            hist_img = None
            output_hist_img = None
        elif operation == "Pseudo-color":
            result_img = self.pseudo_color(img_array, min_val, max_val)
            hist_img = None
            output_hist_img = None
        elif operation == "Negative":
            result_img = 255 - img_array
            hist_img = None
            output_hist_img = None
        elif operation == "Threshold":
            result_img = self.apply_threshold(img_array, threshold)
            hist_img = None
            output_hist_img = None
        elif operation == "Histogram Equalization":
            result_img, hist_img, output_hist_img = self.histogram_equalization(img_array)

        self.display_results(result_img, hist_img, output_hist_img)

    def pseudo_color(self, image, min_val, max_val):
        # Create a mask for the range of interest
        mask = (image >= min_val) & (image <= max_val)

        # Create a 3D array for the colored image
        colored_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        norm = plt.Normalize(min_val, max_val)
        colored_data = plt.cm.viridis(norm(image[mask]))[:, :3] * 255

        # Assign the RGB values to the masked region
        colored_image[mask] = colored_data.astype(np.uint8)

        # Combine the colored region with the original grayscale image
        result_image = np.zeros_like(colored_image, dtype=np.uint8)
        result_image[mask] = colored_image[mask]
        result_image[~mask] = np.stack((image[~mask],)*3, axis=-1)

        return result_image

    def apply_threshold(self, image, threshold):
        return np.where(image > threshold, 255, 0)

    def histogram_equalization(self, image):
        # Calculate the histogram
        hist, bins = np.histogram(image.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()

        # Calculate the equalized image
        equalized_img = np.interp(image.flatten(), bins[:-1], cdf_normalized)
        equalized_img = equalized_img.reshape(image.shape)
        equalized_img = (equalized_img * 255).astype(np.uint8)

        # Calculate the output histogram
        output_hist, _ = np.histogram(equalized_img.flatten(), 256, [0, 256])

        # Plot the histograms
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(cdf_normalized, color='b')
        ax[0].set_title('Input Histogram')
        ax[1].plot(output_hist, color='r')
        ax[1].set_title('Output Histogram')
        plt.tight_layout()

        # Convert the plot to an image
        fig.canvas.draw()
        hist_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        hist_img = hist_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)  # Close the figure to avoid displaying it

        return equalized_img, hist_img, output_hist

    def display_results(self, result_img, hist_img=None, output_hist_img=None):
        result_window = tk.Toplevel(self.root)
        result_window.title("Results")

        if hist_img is not None:
            hist_canvas = tk.Canvas(result_window, width=600, height=400)
            hist_canvas.pack(side=tk.LEFT, padx=10, pady=10)
            hist_img = Image.fromarray(hist_img)
            hist_img = hist_img.resize((600, 400), Image.LANCZOS)
            hist_img = ImageTk.PhotoImage(hist_img)
            hist_canvas.create_image(0, 0, anchor=tk.NW, image=hist_img)
            hist_canvas.image = hist_img

        result_canvas = tk.Canvas(result_window, width=600, height=400)
        result_canvas.pack(side=tk.RIGHT, padx=10, pady=10)
        result_img = Image.fromarray(result_img.astype('uint8'))
        result_img = result_img.resize((600, 400), Image.LANCZOS)
        result_img = ImageTk.PhotoImage(result_img)
        result_canvas.create_image(0, 0, anchor=tk.NW, image=result_img)
        result_canvas.image = result_img

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageEnhancementApp(root)
    root.mainloop()
