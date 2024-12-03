import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import cv2
from sklearn.cluster import KMeans
import shutil
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from datetime import datetime


class WeddingAlbumMaker:
    def __init__(self, input_folder, output_folder, target_photos=125):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.target_photos = target_photos

        # Initialize ML models
        self.quality_model = ResNet50(weights='imagenet', include_top=False)

        # Load OpenCV's face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

    def analyze_image_quality(self, img_path):
        """Score image quality based on various metrics"""
        try:
            # Load and preprocess image for ResNet
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            x = np.expand_dims(x, axis=0)

            # Get features from ResNet
            features = self.quality_model.predict(x)

            # Calculate basic quality metrics
            img = Image.open(img_path)
            width, height = img.size
            aspect_ratio = width / height
            resolution_score = min(1.0, (width * height) / (4000 * 3000))

            # Check for blur
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Calculate exposure score
            exposure_score = self.calculate_exposure_score(gray)

            # Calculate composition score based on rule of thirds
            composition_score = self.analyze_composition(gray)

            # Combine scores
            quality_score = (
                    0.3 * np.mean(features) +  # Visual features
                    0.2 * resolution_score +  # Resolution
                    0.2 * min(1.0, blur_score / 1000) +  # Sharpness
                    0.15 * exposure_score +  # Exposure
                    0.15 * composition_score  # Composition
            )

            return quality_score

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return 0

    def calculate_exposure_score(self, gray_img):
        """Calculate image exposure score using histogram analysis"""
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize

        # Calculate mean brightness
        mean_brightness = np.average(np.arange(256), weights=hist)

        # Calculate standard deviation for contrast
        std_brightness = np.sqrt(np.average((np.arange(256) - mean_brightness) ** 2, weights=hist))

        # Score based on deviation from ideal brightness (128) and contrast
        exposure_score = 1.0 - abs(mean_brightness - 128) / 128
        contrast_score = min(1.0, std_brightness / 64)

        return 0.6 * exposure_score + 0.4 * contrast_score

    def analyze_composition(self, img):
        """Analyze image composition using rule of thirds"""
        height, width = img.shape

        # Define regions of interest (rule of thirds intersections)
        roi_points = [
            (width // 3, height // 3),
            (2 * width // 3, height // 3),
            (width // 3, 2 * height // 3),
            (2 * width // 3, 2 * height // 3)
        ]

        # Calculate edge density in ROIs
        edges = cv2.Canny(img, 100, 200)
        roi_score = 0

        for x, y in roi_points:
            roi = edges[y - 20:y + 20, x - 20:x + 20]
            roi_score += np.sum(roi) / (40 * 40)

        return min(1.0, roi_score / (255 * 4))  # Normalize

    def detect_faces_and_score(self, img_path):
        """Detect faces using OpenCV and score based on composition"""
        try:
            # Read image
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Calculate image dimensions
            height, width = img.shape[:2]
            img_area = height * width

            # Score based on number of faces and their sizes
            face_score = len(faces) * 0.2  # Base score for number of faces

            if len(faces) > 0:
                face_areas = []
                for (x, y, w, h) in faces:
                    face_area = w * h
                    face_areas.append(face_area)

                # Score based on largest face relative to image size
                max_face_ratio = max(face_areas) / img_area
                face_score += min(1.0, max_face_ratio * 10)  # Scale up the ratio

                # Bonus for group photos with good face sizes
                if len(faces) > 1:
                    face_score *= 1.2

            return min(1.0, face_score)  # Cap at 1.0

        except Exception as e:
            print(f"Error detecting faces in {img_path}: {str(e)}")
            return 0

    def ensure_diversity(self, selected_images, features, num_clusters=10):
        """Ensure selected images are visually diverse using clustering"""
        kmeans = KMeans(n_clusters=num_clusters)
        clusters = kmeans.fit_predict(features)

        # Select representative images from each cluster
        diverse_selection = []
        for i in range(num_clusters):
            cluster_images = [img for img, cluster in zip(selected_images, clusters) if cluster == i]
            if cluster_images:
                # Take more images from larger clusters
                cluster_size = len(cluster_images)
                num_to_take = max(1, int(cluster_size * self.target_photos / len(selected_images)))
                diverse_selection.extend(cluster_images[:num_to_take])

        return diverse_selection

    def create_album(self):
        """Main function to create the wedding album"""
        print("Starting album creation...")

        # Get all images from input folder
        image_files = [f for f in os.listdir(self.input_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"Found {len(image_files)} images. Analyzing...")

        # Score all images
        image_scores = []
        for idx, img_file in enumerate(image_files):
            if idx % 20 == 0:  # Progress update
                print(f"Processing image {idx + 1}/{len(image_files)}")

            img_path = os.path.join(self.input_folder, img_file)

            # Calculate combined score
            quality_score = self.analyze_image_quality(img_path)
            face_score = self.detect_faces_and_score(img_path)

            combined_score = 0.6 * quality_score + 0.4 * face_score
            image_scores.append((img_file, combined_score))

        # Sort by score and select top images
        image_scores.sort(key=lambda x: x[1], reverse=True)
        selected_images = [img[0] for img in image_scores[:int(self.target_photos * 1.5)]]

        print("Extracting features for diversity analysis...")

        # Extract features for diversity check
        features = []
        for img_file in selected_images:
            img_path = os.path.join(self.input_folder, img_file)
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            features.append(x.flatten())

        # Ensure diversity in final selection
        final_selection = self.ensure_diversity(selected_images, features)[:self.target_photos]

        print("Creating PDF album...")
        self.create_pdf_album(final_selection)

        # Copy selected images to output folder
        print("Copying selected images...")
        for img_file in final_selection:
            src = os.path.join(self.input_folder, img_file)
            dst = os.path.join(self.output_folder, img_file)
            shutil.copy2(src, dst)

        print(f"Album created successfully with {len(final_selection)} photos!")

    def create_pdf_album(self, selected_images):
        """Create a PDF album with magazine-style layouts and debug info"""
        try:
            # Create full path for PDF
            pdf_path = os.path.join(self.output_folder, 'wedding_album.pdf')
            print(f"\nAttempting to create PDF at: {pdf_path}")

            # Create PDF
            print("Initializing PDF canvas...")
            c = canvas.Canvas(pdf_path, pagesize=A4)
            width, height = A4

            print("Adding title page...")
            # Font settings
            c.setFont("Helvetica-Bold", 24)
            title_page_text = "Wedding Album"
            c.drawString(width / 2 - 80, height - 100, title_page_text)

            # Add date
            c.setFont("Helvetica", 14)
            date_text = datetime.now().strftime("%B %Y")
            c.drawString(width / 2 - 40, height - 130, date_text)

            # Create layouts
            current_image = 0
            layouts = [
                self.full_page_layout,
                self.two_vertical_layout,
                self.three_image_layout,
                self.two_horizontal_layout,
                self.four_image_grid
            ]
            current_layout = 0

            print(f"\nProcessing {len(selected_images)} images for PDF...")
            while current_image < len(selected_images):
                print(f"Processing image {current_image + 1}/{len(selected_images)}")
                c.showPage()

                # Get next layout function
                layout_func = layouts[current_layout]
                print(f"Using layout: {layout_func.__name__}")

                # Calculate how many images this layout needs
                images_needed = self.get_layout_image_count(layout_func)

                try:
                    # Check if we have enough images left
                    if current_image + images_needed <= len(selected_images):
                        image_batch = selected_images[current_image:current_image + images_needed]
                        layout_func(c, image_batch)
                        current_image += images_needed
                    else:
                        # Not enough images for this layout, use single image layout
                        self.full_page_layout(c, [selected_images[current_image]])
                        current_image += 1
                except Exception as e:
                    print(f"Error in layout {layout_func.__name__}: {str(e)}")
                    current_image += 1

                # Rotate to next layout
                current_layout = (current_layout + 1) % len(layouts)

            print("\nSaving PDF...")
            c.save()
            print(f"PDF saved successfully at: {pdf_path}")

            # Verify the PDF was created
            if os.path.exists(pdf_path):
                print(f"PDF file size: {os.path.getsize(pdf_path)} bytes")
            else:
                print("Warning: PDF file was not found after creation!")

        except Exception as e:
            print(f"\nError creating PDF: {str(e)}")
            raise



    def get_layout_image_count(self, layout_func):
        """Return number of images needed for a given layout"""
        if layout_func == self.full_page_layout:
            return 1
        elif layout_func in (self.two_vertical_layout, self.two_horizontal_layout):
            return 2
        elif layout_func == self.three_image_layout:
            return 3
        elif layout_func == self.four_image_grid:
            return 4
        return 1

    def full_page_layout(self, canvas, images):
        """Single large image layout"""
        width, height = A4
        margin = 30

        img_path = os.path.join(self.input_folder, images[0])
        img = Image.open(img_path)
        aspect = img.width / img.height

        if aspect > width / (height - 2 * margin):
            img_width = width - 2 * margin
            img_height = img_width / aspect
        else:
            img_height = height - 2 * margin
            img_width = img_height * aspect

        x = (width - img_width) / 2
        y = (height - img_height) / 2

        canvas.drawImage(img_path, x, y, width=img_width, height=img_height)

    def two_vertical_layout(self, canvas, images):
        """Two images side by side"""
        width, height = A4
        margin = 20
        gap = 10

        img_width = (width - 2 * margin - gap) / 2

        for i, image in enumerate(images):
            img_path = os.path.join(self.input_folder, image)
            img = Image.open(img_path)
            aspect = img.width / img.height

            img_height = img_width / aspect
            x = margin + i * (img_width + gap)
            y = (height - img_height) / 2

            canvas.drawImage(img_path, x, y, width=img_width, height=img_height)

    def three_image_layout(self, canvas, images):
        """One large image with two smaller ones"""
        width, height = A4
        margin = 20
        gap = 10

        # Large image on left
        img_path = os.path.join(self.input_folder, images[0])
        img = Image.open(img_path)
        aspect = img.width / img.height

        large_width = (width - 3 * margin) * 0.6
        large_height = height - 2 * margin

        if aspect > large_width / large_height:
            large_height = large_width / aspect
        else:
            large_width = large_height * aspect

        canvas.drawImage(img_path, margin, (height - large_height) / 2,
                         width=large_width, height=large_height)

        # Two smaller images on right
        small_width = width - large_width - 4 * margin
        for i, image in enumerate(images[1:3]):
            img_path = os.path.join(self.input_folder, image)
            img = Image.open(img_path)
            aspect = img.width / img.height

            small_height = (height - 3 * margin) / 2
            if aspect > small_width / small_height:
                small_height = small_width / aspect

            x = large_width + 3 * margin
            y = margin + i * (height / 2)

            canvas.drawImage(img_path, x, y, width=small_width, height=small_height)

    def two_horizontal_layout(self, canvas, images):
        """Two images stacked vertically"""
        width, height = A4
        margin = 20
        gap = 10

        img_height = (height - 2 * margin - gap) / 2
        for i, image in enumerate(images):
            img_path = os.path.join(self.input_folder, image)
            img = Image.open(img_path)
            aspect = img.width / img.height

            img_width = img_height * aspect
            if img_width > width - 2 * margin:
                img_width = width - 2 * margin
                img_height = img_width / aspect

            x = (width - img_width) / 2
            y = margin + i * (img_height + gap)

            canvas.drawImage(img_path, x, y, width=img_width, height=img_height)

    def four_image_grid(self, canvas, images):
        """Four images in a grid"""
        width, height = A4
        margin = 20
        gap = 10

        img_width = (width - 2 * margin - gap) / 2
        img_height = (height - 2 * margin - gap) / 2

        for i, image in enumerate(images):
            img_path = os.path.join(self.input_folder, image)
            img = Image.open(img_path)
            aspect = img.width / img.height

            if aspect > img_width / img_height:
                current_height = img_width / aspect
                current_width = img_width
            else:
                current_width = img_height * aspect
                current_height = img_height

            row = i // 2
            col = i % 2

            x = margin + col * (img_width + gap)
            y = margin + row * (img_height + gap)

            canvas.drawImage(img_path, x, y, width=current_width, height=current_height)

            # Add this to your main() function to test directly:

# This should be outside the class, at the bottom of the file
def main():
    """Example usage with error handling"""
    try:
        # Set your input and output folders
        input_folder = "input_path"  # Folder containing all your photos
        output_folder = "output_path"  # Folder where the album will be created

        print(f"\nInput folder: {os.path.abspath(input_folder)}")
        print(f"Output folder: {os.path.abspath(output_folder)}")

        # Verify input folder exists and contains images
        if not os.path.exists(input_folder):
            raise ValueError(f"Input folder does not exist: {input_folder}")

        image_files = [f for f in os.listdir(input_folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            raise ValueError(f"No image files found in input folder: {input_folder}")

        print(f"Found {len(image_files)} images in input folder")

        # Create the album maker instance
        album_maker = WeddingAlbumMaker(
            input_folder=input_folder,
            output_folder=output_folder,
            target_photos=125
        )

        # Generate the album
        album_maker.create_album()

    except Exception as e:
        print(f"\nError in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()