Album Maker

album_maker.py is a Python script designed to generate image albums from a collection of images. It allows users to create custom albums by arranging images in a specified format, adding captions, and exporting the final album in a desired format.

Features

	•	Image Organization: Automatically arranges images into album layouts.
	•	Custom Captions: Add captions to individual images or pages.
	•	Export Options: Save the album in various formats (e.g., PDF, image files).
	•	Flexible Layouts: Choose from multiple album templates to suit your needs.
	•	Easy Integration: Can be extended to work with additional image sources or formats.

Requirements

	•	Python 3.6+
	•	Required libraries (install via pip install -r requirements.txt):
	•	Pillow: For image processing
	•	reportlab (if generating PDFs)
	•	Any other dependencies used in the script

Installation

	1.	Clone this repository or download the script file.
	2.	Ensure you have Python installed on your system.
	3.	Install the required dependencies:

pip install -r requirements.txt



Usage

	1.	Prepare your image files in a folder.
	2.	Run the script:

python album_maker.py


	3.	Follow the prompts to:
	•	Select the folder containing your images.
	•	Specify album layout preferences.
	•	Add captions (if desired).
	•	Export the album.
	4.	The generated album will be saved in the output directory.

Example

Here’s an example command to create an album from a folder named my_images:

python album_maker.py --input ./my_images --output ./albums --layout grid --export pdf

This command will generate a PDF album with a grid layout and save it in the albums folder.

Customization

You can customize the script for:
	•	New Layouts: Add more layout options by editing the layout logic in the script.
	•	Additional Export Formats: Integrate with libraries like img2pdf or others for new export options.
	•	Styling: Modify album appearance, such as borders, colors, and fonts.

Troubleshooting

	•	Missing Images: Ensure your input directory contains valid image files supported by Pillow.
	•	Dependency Errors: Ensure all dependencies are correctly installed.

License

This project is licensed under the MIT License. See the LICENSE file for details.

