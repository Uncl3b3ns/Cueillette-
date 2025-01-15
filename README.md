Cueillette- 🌿🍄
Cueillette- is an interactive web application designed to help users identify optimal zones for foraging various fruits and vegetables based on soil pH, vegetation types, and meteorological conditions. By leveraging geospatial data and weather forecasts, Cueillette- provides dynamic maps that highlight favorable areas for different crops across the seasons.

Table of Contents
Features
Demo
Getting Started
Prerequisites
Installation
Usage
Application Workflow
GitHub Actions
Project Structure
Extending the Application
Security
Contributing
License
Features
Seasonal Selection: Choose fruits and vegetables categorized by seasons (Autumn, Winter, Spring, Summer).
Dynamic Mapping: Select different map types to view soil pH, vegetation, and weather overlays.
Weather Integration: View meteorological maps with indicators of favorable zones based on current and forecasted weather data.
Interactive Interface: User-friendly dropdown menus and iframes to seamlessly switch between different views and data layers.
Automated Updates: Daily data processing and map generation through GitHub Actions, ensuring up-to-date information.
Demo
Access the live application here.

Getting Started
Follow these instructions to set up and run the application locally or understand its deployment process.

Prerequisites
Python 3.9 or higher
GitHub Account with access to create repositories and manage GitHub Actions
Git installed on your local machine
GitHub Pages enabled for your repository
Installation
Clone the Repository

bash
Copier le code
git clone https://github.com/Uncl3b3ns/Cueillette-.git
cd Cueillette-
Set Up Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

bash
Copier le code
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies

Ensure you have a requirements.txt file with the necessary libraries.

bash
Copier le code
pip install --upgrade pip
pip install -r requirements.txt
Sample requirements.txt:

plaintext
Copier le code
numpy
requests
rasterio
folium
beautifulsoup4
scipy
tqdm
python-dotenv
Configure Environment Variables

Create a .env file in the root directory and add your GitHub token.

env
Copier le code
GITHUB_TOKEN=your_github_token_here
⚠️ Security Notice: Ensure that your .env file is never committed to the repository. Add it to your .gitignore if necessary.

Set Up GitHub Secrets

Navigate to your GitHub repository.
Go to Settings > Secrets and variables > Actions.
Click on New repository secret and add your GitHub token with the name MY_GH_TOKEN.
Usage
Application Workflow
Selection Interface

Fruit/Vegetable Selection: Choose your desired crop categorized by season from the first dropdown menu.
Map Type Selection: Select the type of map you want to view:
pH + Végétation: Displays soil pH levels and vegetation types.
pH + Végétation + Météo: Includes meteorological data overlays.
Day Selection: If you choose the meteorological map type, select the specific day to view weather forecasts.
Dynamic Map Display

Based on your selections, the application dynamically loads the corresponding HTML map in the iframe. If the meteorological option is selected, it also displays the number of favorable zones based on weather conditions.

GitHub Actions
The application leverages GitHub Actions to automate daily data processing and map generation.

Workflow Configuration

The workflow is defined in .github/workflows/daily_script.yml and is triggered daily at 3 AM UTC.

yaml
Copier le code
name: Daily script

on:
  workflow_dispatch:  # Manual trigger
  schedule:
    - cron: '0 3 * * *'  # Runs daily at 3 AM UTC

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Check out
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run Script_meteo_github.py
        env:
          GITHUB_TOKEN: ${{ secrets.MY_GH_TOKEN }}  # GitHub Token Secret
        run: |
          python Script_meteo_github.py
Workflow Steps

Checkout Repository: Retrieves the latest code from the repository.
Set Up Python: Configures the Python environment.
Install Dependencies: Installs required Python libraries.
Run Python Script: Executes Script_meteo_github.py to process data and upload HTML files.
Project Structure
bash
Copier le code
Cueillette-/
├── .github/
│   └── workflows/
│       └── daily_script.yml
├── meteo_data/
│   └── meteo_*.xml
├── meteo_rasters/
│   └── <crop>/
│       └── meteo_j*.html
├── ph_final_3857.tif
├── clc_final_3857.tif
├── ph_veg_cepes.html
├── index.html
├── Script_meteo_github.py
├── requirements.txt
├── README.md
└── .env  # Not committed
.github/workflows/daily_script.yml: Defines the GitHub Actions workflow.
meteo_data/: Stores downloaded meteorological XML data.
meteo_rasters/: Contains generated HTML maps for each crop and day.
ph_final_3857.tif & clc_final_3857.tif: Raster files for soil pH and land cover classification.
ph_veg_cepes.html: HTML map displaying soil pH and vegetation data.
index.html: Main interface for user interaction.
Script_meteo_github.py: Python script for data processing and map generation.
requirements.txt: Lists Python dependencies.
README.md: Project documentation.
.env: Environment variables (not committed).
Extending the Application
To add new fruits or vegetables to the application:

Update the Dropdown Menu

Modify the <optgroup> sections in index.html to include new crops under the appropriate season.

html
Copier le code
<optgroup label="Nouvelle Saison">
    <option value="nouvelle_culture">Nouvelle Culture</option>
    <!-- Add more options as needed -->
</optgroup>
Generate Corresponding HTML Maps

Update Script_meteo_github.py to handle the new crops by generating ph_veg_<crop>.html and meteorological maps within meteo_rasters/<crop>/.

Run the Script

Execute the Python script manually or wait for the next scheduled GitHub Actions run to generate and upload the new maps.

Security
GitHub Token Management
Revoke Exposed Tokens: If your GitHub token has been exposed publicly, immediately revoke it via GitHub Settings.
Generate a New Token: Create a new Personal Access Token with the necessary permissions (repo scope) and update the repository secrets.
Secure Storage: Store tokens securely using GitHub Secrets and never commit them to the repository.
Best Practices
Limit Token Permissions: Only grant the minimum necessary permissions to your GitHub tokens.
Regularly Rotate Tokens: Periodically update your tokens to minimize security risks.
Monitor Repository Access: Keep track of who has access to your repository and its secrets.
Contributing
Contributions are welcome! Please follow these guidelines:

Fork the Repository

Click the Fork button at the top-right corner of the repository page.

Create a New Branch

bash
Copier le code
git checkout -b feature/YourFeatureName
Make Changes

Implement your feature or fix bugs.

Commit Changes

bash
Copier le code
git commit -m "Add your descriptive commit message"
Push to the Branch

bash
Copier le code
git push origin feature/YourFeatureName
Open a Pull Request

Navigate to the repository on GitHub and click Compare & pull request.

License
This project is licensed under the MIT License.

Made with ❤️ by Uncl3b3ns
