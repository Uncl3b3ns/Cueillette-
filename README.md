# Cueillette- 🌿🍄

**Cueillette-** est une application web interactive conçue pour aider les utilisateurs à identifier les zones optimales pour la cueillette de divers fruits et légumes en fonction du pH du sol, des types de végétation et des conditions météorologiques. En tirant parti des données géospatiales et des prévisions météorologiques, Cueillette- fournit des cartes dynamiques mettant en évidence les zones favorables pour différentes cultures selon les saisons.

## Table des Matières

- [Fonctionnalités](#fonctionnalités)
- [Démo](#démo)
- [Installation](#installation)
  - [Prérequis](#prérequis)
  - [Étapes d'Installation](#étapes-dinstallation)
- [Utilisation](#utilisation)
  - [Flux de Travail de l'Application](#flux-de-travail-de-lapplication)
  - [GitHub Actions](#github-actions)
- [Structure du Projet](#structure-du-projet)
- [Extension de l'Application](#extension-de-lapplication)
- [Sécurité](#sécurité)
- [Contribuer](#contribuer)
- [Licence](#licence)

## Fonctionnalités

- **Sélection Saisonnière** : Choisissez des fruits et légumes catégorisés par saisons (Automne, Hiver, Printemps, Été).
- **Cartographie Dynamique** : Sélectionnez différents types de cartes pour visualiser les niveaux de pH du sol, les types de végétation et les superpositions météorologiques.
- **Intégration Météo** : Visualisez des cartes météorologiques avec des indicateurs de zones favorables basées sur les conditions météorologiques actuelles et prévues.
- **Interface Interactive** : Menus déroulants conviviaux et iframes pour naviguer facilement entre différentes vues et couches de données.
- **Mises à Jour Automatisées** : Traitement quotidien des données et génération de cartes via GitHub Actions, garantissant des informations à jour.

## Démo

Accédez à l'application en direct [ici](https://Uncl3b3ns.github.io/Cueillette-/).

## Installation

Suivez ces instructions pour configurer et exécuter l'application localement ou comprendre son processus de déploiement.

### Prérequis

- **Python 3.9** ou supérieur
- **Compte GitHub** avec accès pour créer des dépôts et gérer GitHub Actions
- **Git** installé sur votre machine locale
- **GitHub Pages** activé pour votre dépôt

### Étapes d'Installation

1. **Cloner le Dépôt**

   ```bash
   git clone https://github.com/Uncl3b3ns/Cueillette-.git
   cd Cueillette-
Configurer un Environnement Virtuel

Il est recommandé d'utiliser un environnement virtuel pour gérer les dépendances.

bash
Copier le code
python3 -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
Installer les Dépendances

Assurez-vous d'avoir un fichier requirements.txt avec les bibliothèques nécessaires.

bash
Copier le code
pip install --upgrade pip
pip install -r requirements.txt
Exemple de requirements.txt :

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
Configurer les Variables d'Environnement

Créez un fichier .env à la racine du projet et ajoutez votre token GitHub.

env
Copier le code
GITHUB_TOKEN=your_github_token_here
⚠️ Remarque de Sécurité : Assurez-vous que votre fichier .env n'est jamais commité dans le dépôt. Ajoutez-le à votre .gitignore si nécessaire.

Configurer les Secrets GitHub

Naviguez vers votre dépôt GitHub.
Allez dans Settings > Secrets and variables > Actions.
Cliquez sur New repository secret et ajoutez votre token GitHub avec le nom MY_GH_TOKEN.
Utilisation
Flux de Travail de l'Application
Interface de Sélection

Sélection de Fruit/Légume : Choisissez votre culture désirée catégorisée par saison depuis le premier menu déroulant.
Sélection du Type de Carte : Sélectionnez le type de carte que vous souhaitez visualiser :
pH + Végétation : Affiche les niveaux de pH du sol et les types de végétation.
pH + Végétation + Météo : Inclut les superpositions de données météorologiques.
Sélection du Jour : Si vous choisissez le type de carte météorologique, sélectionnez le jour spécifique pour visualiser les prévisions météorologiques.
Affichage Dynamique de la Carte

En fonction de vos sélections, l'application charge dynamiquement la carte HTML correspondante dans l'iframe. Si l'option météorologique est sélectionnée, elle affiche également le nombre de zones favorables basé sur les conditions météorologiques.

GitHub Actions
L'application utilise GitHub Actions pour automatiser le traitement quotidien des données et la génération des cartes.

Configuration du Workflow

Le workflow est défini dans .github/workflows/daily_script.yml et est déclenché quotidiennement à 3h du matin UTC.

yaml
Copier le code
name: Daily script

on:
  workflow_dispatch:  # Déclenchement manuel
  schedule:
    - cron: '0 3 * * *'  # Exécute tous les jours à 3h UTC

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
          GITHUB_TOKEN: ${{ secrets.MY_GH_TOKEN }}  # Secret GitHub Token
        run: |
          python Script_meteo_github.py
Étapes du Workflow

Checkout du Dépôt : Récupère le code le plus récent du dépôt.
Configuration de Python : Configure l'environnement Python.
Installation des Dépendances : Installe les bibliothèques Python nécessaires.
Exécution du Script Python : Exécute Script_meteo_github.py pour traiter les données et uploader les fichiers HTML.
Structure du Projet
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
└── .env  # Non commité
.github/workflows/daily_script.yml : Définit le workflow GitHub Actions.
meteo_data/ : Stocke les données météorologiques XML téléchargées.
meteo_rasters/ : Contient les cartes HTML générées pour chaque culture et jour.
ph_final_3857.tif & clc_final_3857.tif : Fichiers raster pour le pH du sol et la classification de couverture terrestre.
ph_veg_cepes.html : Carte HTML affichant les données pH et végétation.
index.html : Interface principale pour l'interaction utilisateur.
Script_meteo_github.py : Script Python pour le traitement des données et la génération des cartes.
requirements.txt : Liste des dépendances Python.
README.md : Documentation du projet.
.env : Variables d'environnement (non commité).
Extension de l'Application
Pour ajouter de nouveaux fruits ou légumes à l'application :

Mettre à Jour le Menu Déroulant

Modifiez les sections <optgroup> dans index.html pour inclure les nouvelles cultures sous la saison appropriée.

html
Copier le code
<optgroup label="Nouvelle Saison">
    <option value="nouvelle_culture">Nouvelle Culture</option>
    <!-- Ajoutez plus d'options si nécessaire -->
</optgroup>
Générer les Cartes HTML Correspondantes

Mettez à jour Script_meteo_github.py pour gérer les nouvelles cultures en générant ph_veg_<crop>.html et les cartes météorologiques dans meteo_rasters/<crop>/.

Exécuter le Script

Exécutez le script Python manuellement ou attendez la prochaine exécution planifiée de GitHub Actions pour générer et uploader les nouvelles cartes.

Sécurité
Gestion des Tokens GitHub
Révoquer les Tokens Exposés : Si votre token GitHub a été exposé publiquement, révoquez-le immédiatement via Paramètres GitHub.
Générer un Nouveau Token : Créez un nouveau Personal Access Token avec les permissions nécessaires (repo scope) et mettez à jour les secrets du dépôt.
Stockage Sécurisé : Stockez les tokens de manière sécurisée en utilisant les Secrets GitHub et ne les commitez jamais dans le dépôt.
Bonnes Pratiques
Limiter les Permissions des Tokens : Accordez uniquement les permissions minimales nécessaires à vos tokens GitHub.
Rotation Régulière des Tokens : Mettez à jour périodiquement vos tokens pour minimiser les risques de sécurité.
Surveiller l'Accès au Dépôt : Gardez un œil sur qui a accès à votre dépôt et à ses secrets.
Contribuer
Les contributions sont les bienvenues ! Veuillez suivre ces directives :

Forker le Dépôt

Cliquez sur le bouton Fork en haut à droite de la page du dépôt.

Créer une Nouvelle Branche

bash
Copier le code
git checkout -b feature/YourFeatureName
Faire des Modifications

Implémentez votre fonctionnalité ou corrigez des bugs.

Commiter les Changements

bash
Copier le code
git commit -m "Ajoute votre message de commit descriptif"
Pousser la Branche

bash
Copier le code
git push origin feature/YourFeatureName
Ouvrir une Pull Request

Rendez-vous sur le dépôt GitHub et cliquez sur Compare & pull request.

Licence
Ce projet est sous licence MIT.
