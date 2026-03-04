# README NeuroDEEp

> **Système de surveillance épileptique intelligent et collaboratif.**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg) ![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg) ![Chart.js](https://img.shields.io/badge/Frontend-Chart.js-orange.svg) ![Status](https://img.shields.io/badge/Status-Hackathon_Prototype-red.svg)

---

## Contexte
Le projet gitlab que vous voyez ici est le fruit de 40h de développement dans le cadre du **Hackathon Paris-Saclay SynaPCS 2026**.

La problématique à laquelle nous devions répondre était la suivante : Développer des méthodes moins contraignantes et non invasives (imagerie MRI, EEG, digital device...) pour montrer l'efficacité des médicaments/la réponse au traitement dans les épilepsies développementales rares pédiatriques.

Pour plus d'information, vous pouvez vous référer à [ce lien](https://www.linkedin.com/feed/update/urn:li:ugcPost:7425238456177504257/?originTrackingId=kUE120PJcLosmcENTRtpwA%3D%3D)



## Installation

On a utilisé l'outils [uv](https://github.com/astral-sh/uv) pour une gestion ultra-rapide des dépendances et de l'environnement virtuel, ainsi, pour l'installer si ce n'est pas fait, réferez-vous au lien github ci dessus.

Si vous décidez d'utiliser l'outils pip à la place, vous trouverez un Pour l'utiliser, vous trouverez un pyproject.toml avec les librairies et leurs versions.

## Lancement 

Pour lancer l'application, veuillez vous placer dans un terminal à la racine du projet, et faites un 

uv sync

Ensuite, vous pourrez lancer le serveur avec : 

uv run python serveur_flask.py

et y accéder sur un navigateur (exemple d'adresse locale : http://127.0.0.1:5000)

## Présentation des fichiers du GitLab : 

- serveur_flask.py : Le script qui lance le serveur, et se charge sur un navigateur en local.

- test_algo.py, decouper_eeg.py, evaluer_modele.py, scenario_demo.py, (et le modèle : model_eeg.pkl) sont des scripts pour respectivement créer notre base de donnée, entrainer un modèle dessus, tester la robustesse du modèle, et créer des données cohérentes pour la démonstration. Ces scripts sont très mal structurés, dû au court temps disponible dans le cadre du hackathon, mais cela permet de faire une vitrine de démonstration très satisfaisante.

- Les différents dossiers, ainsi que historique_crises_dense.csv contiennent des données utilisées par le serveur pour la démonstration.


## Présentation des interfaces serveurs par gémini : 

### 🏠 Espace Famille (Suivi & Rassurant)

Pour se connecter, user : parent et password : mdp_famille

* **Monitoring Live :** Indicateur simple "Stable" / "Crise en cours" 🟢🔴.
* **Statistiques :** Visualisation de la fréquence des crises sur 7, 30 et 90 jours (Graphes interactifs).
* **Bilan de santé :** Remplissage simplifié du questionnaire QOLIE-31 (Qualité de vie) via une interface modale.
* **Messagerie :** Chat direct et sécurisé avec le neurologue.
* **Historique :** Journal complet des événements exportable.

### 🩺 Espace Praticien (Clinique & Précis)

Pour se connecter, user : doc et password : mdp_hopital

* **Scope EEG Temps Réel :** Visualisation fluide du signal (FP1/FP2) avec défilement fluide ("Smooth Scrolling").
* **Alertes Visuelles :** Changement de couleur des courbes en cas de détection positive par l'IA.
* **Dossier Patient :** Accès immédiat aux derniers bilans QOLIE-31 remplis par la famille.
* **Analyse Tendance :** Graphiques d'évolution pour adapter le traitement.

### ⚙️ Espace Admin / Debug

Pour se connecter, user : admin et password : mdp_admin

* **IA Monitoring :** Visualisation de la décision "Brute" vs "Lissée" (Vote majoritaire).
* **Performance :** Mesure de la latence et de la gigue réseau.

---

## 🛠️ Stack Technique

* **Backend :** Python (Flask), Numpy (Traitement signal), Pandas (CSV/Stats).
* **IA / ML :** Scikit-Learn (Simple Random Forest pré-entraîné `model_eeg.pkl`).
* **Frontend :** HTML5, Bootstrap 5, Chart.js (Rendu graphique haute fréquence).
* **Données :** Simulation de flux temps réel via lecture de fichiers `.npy` (Dataset EEG).

