from flask import Flask, render_template, redirect, url_for, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, current_user, logout_user
import numpy as np
import os
import glob
import random
import datetime
import csv  # Nécessaire pour le CSV
import joblib


app = Flask(__name__)
app.secret_key = 'hackathon_secret' 

# --- CONFIGURATION LOGIN ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

USERS = {
    "parent":   {"password": "mdp_famille", "role": "parent", "name": "M. Dupont"},
    "doc":      {"password": "mdp_hopital", "role": "praticien", "name": "Dr. House"},
    "admin":    {"password": "mdp_admin",   "role": "admin", "name": "Super Admin"}
}

# --- VARIABLES GLOBALES DE MÉMOIRE (STATE MACHINE) ---
SEIZURE_STATE = {
    "is_active": False,
    "start_time": None,
    "max_ll": 0,    # Pour garder le pic d'intensité
    "duration": 0   # Nombre d'époques concernées
}

LAST_SEIZURE_TIME = None

model = joblib.load('model_eeg.pkl')




# --- A AJOUTER AVANT detect_seizure_advanced ---
def compute_features(fp1, fp2):
    feat = []
    for canal in [fp1, fp2]:
        # Doit être IDENTIQUE au script d'entraînement (train_model.py)
        ll = np.sum(np.abs(np.diff(canal)))
        var = np.var(canal)
        canal_centered = canal - np.mean(canal)
        zcr = np.count_nonzero(np.diff(np.sign(canal_centered)))
        amp = np.max(np.abs(canal))
        feat.extend([ll, var, zcr, amp])
    return feat

# --- FONCTION DE DÉTECTION AVANCÉE (LL + ZCR) ---
def detect_seizure_advanced(fp1, fp2):
    # 1. Prétraitement : On centre le signal (retirer la moyenne)
    # C'est crucial pour le calcul du ZCR
    fp1 = fp1 - np.mean(fp1)
    fp2 = fp2 - np.mean(fp2)

    # 2. Calcul Line Length (L'intensité)
    ll_fp1 = np.sum(np.abs(np.diff(fp1)))
    ll_fp2 = np.sum(np.abs(np.diff(fp2)))
    avg_ll = (ll_fp1 + ll_fp2) / 2

    # 3. Calcul Zero Crossing Rate (La fréquence/nervosité)
    # On compte combien de fois le signal change de signe
    zcr_fp1 = np.count_nonzero(np.diff(np.sign(fp1)))
    zcr_fp2 = np.count_nonzero(np.diff(np.sign(fp2)))
    avg_zcr = (zcr_fp1 + zcr_fp2) / 2

    # 4. Autres stats
    var = (np.var(fp1) + np.var(fp2)) / 2
    max_amp = max(np.max(np.abs(fp1)), np.max(np.abs(fp2)))

    # --- LOGIQUE DE DÉCISION (LE COEUR DU MODÈLE) ---
    is_seizure = False
    
    # SEUILS À CALIBRER (Regarde tes prints !)
    # Si le signal est très fort (LL) ET qu'il bouge vite (ZCR), c'est une crise.
    # Si le signal est fort mais lent (ZCR bas), c'est un clignement d'yeux.
    
    SEUIL_LL = 2000000  # Seuil d'intensité
    SEUIL_ZCR = 50      # Seuil de fréquence (ex: doit traverser 50 fois le zéro en 30s)

    # Calculer les features exactement comme dans l'entraînement
    features = compute_features(fp1, fp2) # (Copie la fonction compute_features du script train)

    # Prédire
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0] # 0 ou 1
    
    # CONVERSION EXPLICITE POUR FLASK
    is_seizure = bool(prediction == 1) 
    
    # On convertit aussi les stats au cas où (float Python)
    return is_seizure, float(avg_ll), float(avg_zcr), float(var), float(max_amp)


class User(UserMixin):
    def __init__(self, id, role, name):
        self.id = id
        self.role = role
        self.name = name

@login_manager.user_loader
def load_user(user_id):
    if user_id in USERS:
        u = USERS[user_id]
        return User(user_id, u['role'], u['name'])
    return None

# --- CONFIGURATION DATASET & LOGS ---
# Dans serveur_flask.py
DATA_DIR = 'dataset_LIVE'
FILES_LIST = sorted(glob.glob(os.path.join(DATA_DIR, '*.npy')))

# --- AJOUTE CES 3 LIGNES POUR VÉRIFIER ---
print(f"--- DÉBUG ---")
print(f"Dossier visé : {os.path.abspath(DATA_DIR)}")
print(f"Fichiers trouvés : {len(FILES_LIST)}")
print(f"----------------------------------")

current_file_index = 0
CSV_FILE = 'historique_crises.csv'


# Initialisation du CSV si inexistant
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # ✅ BONNE VERSION (Correspond à votre CSV actuel)
        writer.writerow(["Date_Heure", "Duree_sec", "Intensite_LL", "Intervalle_Precedente", "Type"])
# --- VARIABLES GLOBALES TCHAT (Code précédent) ---
MESSAGES = []

# --- ROUTES ---
@app.route('/')
def index(): return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in USERS and USERS[username]['password'] == password:
            u = USERS[username]
            login_user(User(username, u['role'], u['name']))
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/dashboard')
@login_required
def dashboard():
    role = current_user.role
    if role == 'parent': return render_template('dash_parent.html', user=current_user)
    elif role == 'praticien': return render_template('dash_doc.html', user=current_user)
    elif role == 'admin': return render_template('dash_admin.html', user=current_user)
    return "Rôle inconnu"

@app.route('/logout')
@login_required
def logout(): logout_user(); return redirect(url_for('login'))

# ... (imports et configurations précédents) ...

# --- ROUTE MANQUANTE A AJOUTER (Vérifiez qu'elle est bien présente et indentée comme ceci) ---
@app.route('/api/get_patient_history')
@login_required
def get_patient_history():
    history = []
    # Vérification de l'existence du fichier CSV
    if os.path.exists(CSV_FILE):
        try:
            with open(CSV_FILE, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # On vérifie que la ligne n'est pas vide et contient une date
                    if row.get("Date_Heure"):
                        history.append({
                            "date": row.get("Date_Heure", "N/A"),
                            "duree": row.get("Duree_sec", "0"),
                            "intensite": row.get("Intensite_LL", "0"),
                            "intervalle": row.get("Intervalle_Precedente", "-"),
                            "type": row.get("Type", "Inconnu")
                        })
                history.reverse() # Plus récent en haut
        except Exception as e:
            print(f"Erreur CSV: {e}")
            # En cas d'erreur, on peut renvoyer une liste vide ou l'erreur pour le debug
            return jsonify([]) 
    
    return jsonify(history)

# --- API MESSAGERIE (Gardée du tour précédent) ---
@app.route('/api/send_message', methods=['POST'])
@login_required
def send_message():
    data = request.json
    MESSAGES.append({
        'from': data.get('force_role', current_user.role),
        'author': data.get('force_name', current_user.name),
        'content': data.get('message'),
        'time': datetime.datetime.now().strftime("%H:%M")
    })
    return jsonify({'status': 'ok'})

@app.route('/api/get_messages')
@login_required
def get_messages(): return jsonify(MESSAGES[-20:])

@app.route('/api/get_live_data')
@login_required
def get_live_data():
    global current_file_index, SEIZURE_STATE
    
    if not FILES_LIST: return jsonify({"error": "No data"})
    if current_file_index >= len(FILES_LIST): current_file_index = 0
    
    filename = FILES_LIST[current_file_index]
    filename_base = os.path.basename(filename) # ex: live_005_CRISE.npy
    
    try:
        data_array = np.load(filename)
        fp1 = data_array[0] * 1000000 
        fp2 = data_array[1] * 1000000 
        
        # 1. DÉTECTION IA
        # On appelle ta fonction (qui utilise le modèle joblib)
        is_seizure_pred, ll, zcr, var, amp = detect_seizure_advanced(fp1, fp2)
        
        # Correction JSON importante : Convertir numpy.bool_ en python bool
        is_seizure_pred = bool(is_seizure_pred)

        # 2. VÉRITÉ TERRAIN (LECTURE DU NOM DE FICHIER)
        # C'est ici qu'on remplace l'ancienne logique de temps
        if "CRISE" in filename_base:
            is_real_seizure = True
            label_debug = "VRAIE CRISE"
        else:
            is_real_seizure = False
            label_debug = "CALME"

        # 3. GESTION ETAT (inchangé)
        status = "Stable"
        status_color = "success"

        if is_seizure_pred:
            if not SEIZURE_STATE["is_active"]:
                SEIZURE_STATE["is_active"] = True
                SEIZURE_STATE["start_time"] = datetime.datetime.now()
                SEIZURE_STATE["max_ll"] = ll
                SEIZURE_STATE["duration"] = 5
            else:
                SEIZURE_STATE["duration"] += 5
                if ll > SEIZURE_STATE["max_ll"]: SEIZURE_STATE["max_ll"] = ll
            
            status = "CRISE DÉTECTÉE (IA)"
            status_color = "danger"
        else:
            # CAS : Fin de crise (C'était actif, maintenant c'est calme)
            if SEIZURE_STATE["is_active"]:
                global LAST_SEIZURE_TIME # On a besoin de modifier la variable globale

                print("<<< FIN DE CRISE. SAUVEGARDE CSV.")
                
                # 1. Calcul de l'intervalle avec la crise d'avant
                interval_str = "Premiere crise" # Par défaut
                current_start = SEIZURE_STATE["start_time"]
                
                if LAST_SEIZURE_TIME is not None:
                    # Calcul de la différence (delta)
                    delta = current_start - LAST_SEIZURE_TIME
                    # On le formate joliment (ex: "0:30:00" pour 30 minutes)
                    interval_str = str(delta).split('.')[0] # On enlève les microsecondes
                
                # Mémorisation pour la prochaine fois
                LAST_SEIZURE_TIME = current_start

                # 2. Ecriture dans le CSV
                with open(CSV_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    start_str = SEIZURE_STATE["start_time"].strftime("%Y-%m-%d %H:%M:%S")
                    
                    writer.writerow([
                        start_str, 
                        SEIZURE_STATE['duration'], 
                        round(SEIZURE_STATE['max_ll'], 2), 
                        interval_str,  # <--- Nouvelle info ajoutée
                        "Crise Generalisee"
                    ])
                
                # Reset
                SEIZURE_STATE["is_active"] = False
                SEIZURE_STATE["duration"] = 0

        current_file_index += 1
        
        return jsonify({
            "fp1": fp1[::5].tolist(), 
            "fp2": fp2[::5].tolist(),
            "parent_status": status,
            "parent_color": status_color,
            "is_seizure": is_seizure_pred, 
            "debug_label": label_debug,    # On renvoie le label lu dans le fichier
            "debug_match": bool(is_seizure_pred == is_real_seizure), # Comparaison
            "stats": {
                "max": round(float(amp), 2),
                "var": round(float(var), 2),
                "mean": round(float(ll), 2),
                "std": round(float(zcr), 2)
            },
            "filename": filename_base
        })

    except Exception as e:
        print(f"Erreur: {e}")
        return jsonify({"error": str(e)})



@app.route('/api/get_stats')
@login_required
def get_stats():
    # 1. Récupérer la période demandée (défaut = 7 jours)
    days = int(request.args.get('days', 7))
    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
    
    crises_in_range = []
    durations = []
    timestamps = []

    # 2. Lire le CSV
    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # On utilise le bon nom de colonne 'Date_Heure'
                    row_date = datetime.datetime.strptime(row['Date_Heure'], "%Y-%m-%d %H:%M:%S")

                    if row_date > cutoff_date:
                        crises_in_range.append(row)
                        # On nettoie la durée (enlève " sec" si présent et convertit en float)
                        duree_clean = float(row['Duree_sec'].replace(' sec', ''))
                        durations.append(duree_clean)
                        timestamps.append(row_date)
                except Exception as e:
                    continue # On saute les lignes mal formées

    # 3. Calculs Mathématiques
    count = len(crises_in_range)
    avg_duration = round(sum(durations) / count) if count > 0 else 0
    
    # Calcul intervalle moyen (temps entre deux crises)
    avg_interval_hours = 0
    if count > 1:
        timestamps.sort()
        # Calcul des différences entre dates consécutives
        diffs = [(t2 - t1).total_seconds() for t1, t2 in zip(timestamps[:-1], timestamps[1:])]
        avg_seconds = sum(diffs) / len(diffs)
        avg_interval_hours = round(avg_seconds / 3600, 1) # En heures

    return jsonify({
        "count": count,
        "avg_duration": avg_duration, # en secondes
        "avg_interval": avg_interval_hours # en heures
    })





if __name__ == '__main__':
    # host='0.0.0.0' signifie "Écoute tout le monde sur le réseau"
    app.run(host='0.0.0.0', port=5000, debug=True)