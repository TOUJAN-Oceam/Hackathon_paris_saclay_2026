import numpy as np
import glob
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# --- CONFIGURATION ---
DATA_DIR = 'dataset_LIVE'
MODEL_FILE = 'model_eeg.pkl'

# --- FONCTION FEATURES (DOIT ETRE STRICTEMENT IDENTIQUE AU TRAINING) ---
def compute_features(fp1, fp2):
    feat = []
    for canal in [fp1, fp2]:
        # 1. Line Length
        ll = np.sum(np.abs(np.diff(canal)))
        # 2. Variance
        var = np.var(canal)
        # 3. ZCR
        canal_centered = canal - np.mean(canal)
        zcr = np.count_nonzero(np.diff(np.sign(canal_centered)))
        # 4. Max Amp
        amp = np.max(np.abs(canal))
        
        feat.extend([ll, var, zcr, amp])
    return feat

def evaluer_demo():
    # 1. Chargement du modèle
    if not os.path.exists(MODEL_FILE):
        print(f"❌ ERREUR: Modèle {MODEL_FILE} introuvable.")
        return
    
    print(f"Chargement du modèle {MODEL_FILE}...")
    model = joblib.load(MODEL_FILE)
    
    # 2. Récupération des fichiers
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.npy')))
    if not files:
        print(f"❌ ERREUR: Aucun fichier dans {DATA_DIR}.")
        return

    print(f"Analyse de {len(files)} fichiers de démo...")

    y_true = []
    y_pred = []
    erreurs = []

    # 3. Boucle de Prédiction
    for f in files:
        filename = os.path.basename(f)
        
        # A. Extraction de la Vérité Terrain depuis le nom
        # Ex: live_045_CRISE.npy -> Label = 1
        if "_CRISE" in filename:
            label_reel = 1
        elif "_CALME" in filename:
            label_reel = 0
        else:
            print(f"⚠️ Fichier ignoré (pas de label dans le nom) : {filename}")
            continue
            
        # B. Prédiction Modèle
        try:
            data = np.load(f)
            fp1, fp2 = data[0]*1000000, data[1]*1000000 # µV
            
            features = compute_features(fp1, fp2)
            features_array = np.array(features).reshape(1, -1)
            
            pred = model.predict(features_array)[0] # 0 ou 1
            
            y_true.append(label_reel)
            y_pred.append(pred)
            
            if label_reel != pred:
                erreurs.append(filename)
                
        except Exception as e:
            print(f"Erreur lecture {filename}: {e}")

    # 4. Affichage des Résultats
    print("\n" + "="*50)
    print(" RÉSULTATS DE L'ÉVALUATION ")
    print("="*50)
    
    print(classification_report(y_true, y_pred, target_names=['Calme (0)', 'Crise (1)']))
    
    f1 = f1_score(y_true, y_pred)
    print(f"🎯 F1-SCORE GLOBAL : {f1:.4f}")
    
    print("\n--- MATRICE DE CONFUSION ---")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Vrais Négatifs (Calme bien détecté) : {tn}")
    print(f"Vrais Positifs (Crise bien détectée): {tp}")
    print(f"Faux Positifs (Fausse Alerte)       : {fp}")
    print(f"Faux Négatifs (Crise Ratée !!!)     : {fn}")
    
    if fn > 0:
        print("\n❌ ATTENTION : Tu as raté des crises ! Vérifie ces fichiers :")
        print(erreurs)
    elif fp > 0:
        print("\n⚠️ ATTENTION : Tu as des fausses alertes. Vérifie ces fichiers :")
        print(erreurs)
    else:
        print("\n✅ PARFAIT ! Ton modèle fait un sans-faute sur le scénario.")

    # 5. Bonus Visuel (Matrice)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred: Calme', 'Pred: Crise'], yticklabels=['Vrai: Calme', 'Vrai: Crise'])
    plt.title('Performance Modèle sur Scénario Live')
    plt.ylabel('Vérité Terrain')
    plt.xlabel('Prédiction Modèle')
    plt.show()

if __name__ == '__main__':
    evaluer_demo()