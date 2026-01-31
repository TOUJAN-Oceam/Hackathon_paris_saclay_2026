import numpy as np
import glob
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

# --- FEATURE ENGINEERING ---
def compute_features(fp1, fp2):  
    feat = []
    for canal in [fp1, fp2]:
        # 1. Line Length
        ll = np.sum(np.abs(np.diff(canal)))
        # 2. Variance
        var = np.var(canal)
        # 3. ZCR (Zero Crossing Rate) - centré
        canal_centered = canal - np.mean(canal)
        zcr = np.count_nonzero(np.diff(np.sign(canal_centered)))
        # 4. Max Amplitude
        amp = np.max(np.abs(canal))
        
        feat.extend([ll, var, zcr, amp])
    return feat

# --- CHARGEMENT ---
X = []
y = []

print("Chargement des données d'entraînement...")

# 1. Chargement CALME
files_calme = glob.glob('dataset_TRAIN/calme/*.npy')
for f in files_calme:
    data = np.load(f)
    # --- CORRECTION CRITIQUE : CONVERSION EN µV ---
    fp1 = data[0] * 1000000
    fp2 = data[1] * 1000000
    
    X.append(compute_features(fp1, fp2))
    y.append(0) 

# 2. Chargement CRISE
files_crise = glob.glob('dataset_TRAIN/crise/*.npy')
for f in files_crise:
    data = np.load(f)
    # --- CORRECTION CRITIQUE : CONVERSION EN µV ---
    fp1 = data[0] * 1000000
    fp2 = data[1] * 1000000
    
    X.append(compute_features(fp1, fp2))
    y.append(1)

X = np.array(X)
y = np.array(y)

# --- ENTRAINEMENT ---
print(f"Dataset : {len(X)} échantillons (Calme: {len(files_calme)}, Crise: {len(files_crise)})")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# --- EVALUATION CLASSIQUE ---
print("\n--- RÉSULTATS TEST SET (20%) ---")
print(classification_report(y_test, clf.predict(X_test), target_names=['Calme', 'Crise']))

# --- VALIDATION CROISÉE (ROBUSTESSE) ---
print("\n" + "="*40)
print(" TEST DE ROBUSTESSE (CROSS-VALIDATION 5-FOLD)")
print("="*40)
# On découpe les données en 5 parties différentes et on teste 5 fois
scores = cross_val_score(clf, X, y, cv=5)

print(f"Scores individuels : {scores}")
print(f"Moyenne Précision  : {scores.mean()*100:.2f}%")
print(f"Stabilité (+/-)    : {scores.std()*100:.2f}%")

if scores.mean() > 0.90:
    print("\n✅ EXCELLENT : Le modèle est stable et performant.")
else:
    print("\n⚠️ ATTENTION : Le modèle est instable ou manque de données.")

# --- SAUVEGARDE ---
joblib.dump(clf, 'model_eeg.pkl')
print("\n💾 Modèle sauvegardé sous 'model_eeg.pkl'.")