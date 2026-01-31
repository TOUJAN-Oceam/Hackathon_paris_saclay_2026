import mne
import numpy as np
import os
import shutil

# --- CONFIGURATION ---
SOURCE_DIR = 'dataset_EEG'   # Dossier contenant tes .edf
OUTPUT_DIR = 'dataset_LIVE'
EPOCH_DURATION = 5           # Durée d'une époque (doit matcher ton modèle)

# Combien de temps on veut voir avant et après la crise (en secondes)
BUFFER_BEFORE = 30 
BUFFER_AFTER = 30

# --- VÉRITÉ TERRAIN (Temps précis des crises) ---
# Format: 'Fichier': [Debut_Crise_Secondes, Fin_Crise_Secondes]
SEIZURES = {
    'chb01_21.edf': [327, 420],   # Crise de 93s
    'chb01_04.edf': [1467, 1494], # Crise de 27s
    'chb01_15.edf': [1732, 1772], # Crise de 40s
    'chb01_16.edf': [1015, 1066], # Crise de 51s
    'chb01_26.edf': [1862, 1963]  # Crise de 101s
}

def generate_realistic_scenario():
    # 1. Nettoyage
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    global_index = 0

    print(f"Génération du scénario réaliste dans {OUTPUT_DIR}...")

    # On parcourt chaque fichier pour créer une 'séquence'
    for filename, times in SEIZURES.items():
        filepath = os.path.join(SOURCE_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️ Fichier {filename} introuvable, on passe.")
            continue

        print(f"Traitement de {filename}...")
        
        try:
            # Chargement sans verbose pour aller vite
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
            
            # Sélection Canaux
            picks = [ch for ch in raw.ch_names if 'FP1' in ch.upper() or 'FP2' in ch.upper()]
            if len(picks) < 2: picks = raw.ch_names[:2]
            raw.pick_channels(picks[:2])

            # Définition de la fenêtre temporelle (Avant -> Crise -> Après)
            seizure_start = times[0]
            seizure_end = times[1]
            
            # On coupe large autour de la crise
            crop_min = max(0, seizure_start - BUFFER_BEFORE)
            crop_max = min(raw.times[-1], seizure_end + BUFFER_AFTER)
            
            raw.crop(tmin=crop_min, tmax=crop_max)
            
            # Récupération des données coupées
            data = raw.get_data() # [2, samples]
            sfreq = raw.info['sfreq']
            
            # On doit recalculer les temps relatifs à notre crop
            # Dans le nouveau raw, le temps commence à 0.
            # La crise commence donc à (BUFFER_BEFORE) secondes (si on n'était pas au début du fichier)
            
            # Pour faire simple, on utilise le temps absolu pour labelliser
            current_time_absolute = crop_min
            current_idx = 0
            samples_per_epoch = int(EPOCH_DURATION * sfreq)
            
            while current_idx + samples_per_epoch <= data.shape[1]:
                # Extraction du segment
                segment = data[:, current_idx : current_idx + samples_per_epoch]
                
                # Labellisation : Est-ce que ce segment est DANS la crise ?
                # On regarde le milieu du segment
                segment_mid_time = current_time_absolute + (EPOCH_DURATION / 2)
                
                if seizure_start <= segment_mid_time <= seizure_end:
                    label = "CRISE"
                else:
                    label = "CALME"
                
                # Sauvegarde avec numérotation continue
                save_name = f"live_{global_index:03d}_{label}.npy"
                np.save(os.path.join(OUTPUT_DIR, save_name), segment)
                
                # Avance
                current_idx += samples_per_epoch
                current_time_absolute += EPOCH_DURATION
                global_index += 1

        except Exception as e:
            print(f"Erreur sur {filename}: {e}")

    print(f"✅ Terminé ! {global_index} fichiers générés.")
    print("Le scénario va enchaîner les crises les unes après les autres avec des pauses entre elles.")
    
if __name__ == '__main__':
    generate_realistic_scenario()