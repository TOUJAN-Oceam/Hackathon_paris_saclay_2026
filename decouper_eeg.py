import mne
import numpy as np
import os
import shutil

# --- CONFIGURATION OPTIMISÉE ---
SOURCE_DIR = 'dataset_EEG'   # Là où sont tes .edf (04, 15, 16, 21, 26)
OUTPUT_DIR = 'dataset_TRAIN'
EPOCH_DURATION = 5  # On passe à 5 secondes pour plus de précision

# --- VÉRITÉ TERRAIN (PATIENT 01) ---
# [Début, Fin] en secondes
SEIZURE_TIMES = {
    'chb01_04.edf': [1467, 1494],
    'chb01_15.edf': [1732, 1772],
    'chb01_16.edf': [1015, 1066],
    'chb01_21.edf': [327, 420],
    'chb01_26.edf': [1862, 1963]
}

if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
os.makedirs(os.path.join(OUTPUT_DIR, 'crise'))
os.makedirs(os.path.join(OUTPUT_DIR, 'calme'))

def extract_features_from_file(filename, times):
    filepath = os.path.join(SOURCE_DIR, filename)
    if not os.path.exists(filepath):
        print(f"⚠️ Fichier manquant : {filename}")
        return

    print(f"Traitement {filename}...")
    try:
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
    except:
        print("Erreur lecture.")
        return

    # Sélection Canaux FP1/FP2
    picks = ['FP1-F7', 'FP2-F8']
    if len(picks) < 2: picks = raw.ch_names[:2] # Fallback
    raw.pick_channels(picks[:2])
    
    sfreq = raw.info['sfreq']
    data = raw.get_data() # [2, samples]
    
    # Indices temporels
    s_start, s_end = times
    idx_start_seizure = int(s_start * sfreq)
    idx_end_seizure = int(s_end * sfreq)
    
    samples_per_epoch = int(EPOCH_DURATION * sfreq)
    
    # 1. EXTRACTION CRISES (Label 1)
    current = idx_start_seizure
    count_crise = 0
    while current + samples_per_epoch < idx_end_seizure:
        segment = data[:, current : current + samples_per_epoch]
        np.save(os.path.join(OUTPUT_DIR, 'crise', f"{filename[:-4]}_{count_crise}.npy"), segment)
        current += samples_per_epoch
        count_crise += 1
        
    # 2. EXTRACTION CALME (Label 0)
    # On prend du calme AVANT la crise (ex: 10 min avant) pour éviter les effets post-crise
    # On génère 3x plus de calme que de crise pour être réaliste, mais pas trop pour équilibrer
    idx_start_calm = max(0, idx_start_seizure - (count_crise * 3 * samples_per_epoch) - 1000)
    
    current = idx_start_calm
    count_calme = 0
    # On s'arrête bien avant la crise
    while count_calme < count_crise * 2 and current + samples_per_epoch < idx_start_seizure - 500:
        segment = data[:, current : current + samples_per_epoch]
        np.save(os.path.join(OUTPUT_DIR, 'calme', f"{filename[:-4]}_{count_calme}.npy"), segment)
        current += samples_per_epoch
        count_calme += 1

    print(f"   -> {count_crise} époques CRISE / {count_calme} époques CALME")

for fname, t in SEIZURE_TIMES.items():
    extract_features_from_file(fname, t)