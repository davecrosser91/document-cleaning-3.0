from pathlib import Path
import numpy as np
import shutil
import os

# Optional import for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    print("Warning: matplotlib not found. Visualization will be disabled.")
    HAS_MATPLOTLIB = False
from src.dataset_maker.PairedPDFBuilder import PairedPDFBuilder

# Clear dataset cache to avoid NonMatchingSplitsSizesError
cache_dir = Path(os.path.expanduser("~/.cache/huggingface/datasets"))
if cache_dir.exists():
    print(f"Clearing dataset cache at {cache_dir}...")
    for item in cache_dir.glob("paired_pdf_builder*"):
        if item.is_dir():
            shutil.rmtree(item)
            print(f"Removed {item}")
        else:
            item.unlink()
            print(f"Removed {item}")

# Pfad zum Datensatz
root = Path("src/dataset_maker/data")

# Builder mit den richtigen Parametern initialisieren
builder = PairedPDFBuilder(name="default", data_root=root)

# Dataset vorbereiten mit force_download=True, um den Cache zu überschreiben
print("Rebuilding dataset...")
builder.download_and_prepare(download_mode="force_redownload")

# Dataset laden
dataset = builder.as_dataset(split="train")

# Informationen über das Dataset ausgeben
print(f"Dataset geladen: {len(dataset)} Beispiele")

# Beispiel für den Zugriff auf die Daten
if len(dataset) > 0:
    example = dataset[0]
    print("\nErstes Beispiel:")
    print(f"Clean file: {example['clean_file']}")
    print(f"Dirty file: {example['dirty_file']}")
    
    # Dirt type ist ein numerischer Index, der zu einem Label gemappt werden muss
    dirt_type_idx = example['dirt_type']
    
    # Sicherstellen, dass features nicht None ist
    assert builder.info.features is not None, "Features should not be None"
    dirt_type_names = builder.info.features["dirt_type"].names
    dirt_type = dirt_type_names[dirt_type_idx]
    print(f"Dirt type: {dirt_type} (Index: {dirt_type_idx})")
    
    # Alle verfügbaren Verschmutzungstypen anzeigen
    print(f"\nVerfügbare Verschmutzungstypen: {dirt_type_names}")
    
    # Zugriff auf die Bilddaten
    print("\nBilddaten:")
    clean_image = example['clean_image']
    dirty_image = example['dirty_image']
    
    # Überprüfen, ob die Bilddaten NumPy-Arrays sind und ggf. konvertieren
    if not isinstance(clean_image, np.ndarray) and isinstance(clean_image, list):
        clean_image = np.array(clean_image)
        print("Converted clean image from list to numpy array")
    
    if not isinstance(dirty_image, np.ndarray) and isinstance(dirty_image, list):
        dirty_image = np.array(dirty_image)
        print("Converted dirty image from list to numpy array")
    
    if isinstance(clean_image, np.ndarray) and isinstance(dirty_image, np.ndarray):
        print(f"Clean image shape: {clean_image.shape}")
        print(f"Dirty image shape: {dirty_image.shape}")
    else:
        print(f"Clean image type: {type(clean_image)}")
        print(f"Dirty image type: {type(dirty_image)}")
    
    # Bilder anzeigen (optional, nur wenn matplotlib verfügbar ist)
    if HAS_MATPLOTLIB and isinstance(clean_image, np.ndarray) and isinstance(dirty_image, np.ndarray):
        try:
            plt.figure(figsize=(15, 15))
            
            plt.title("Clean Image")
            # Konvertiere von (C, H, W) zu (H, W, C) und normalisiere für die Anzeige
            plt.imshow(np.transpose(clean_image, (1, 2, 0)))
            plt.axis('off')
            plt.show()            
            plt.title(f"Dirty Image ({dirt_type})")
            plt.imshow(np.transpose(dirty_image, (1, 2, 0)))
            plt.axis('off')
            plt.show()
            
            print("\nBilder wurden als 'example_images.png' gespeichert.")
        except Exception as e:
            print(f"Konnte Bilder nicht anzeigen oder speichern: {e}")
    
    # Beispiel für die Verwendung in einem PyTorch-Modell
    print("\nBeispiel für die Verwendung in einem PyTorch-Modell:")
    print("```python")
    print("import torch")
    print("from torch import nn")
    print("from torch.utils.data import DataLoader")
    print("")
    print("# Dataset in DataLoader umwandeln")
    print("dataloader = DataLoader(dataset, batch_size=4, shuffle=True)")
    print("")
    print("# Batch laden")
    print("batch = next(iter(dataloader))")
    print("clean_images = torch.tensor(batch['clean_image']).float()")
    print("dirty_images = torch.tensor(batch['dirty_image']).float()")
    print("")
    print("# In ein Modell einspeisen")
    print("model = nn.Sequential(...)")
    print("outputs = model(dirty_images)")
    print("loss = nn.MSELoss()(outputs, clean_images)")
    print("```")
else:
    print("Keine Beispiele im Dataset gefunden!")