import os
import tempfile
from pathlib import Path
from pyarrow import dataset
import pytest
from typing import Generator, cast

from src.dataset_maker.PairedPDFBuilder import PairedPDFBuilder, PairedPDFConfig


@pytest.fixture
def mock_dataset_structure() -> Generator[Path, None, None]:
    """
    Erstellt eine temporäre Verzeichnisstruktur für Tests mit clean/ und dirty/ Unterverzeichnissen
    und einigen Beispieldateien.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Erstelle die Verzeichnisstruktur
        root_path = Path(temp_dir)
        clean_dir = root_path / "clean"
        dirty_dir = root_path / "dirty"
        
        clean_dir.mkdir()
        dirty_dir.mkdir()
        
        # Erstelle einige Beispieldateien
        # Clean PDF-Dateien
        (clean_dir / "document1.pdf").touch()
        (clean_dir / "document2.pdf").touch()
        
        # Dirty PNG-Dateien mit Namenskonvention
        (dirty_dir / "document1.pdf_bleed_through_001.png").touch()
        (dirty_dir / "document1.pdf_ink_shifter_001.png").touch()
        (dirty_dir / "document2.pdf_bleed_through_001.png").touch()
        
        yield root_path


def test_paired_pdf_builder_initialization(mock_dataset_structure: Path) -> None:
    """Test, ob der PairedPDFBuilder korrekt initialisiert werden kann."""
    # Create the builder with configuration parameters
    builder = PairedPDFBuilder(name="default", data_root=mock_dataset_structure)
    
    assert builder is not None
    # The data_dir is set from the data_root parameter
    data_dir = builder.config.data_dir
    assert data_dir is not None
    assert Path(data_dir).exists()
    # Check that the data_root is correctly passed to the config
    cfg = cast(PairedPDFConfig, builder.config)
    assert cfg.data_root == mock_dataset_structure


def test_paired_pdf_dataset_creation(mock_dataset_structure: Path) -> None:
    """Test, ob das Dataset korrekt erstellt werden kann."""
    # Create the builder with configuration parameters
    builder = PairedPDFBuilder(name="default", data_root=mock_dataset_structure)
    
    # Prepare the dataset before accessing it
    builder.download_and_prepare()
    
    # Dataset erstellen
    dataset = builder.as_dataset(split="train")
    
    # Überprüfen, ob das Dataset korrekt erstellt wurde
    assert dataset is not None
    assert len(dataset) == 3  # Wir haben 3 "dirty" Dateien erstellt
    
    # Überprüfen der Features - mit explizitem Typ-Cast für den Typprüfer
    from datasets import Dataset
    assert builder.info.features is not None, "Features should not be None"
    assert isinstance(dataset, Dataset), "dataset should be a Dataset"

    typed_dataset = Dataset.cast(dataset, features=builder.info.features)
    features = typed_dataset.features

    # Überprüfen der Features
    assert "clean_file" in features
    assert "dirty_file" in features
    assert "dirt_type" in features
    
    # Überprüfen der Daten
    first_example = typed_dataset[0]
    assert "clean_file" in first_example
    assert "dirty_file" in first_example
    assert "dirt_type" in first_example
        
    # The dirt_type field uses ClassLabel, so it returns integer indices
    # We need to map these indices to their string representations
    dirt_type_names = builder.info.features["dirt_type"].names
    dirt_type_idx = first_example["dirt_type"]
    assert dirt_type_idx in range(len(dirt_type_names))
    assert dirt_type_names[dirt_type_idx] in ["bleed_through", "ink_shifter"]


def test_paired_pdf_dataset_content(mock_dataset_structure: Path) -> None:
    """Test, ob der Inhalt des Datasets korrekt ist."""
    # Create the builder with configuration parameters
    builder = PairedPDFBuilder(name="default", data_root=mock_dataset_structure)
    
    # Prepare the dataset before accessing it
    builder.download_and_prepare()
    
    # Dataset erstellen
    dataset = builder.as_dataset(split="train")
    
    # Sammle alle dirt_types mit explizitem Indexing
    dirt_type_indices = [dataset[i]["dirt_type"] for i in range(len(dataset))]
    
    # Get the string labels from the indices
    assert builder.info.features is not None, "Features should not be None"
    dirt_type_names = builder.info.features["dirt_type"].names
    dirt_types = [dirt_type_names[idx] for idx in dirt_type_indices]
    
    # Überprüfen, ob beide Arten von Verschmutzungen vorhanden sind
    assert "bleed_through" in dirt_types
    assert "ink_shifter" in dirt_types
    
    # Überprüfen, ob die Dateipfade korrekt sind
    for i in range(len(dataset)):
        example = dataset[i]
        clean_file = Path(str(example["clean_file"]))
        dirty_file = Path(str(example["dirty_file"]))
        
        # Überprüfen, ob die Dateien existieren würden (sie sind nur Touch-Dateien im Test)
        assert clean_file.name.endswith(".pdf")
        assert dirty_file.name.endswith(".png")
        
        # Überprüfen der Beziehung zwischen clean und dirty
        clean_basename = clean_file.stem
        assert clean_basename in dirty_file.name
