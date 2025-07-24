# paired_pdf_dataset.py
from __future__ import annotations

import io
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, cast, Any, Callable, Union

import datasets
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# PyMuPDF is imported as fitz
# Requires PyMuPDF >= 1.26.3 for page.get_pixmap() method
try:
    import fitz  # type: ignore
    # Verify PyMuPDF version
    if not hasattr(fitz.Page, "get_pixmap"):
        print("Warning: PyMuPDF version is too old. Please upgrade to version >= 1.26.3")
        print("Current version may not support all required features.")
except ImportError:
    print("Warning: PyMuPDF (fitz) not found. PDF loading will not work.")
from pydantic import Field

_CLEAN_DIR = "clean"
_DIRTY_DIR = "dirty"


class PairedPDFConfig(datasets.BuilderConfig):
    """Dataset config pointing at a data root that contains /clean and /dirty."""
    data_root: Path = Field(..., description="Folder containing clean/ and dirty/ sub-dirs")

    def __init__(
        self,
        data_root: Path,
        name: str = "default",
        version: Optional[str] = "0.0.1",
        description: Optional[str] = "Clean–dirty PDF/PNG pairs",
        **kwargs,
    ):
        super().__init__(name=name, version=version, description=description, data_dir=str(data_root), **kwargs)


class PairedPDFBuilder(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")
    BUILDER_CONFIG_CLASS = PairedPDFConfig
    
    # Define a default configuration that will be used if no specific config is provided
    BUILDER_CONFIGS = [PairedPDFConfig(data_root=Path("."))]
    
    @staticmethod
    def _pdf_to_tensor(pdf_path: Path, transform: Callable[[Image.Image], torch.Tensor]) -> np.ndarray:
        """Convert a PDF file to a tensor."""
        result = PairedPDFBuilder._pdf_to_image(pdf_path, transform=transform)
        # Ensure we're returning a numpy array
        if isinstance(result, np.ndarray):
            return result
        raise TypeError("Expected numpy array from _pdf_to_image when transform is provided")
    
    @staticmethod
    def _pdf_to_image(pdf_path: Path, transform: Optional[Callable[[Image.Image], torch.Tensor]] = None, output_path: Optional[Path] = None, page_num: int = 0) -> Union[np.ndarray, Path]:
        """Convert a PDF file to an image or tensor.
        
        Args:
            pdf_path: Path to the PDF file
            transform: Optional transform to apply to the image
            output_path: Optional path to save the image
            page_num: Page number to render (default: 0, first page)
            
        Returns:
            If transform is provided, returns a numpy array of the transformed image
            If output_path is provided, saves the image and returns the path
        """
        # Open the PDF file
        pdf_document = fitz.open(pdf_path)
        
        # Get the specified page
        if page_num >= len(pdf_document):
            page_num = 0  # Default to first page if requested page doesn't exist
        page = pdf_document[page_num]
        
        # Render page to an image with a much higher resolution
        # Using get_pixmap method which is available in PyMuPDF >= 1.26.3
        # Increased scaling factor from 2x to 6x for higher quality PDF rendering
        # This provides approximately 300 DPI for standard PDF documents
        pix = page.get_pixmap(matrix=fitz.Matrix(6, 6))
        
        # Convert to PIL Image
        img_data = pix.samples
        img = Image.frombytes("RGB", (pix.width, pix.height), img_data)
        
        # Save the image if output path is provided
        if output_path is not None:
            # Ensure the parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_path, format="PNG")
            pdf_document.close()
            return output_path
        
        # Apply transformations if provided
        if transform is not None:
            # We know transform returns a torch.Tensor
            tensor = transform(img)
            # Explicitly annotate to help type checker
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("Expected transform to return a torch.Tensor")
            img_tensor = tensor.detach().cpu().numpy()
            pdf_document.close()
            return img_tensor
        
        # If no transform or output path, convert PIL image to numpy array and return
        pdf_document.close()
        return np.array(img)

    def _info(self) -> datasets.DatasetInfo:
        # Using A4 aspect ratio (1:√2 or 1:1.414) at 300 DPI for document images
        # Width: 1754, Height: 2480 (A4 at 300 DPI)
        # This provides high-quality resolution for document processing tasks
        features = datasets.Features(
            {
                "clean_file": datasets.Value("string"),
                "dirty_file": datasets.Value("string"),
                "dirt_type": datasets.ClassLabel(names=["bleed_through_stains_dot_matrix", "ink_shifter_color_paper_mark_up"]),
                "clean_image": datasets.Array3D(shape=(3, 2480, 1754), dtype="float32"),
                "dirty_image": datasets.Array3D(shape=(3, 2480, 1754), dtype="float32"),
            }
        )
        return datasets.DatasetInfo(description="Clean–dirty PDF/PNG pairs", features=features)

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        cfg = cast(PairedPDFConfig, self.config)
        # You could optionally perform an 80/20 random split here
        return [
            datasets.SplitGenerator(name="train", gen_kwargs={"data_root": cfg.data_root}),
        ]

    def _generate_examples(self, **kwargs) -> Iterable[Tuple[int, Dict]]:
        data_root = kwargs.get('data_root')
        # Check if it's path-like (has __truediv__ for / operator)
        if not hasattr(data_root, '__truediv__'):
            raise ValueError(f"Expected data_root to be a path-like object with / operator, got {type(data_root)}")
        # Convert to Path object if it's not already
        data_root = Path(data_root) if not isinstance(data_root, Path) else data_root
        clean_dir, dirty_dir = data_root / _CLEAN_DIR, data_root / _DIRTY_DIR
        clean_basenames = {p.stem for p in clean_dir.glob("*.pdf")}

        def _generate_examples(clean_dir: Path, dirty_dir: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
            """Generate examples from the dataset."""
            # Define image transformation pipeline with A4 aspect ratio at 300 DPI
            # Best practice: Use document-appropriate aspect ratio (A4: 1:√2 or 1:1.414) at high resolution
            # Height: 2480, Width: 1754 pixels (A4 at 300 DPI)
            transform = transforms.Compose([
                transforms.Resize((2480, 1754)),  # High-resolution document format (A4 at 300 DPI)
                transforms.ToTensor(),
            ])
            
            # Create a directory for generated PNGs if it doesn't exist
            generated_png_dir = clean_dir.parent / "generated_pngs"
            generated_png_dir.mkdir(exist_ok=True, parents=True)
            
            # Get all clean PDF files
            clean_files = list(clean_dir.glob("*.pdf"))
            
            for idx, clean_file in enumerate(clean_files):
                # Get the corresponding dirty file
                dirty_files = list(dirty_dir.glob(f"{clean_file.name}*"))
                
                if not dirty_files:
                    print(f"No dirty file found for {clean_file.name}")
                    continue
                    
                # Get the first dirty file (there should only be one)
                dirty_file = dirty_files[0]
                
                # Extract dirt type from filename
                dirty_filename = dirty_file.name
                # Extract dirt type from file name and convert to integer index
                # 0 = bleed_through_stains_dot_matrix, 1 = ink_shifter_color_paper_mark_up
                if "bleed_through_stains_dot_matrix" in str(dirty_file):
                    dirt_type_idx = 0  # Index for "bleed_through_stains_dot_matrix"
                elif "ink_shifter_color_paper_mark_up" in str(dirty_file):
                    dirt_type_idx = 1  # Index for "ink_shifter_color_paper_mark_up"
                else:
                    # Default to the first type if unknown
                    dirt_type_idx = 0  # Default to first class
                
                # Check if the dirty file is a PNG, if not, try to convert it
                dirty_path = dirty_file
                if not str(dirty_path).lower().endswith(".png"):
                    # Try to generate a PNG from the dirty file if it's a PDF
                    if str(dirty_path).lower().endswith(".pdf"):
                        png_output_path = generated_png_dir / f"{dirty_file.stem}.png"
                        try:
                            # _pdf_to_image returns the output_path when saving to file
                            dirty_path = self._pdf_to_image(dirty_file, output_path=png_output_path)
                            print(f"Generated PNG from {dirty_file} at {dirty_path}")
                        except Exception as e:
                            print(f"Failed to generate PNG from {dirty_file}: {e}")
                            continue
                    else:
                        print(f"Skipping non-PNG, non-PDF file: {dirty_file}")
                        continue
                
                try:
                    # Load dirty image (PNG)
                    # Ensure dirty_path is a Path object (not a numpy array)
                    if isinstance(dirty_path, np.ndarray):
                        raise TypeError(f"Cannot open numpy array as image")
                    dirty_image = Image.open(dirty_path).convert("RGB")
                    # Apply transformation to get tensor
                    tensor = transform(dirty_image)
                    # Explicitly annotate to help type checker
                    if not isinstance(tensor, torch.Tensor):
                        raise TypeError("Expected transform to return a torch.Tensor")
                    dirty_tensor = tensor.detach().cpu().numpy()  # Convert torch tensor to numpy
                    
                    # Load clean image (PDF) and convert to image
                    clean_tensor = self._pdf_to_tensor(clean_file, transform)
                    
                    # Generate a clean PNG if it doesn't exist (for reference/visualization)
                    clean_png_path = generated_png_dir / f"{clean_file.stem}.png"
                    if not clean_png_path.exists():
                        try:
                            self._pdf_to_image(clean_file, output_path=clean_png_path)
                            print(f"Generated clean PNG from {clean_file} at {clean_png_path}")
                        except Exception as e:
                            print(f"Failed to generate clean PNG from {clean_file}: {e}")
                    
                    yield idx, {
                        "clean_file": str(clean_file),
                        "dirty_file": str(dirty_path),
                        "dirt_type": dirt_type_idx,  # Using integer index instead of string
                        "clean_image": clean_tensor,
                        "dirty_image": dirty_tensor,
                    }
                except Exception as e:
                    print(f"Error processing {clean_file} and {dirty_file}: {e}")
                    continue
        
        return _generate_examples(clean_dir, dirty_dir)