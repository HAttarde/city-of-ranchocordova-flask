"""
Enhanced Data Loader for Energy Data with PDF Support
======================================================

Loads and manages all energy-related datasets:
- Hourly consumption (8760 hours × 100 accounts)
- TOU rates
- Rebates
- Benchmarks
- PDF documents (annual reports, technical documents)
- Text files
"""

import os
import warnings
from typing import Dict, List, Optional

import pandas as pd
from pypdf import PdfReader

warnings.filterwarnings("ignore")


class EnergyDataLoader:
    """Singleton data loader for all energy datasets including PDFs"""

    _instance = None
    _data_loaded = False

    # Data storage
    original_energy_df = None
    customer_service_df = None
    department_df = None
    hourly_consumption_df = None
    tou_rates_df = None
    rebates_df = None
    benchmarks_df = None

    # PDF content storage
    pdf_contents = {}

    # Text file storage
    text_contents = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize data loader (loads data only once)"""
        if not self._data_loaded:
            self.load_all_data()

    def load_all_data(self):
        """Load all datasets from data directory"""
        print("📊 Loading energy datasets...")

        base_path = os.path.join(os.path.dirname(__file__), "data")

        try:
            # Original required data
            self.original_energy_df = self._load_csv(
                base_path, "Energy.txt", required=True
            )
            self.customer_service_df = self._load_csv(
                base_path, "CustomerService.txt", required=True
            )
            self.department_df = self._load_csv(
                base_path, "Department-city of Rancho Cordova.txt", required=False
            )

            # CSV datasets (optional - gracefully handle missing files)
            self.hourly_consumption_df = self._load_csv(
                base_path,
                "AI-Data-2025-R1.csv",
                description="Hourly consumption data (8760 hours)",
            )

            self.tou_rates_df = self._load_csv(
                base_path, "SMUD_TOU_Rates.csv", description="Time-of-Use rate tables"
            )

            self.rebates_df = self._load_csv(
                base_path, "SMUD_Rebates.csv", description="Rebate programs"
            )

            self.benchmarks_df = self._load_csv(
                base_path,
                "CA_Benchmarks.csv",
                description="California utility benchmarks",
            )

            # Load PDF documents
            self._load_pdfs(base_path)

            # Load additional text files
            self._load_text_files(base_path)

            self._data_loaded = True
            print("✅ Energy datasets loaded successfully")
            self._print_summary()

        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
            raise

    def _load_csv(
        self,
        base_path: str,
        filename: str,
        required: bool = False,
        description: str = None,
    ) -> Optional[pd.DataFrame]:
        """Load a single CSV file with error handling"""
        filepath = os.path.join(base_path, filename)

        if not os.path.exists(filepath):
            if required:
                raise FileNotFoundError(f"Required file not found: {filename}")
            else:
                print(f"  ⚠️  {filename} not found (optional)")
                return None

        try:
            df = pd.read_csv(filepath)
            desc = description or filename
            print(f"  ✓ Loaded {filename}: {len(df)} rows")
            return df
        except Exception as e:
            if required:
                raise Exception(f"Error loading {filename}: {e}")
            else:
                print(f"  ⚠️  Could not load {filename}: {e}")
                return None

    def _load_pdfs(self, base_path: str):
        """Load all PDF files from data directory"""
        print("\n📄 Loading PDF documents...")

        pdf_files = [f for f in os.listdir(base_path) if f.endswith(".pdf")]

        for pdf_file in pdf_files:
            filepath = os.path.join(base_path, pdf_file)
            try:
                reader = PdfReader(filepath)
                text_content = []

                for page_num, page in enumerate(reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"[Page {page_num}]\n{page_text}")

                full_text = "\n\n".join(text_content)
                self.pdf_contents[pdf_file] = {
                    "filename": pdf_file,
                    "num_pages": len(reader.pages),
                    "text": full_text,
                    "pages": text_content,
                }

                print(
                    f"  ✓ Loaded PDF {pdf_file}: {len(reader.pages)} pages, {len(full_text)} characters"
                )

            except Exception as e:
                print(f"  ⚠️  Error loading PDF {pdf_file}: {e}")

    def _load_text_files(self, base_path: str):
        """Load additional text files (non-CSV, non-PDF)"""
        print("\n📝 Loading text files...")

        # Files already loaded as DataFrames
        skip_files = [
            "Energy.txt",
            "CustomerService.txt",
            "Department-city of Rancho Cordova.txt",
        ]

        all_files = os.listdir(base_path)
        for filename in all_files:
            if (
                filename in skip_files
                or filename.endswith(".csv")
                or filename.endswith(".pdf")
            ):
                continue

            filepath = os.path.join(base_path, filename)
            if os.path.isfile(filepath):
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    self.text_contents[filename] = content
                    print(f"  ✓ Loaded text file {filename}: {len(content)} characters")
                except Exception as e:
                    print(f"  ⚠️  Error loading text file {filename}: {e}")

    def get_pdf_content(self, filename: str) -> Optional[Dict]:
        """Get content from a specific PDF"""
        return self.pdf_contents.get(filename)

    def get_all_pdf_contents(self) -> Dict:
        """Get all PDF contents"""
        return self.pdf_contents

    def get_text_content(self, filename: str) -> Optional[str]:
        """Get content from a specific text file"""
        return self.text_contents.get(filename)

    def search_pdf_content(self, search_term: str) -> List[Dict]:
        """Search for a term across all PDFs"""
        results = []
        for filename, pdf_data in self.pdf_contents.items():
            if search_term.lower() in pdf_data["text"].lower():
                results.append(
                    {
                        "filename": filename,
                        "num_pages": pdf_data["num_pages"],
                        "match": True,
                    }
                )
        return results

    def _print_summary(self):
        """Print summary of loaded data"""
        print("\n📈 Dataset Summary:")
        print(
            f"  Original Energy: {len(self.original_energy_df) if self.original_energy_df is not None else 0} records"
        )
        print(
            f"  Customer Service: {len(self.customer_service_df) if self.customer_service_df is not None else 0} calls"
        )
        print(
            f"  Departments: {len(self.department_df) if self.department_df is not None else 0} departments"
        )

        if self.hourly_consumption_df is not None:
            print(f"  Hourly Data: {len(self.hourly_consumption_df)} hours")
        if self.tou_rates_df is not None:
            print(f"  TOU Rates: {len(self.tou_rates_df)} rate periods")
        if self.rebates_df is not None:
            print(f"  Rebates: {len(self.rebates_df)} programs")
        if self.benchmarks_df is not None:
            print(f"  Benchmarks: {len(self.benchmarks_df)} utilities")

        if self.pdf_contents:
            print(f"  PDF Documents: {len(self.pdf_contents)} files")
            for filename, data in self.pdf_contents.items():
                print(f"    - {filename}: {data['num_pages']} pages")

        if self.text_contents:
            print(f"  Text Files: {len(self.text_contents)} files")

        print()

    def get_available_datasets(self) -> Dict[str, bool]:
        """Return which datasets are available"""
        return {
            "original_energy": self.original_energy_df is not None,
            "customer_service": self.customer_service_df is not None,
            "department": self.department_df is not None,
            "hourly_consumption": self.hourly_consumption_df is not None,
            "tou_rates": self.tou_rates_df is not None,
            "rebates": self.rebates_df is not None,
            "benchmarks": self.benchmarks_df is not None,
            "pdfs": len(self.pdf_contents) > 0,
            "text_files": len(self.text_contents) > 0,
        }

    def reload(self):
        """Force reload all data"""
        self._data_loaded = False
        self.pdf_contents = {}
        self.text_contents = {}
        self.load_all_data()


# Global instance
_data_loader = None


def get_data_loader() -> EnergyDataLoader:
    """Get singleton instance of data loader"""
    global _data_loader
    if _data_loader is None:
        _data_loader = EnergyDataLoader()
    return _data_loader


# Convenience functions
def get_hourly_data() -> Optional[pd.DataFrame]:
    """Get hourly consumption data"""
    return get_data_loader().hourly_consumption_df


def get_tou_rates() -> Optional[pd.DataFrame]:
    """Get TOU rate data"""
    return get_data_loader().tou_rates_df


def get_rebates() -> Optional[pd.DataFrame]:
    """Get rebate programs data"""
    return get_data_loader().rebates_df


def get_benchmarks() -> Optional[pd.DataFrame]:
    """Get benchmark data"""
    return get_data_loader().benchmarks_df


def get_pdf_content(filename: str) -> Optional[Dict]:
    """Get content from a specific PDF"""
    return get_data_loader().get_pdf_content(filename)


def get_all_pdfs() -> Dict:
    """Get all PDF contents"""
    return get_data_loader().get_all_pdf_contents()


def search_pdfs(search_term: str) -> List[Dict]:
    """Search for a term across all PDFs"""
    return get_data_loader().search_pdf_content(search_term)
