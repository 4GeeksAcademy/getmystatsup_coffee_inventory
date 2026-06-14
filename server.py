import csv
from pathlib import Path
import shutil
from threading import Lock
from typing import List
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


DATA_FILE = Path("products.csv")
LEGACY_DATA_FILE = Path("inventory.csv")
CSV_HEADERS = ["id", "name", "stock", "min_stock", "unit"]


class Product(BaseModel):
    id: str
    name: str
    stock: int = Field(ge=0)
    min_stock: int = Field(ge=0)
    unit: str = "units"


class ProductCreate(BaseModel):
    name: str
    stock: int = Field(ge=0)
    min_stock: int = Field(ge=0)
    unit: str = "units"


class StockUpdate(BaseModel):
    stock: int = Field(ge=0)


class InventoryStore:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._lock = Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        if self.file_path.exists():
            return
        with self.file_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
            writer.writeheader()

    def _read_all(self) -> List[Product]:
        with self.file_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            products: List[Product] = []
            for row in reader:
                products.append(
                    Product(
                        id=row["id"],
                        name=row["name"],
                        stock=int(row["stock"]),
                        min_stock=int(row["min_stock"]),
                        unit=row["unit"] or "units",
                    )
                )
            return products

    def _write_all(self, products: List[Product]) -> None:
        with self.file_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADERS)
            writer.writeheader()
            for product in products:
                writer.writerow(product.model_dump())

    def list_products(self) -> List[Product]:
        with self._lock:
            return self._read_all()

    def create_product(self, payload: ProductCreate) -> Product:
        with self._lock:
            products = self._read_all()
            if any(item.name.lower() == payload.name.lower() for item in products):
                raise HTTPException(status_code=409, detail="Product already exists")

            product = Product(
                id=str(uuid4()),
                name=payload.name,
                stock=payload.stock,
                min_stock=payload.min_stock,
                unit=payload.unit,
            )
            products.append(product)
            self._write_all(products)
            return product

    def update_stock(self, product_id: str, stock: int) -> Product:
        with self._lock:
            products = self._read_all()
            for index, item in enumerate(products):
                if item.id == product_id:
                    updated = item.model_copy(update={"stock": stock})
                    products[index] = updated
                    self._write_all(products)
                    return updated

            raise HTTPException(status_code=404, detail="Product not found")


app = FastAPI(title="Coffee Inventory API", version="1.0.0")

if not DATA_FILE.exists() and LEGACY_DATA_FILE.exists():
    shutil.copy2(LEGACY_DATA_FILE, DATA_FILE)

store = InventoryStore(DATA_FILE)


@app.get("/products", response_model=List[Product])
def list_products() -> List[Product]:
    return store.list_products()


@app.post("/products", response_model=Product, status_code=201)
def create_product(payload: ProductCreate) -> Product:
    return store.create_product(payload)


@app.patch("/products/{product_id}/stock", response_model=Product)
def update_stock(product_id: str, payload: StockUpdate) -> Product:
    return store.update_stock(product_id=product_id, stock=payload.stock)


@app.get("/alerts/low-stock", response_model=List[Product])
def low_stock_alerts() -> List[Product]:
    return [item for item in store.list_products() if item.stock <= item.min_stock]
