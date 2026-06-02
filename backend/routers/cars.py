"""
REST Router — /api/cars
Endpoint untuk frontend mengambil data mobil secara langsung
(tidak lewat WebSocket).
"""
from fastapi import APIRouter, HTTPException, Query
from backend.db.car_db import get_all_cars, get_car_by_id, search_cars
from backend.models.schemas import CarSpec

router = APIRouter(prefix="/api/cars", tags=["cars"])


@router.get("/", response_model=list[CarSpec])
def list_cars():
    """Ambil semua mobil yang tersedia di showroom."""
    return get_all_cars()


@router.get("/search", response_model=list[CarSpec])
def search(q: str = Query(..., min_length=1, description="Kata kunci pencarian")):
    """Cari mobil berdasarkan brand, model, atau tipe."""
    results = search_cars(q)
    if not results:
        raise HTTPException(status_code=404, detail="Tidak ada mobil yang cocok.")
    return results


@router.get("/{car_id}", response_model=CarSpec)
def get_car(car_id: str):
    """Ambil detail satu mobil berdasarkan ID."""
    car = get_car_by_id(car_id)
    if not car:
        raise HTTPException(status_code=404, detail=f"Mobil '{car_id}' tidak ditemukan.")
    return car
