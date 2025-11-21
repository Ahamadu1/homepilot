from haversine import haversine, Unit

def haversine_km(a: tuple, b: tuple) -> float:
    try:
        return haversine(a, b, unit=Unit.KILOMETERS)
    except Exception:
        return None
