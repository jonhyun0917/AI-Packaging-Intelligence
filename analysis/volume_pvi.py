def calculate_volume_pvi(volume_occupancy_percent: float) -> dict:
    occ = max(0.0, min(float(volume_occupancy_percent), 100.0))
    score = round(occ)
    if score >= 85:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 50:
        grade = "C"
    else:
        grade = "D"
    return {"volume_pvi": score, "grade": grade}
