"""
FloodGuard: Risk Decision Engine & Disaster Protocol Utility
Maps AI/ML prediction outputs to civil protection actions, risk advisories,
and emergency response checklists.
"""

def get_risk_meta(risk_level: int):
    """
    Returns metadata for a given risk level:
    0: Normal (Safe)
    1: Advisory (Watch)
    2: Warning (High)
    3: Emergency (Severe)
    """
    meta_table = {
        0: {
            "title": "Level 0: Normal / Safe",
            "color": "#10B981", # Emerald green
            "bg_color": "#ECFDF5",
            "badge": "SAFE",
            "severity": "Low",
            "desc": "River levels and precipitation within standard catchment capacity.",
            "actions": [
                "Maintain baseline automated telemetry polling (every 6 hours).",
                "Ensure stormwater channels and retention ponds are unobstructed.",
                "Routine operational status at all barrage gates."
            ]
        },
        1: {
            "title": "Level 1: Hydrological Watch / Advisory",
            "color": "#F59E0B", # Amber
            "bg_color": "#FFFBEB",
            "badge": "ADVISORY",
            "severity": "Moderate",
            "desc": "Soil saturation elevated. Sustained rain may cause local low-lying waterlogging.",
            "actions": [
                "Increase sensor monitoring frequency to 1-hour intervals.",
                "Inspect urban pump stations and verify backup power generators.",
                "Issue advisory notices to agricultural and riparian communities.",
                "Pre-stage inflatable flood barriers in low-elevation underpasses."
            ]
        },
        2: {
            "title": "Level 2: Flood Warning / Critical",
            "color": "#F97316", # Orange
            "bg_color": "#FFF7ED",
            "badge": "HIGH RISK",
            "severity": "High",
            "desc": "River discharge approaching bankfull capacity. Rapid inundation risk within 12 hours.",
            "actions": [
                "Initiate controlled reservoir spillway releases to create buffer headroom.",
                "Mobilize municipal rapid response rescue teams to high-risk zones.",
                "Erect primary flood defense barriers along vulnerable riverbanks.",
                "Issue voluntary evacuation guidance for ground-floor residents in floodplain zones."
            ]
        },
        3: {
            "title": "Level 3: Severe Flood Emergency",
            "color": "#EF4444", # Red
            "bg_color": "#FEF2F2",
            "badge": "EMERGENCY",
            "severity": "Severe / Critical",
            "desc": "Forecasted water levels breach major flood defense thresholds. Imminent life/property hazard.",
            "actions": [
                "Sound municipal flood sirens and broadcast emergency cell-broadcast alerts.",
                "Enforce mandatory evacuation of flood zones Zone A and Zone B.",
                "Shut down vulnerable electrical substations in inundated sectors.",
                "Open emergency relief shelters on elevated grounds with medical supplies."
            ]
        }
    }
    return meta_table.get(risk_level, meta_table[0])
