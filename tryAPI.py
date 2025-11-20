import os
import sys
import requests
from typing import Dict, Any, Optional

API_KEY = "SIKE"

# Toggle attach-on-start by setting DEBUGPY=1 in your env
if os.getenv("DEBUGPY") == "1":
    try:
        import debugpy  # type: ignore
        debugpy.listen(("127.0.0.1", 5678))
        print("Waiting for debugger attach on 5678...")
        debugpy.wait_for_client()
    except Exception as e:
        print(f"debugpy setup failed: {e}", file=sys.stderr)

def _check_api_key():
    if API_KEY in ("REPLACE_ME", "NOPE", "", None):
        raise RuntimeError(
            "Google Maps API key missing. Set GOOGLE_MAPS_API_KEY env var or edit API_KEY."
        )

def get_place_id(address: str, *, session: Optional[requests.Session] = None) -> str:
    """Use Places Text Search to get the place_id from an address."""
    _check_api_key()
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": address, "key": API_KEY}
    s = session or requests
    resp = s.get(url, params=params, timeout=15)
    data = resp.json()

    status = data.get("status")
    if status != "OK":
        # Common cases: ZERO_RESULTS, REQUEST_DENIED, OVER_QUERY_LIMIT, INVALID_REQUEST
        raise RuntimeError(f"Text Search failed: {status} - {data.get('error_message','')}")

    results = data.get("results", [])
    if not results:
        raise RuntimeError("No results returned for that address.")
    return results[0]["place_id"]

def get_opening_hours(place_id: str, *, session: Optional[requests.Session] = None) -> Dict[str, Any]:
    """Fetch opening and closing times from the Place Details API."""
    _check_api_key()
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    # You can request nested fields; Google will bill per field set.
    fields = "name,formatted_address,opening_hours"
    params = {"place_id": place_id, "fields": fields, "key": API_KEY}
    s = session or requests
    resp = s.get(url, params=params, timeout=15)
    data = resp.json()

    status = data.get("status")
    if status != "OK":
        raise RuntimeError(f"Place Details failed: {status} - {data.get('error_message','')}")

    result = data.get("result", {})
    hours = result.get("opening_hours", {}) or {}

    return {
        "name": result.get("name"),
        "address": result.get("formatted_address"),
        "open_now": hours.get("open_now"),
        "weekday_text": hours.get("weekday_text", []),
    }

def main():
    address_input = "Lakeville United Methodist Church"
    with requests.Session() as s:
        place_id = get_place_id(address_input, session=s)
        opening_info = get_opening_hours(place_id, session=s)

    print(f"Place ID: {place_id}\n")
    title = f" {opening_info.get('name','Unknown Place')} "
    print(f"{title:=^60}")
    print(f"Address: {opening_info.get('address','N/A')}")
    open_now = opening_info.get("open_now")
    if open_now is None:
        print("Open now? Unknown")
    else:
        print(f"Open now? {'Yes' if open_now else 'No'}")
    print("\nOpening hours:")
    weekday_text = opening_info.get("weekday_text") or []
    if weekday_text:
        for line in weekday_text:
            print("  " + line)
    else:
        print("  Not provided by the place.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
