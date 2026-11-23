"""
ADS-B live map, decoupled from GNU Radio.
Reads adsb_tracks.csv (written by the GRC Python block) and serves a Leaflet map.

  python adsb_map_server.py                       live map
  python adsb_map_server.py --replay --timeout 900   show a finished recording
  options: --csv PATH  --port 8000  --timeout 60
"""
import argparse
import csv
import json
import os
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ADS-B Live Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
 html,body{margin:0;height:100%;font-family:system-ui,sans-serif}
 #wrap{display:flex;height:100%}
 #map{flex:1}
 #side{width:340px;overflow:auto;background:#111;color:#eee;font-size:13px}
 #side h3{margin:0;padding:10px;background:#222}
 table{width:100%;border-collapse:collapse}
 th,td{padding:5px 6px;text-align:left;border-bottom:1px solid #333}
 tr:hover{background:#222;cursor:pointer}
 .plane{background:none;border:none}
</style></head>
<body><div id="wrap"><div id="map"></div>
<div id="side"><h3>Aircraft: <span id="n">0</span>
<span id="st" style="font-weight:normal;font-size:11px"></span></h3>
<table><thead><tr><th>Callsign</th><th>ICAO</th><th>Alt ft</th><th>kt</th></tr></thead>
<tbody id="rows"></tbody></table></div></div>
<script>
var map = L.map('map').setView([12.95, 77.62], 9);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom: 18, attribution: '&copy; OpenStreetMap'}).addTo(map);
var markers = {}, trails = {}, fitted = false;

function icon(h){
  return L.divIcon({className:'plane', iconSize:[28,28], iconAnchor:[14,14],
    html:'<svg viewBox="0 0 24 24" width="28" height="28" style="transform:rotate('+h+'deg)">'+
    '<path d="M12 2L15 10L22 14V16L15 14L14 20L16 22V23L12 22L8 23V22L10 20L9 14L2 16V14L9 10Z" '+
    'fill="#e53" stroke="#000" stroke-width="1"/></svg>'});
}
function fmt(v){ return (v===null||v===undefined) ? '-' : Math.round(v); }

document.getElementById('rows').onclick = function(e){
  var tr = e.target.closest('tr');
  if(tr && markers[tr.dataset.i]){
    map.setView(markers[tr.dataset.i].getLatLng(), 11);
    markers[tr.dataset.i].openPopup();
  }
};

async function tick(){
  try{
    var data = await (await fetch('/data')).json();
    var s = await (await fetch('/status')).json();
    document.getElementById('st').textContent =
      ' | rows: ' + s.rows + ' | ' + s.note;
    var seen = {}, rows = '', pts = [];
    data.forEach(function(p){
      seen[p.icao] = 1;
      var name = p.callsign || p.icao;
      rows += '<tr data-i="'+p.icao+'"><td>'+name+'</td><td>'+p.icao+
              '</td><td>'+fmt(p.altitude)+'</td><td>'+fmt(p.speed)+'</td></tr>';
      if(p.latitude===null || p.longitude===null) return;
      var ll = [p.latitude, p.longitude]; pts.push(ll);
      var tip = '<b>'+name+'</b><br>ICAO '+p.icao+'<br>Alt '+fmt(p.altitude)+' ft<br>'+
                'Speed '+fmt(p.speed)+' kt<br>Hdg '+fmt(p.heading)+'&deg;<br>'+
                'Climb '+fmt(p.vertical_rate)+' ft/min<br>SNR '+(p.snr?p.snr.toFixed(1):'-')+' dB';
      if(markers[p.icao]){
        markers[p.icao].setLatLng(ll).setIcon(icon(p.heading||0)).setPopupContent(tip);
        trails[p.icao].setLatLngs(p.trail);
      } else {
        markers[p.icao] = L.marker(ll, {icon: icon(p.heading||0)}).bindPopup(tip).addTo(map);
        trails[p.icao] = L.polyline(p.trail, {color:'#e53', weight:2}).addTo(map);
      }
    });
    Object.keys(markers).forEach(function(k){
      if(!seen[k]){ map.removeLayer(markers[k]); map.removeLayer(trails[k]);
                    delete markers[k]; delete trails[k]; }
    });
    document.getElementById('rows').innerHTML = rows;
    document.getElementById('n').textContent = data.length;
    if(!fitted && pts.length){ map.fitBounds(pts, {maxZoom:11, padding:[40,40]}); fitted = true; }
  }catch(e){ document.getElementById('st').textContent = ' ERR: ' + e; }
}
setInterval(tick, 1000); tick();
</script></body></html>
"""


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2 +
         math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(math.sqrt(a))


class Tracker:
    def __init__(self, path, timeout, replay, home=(12.95, 77.62), max_range_km=450.0):
        self.path, self.timeout, self.replay = path, timeout, replay
        self.home, self.max_range_km = home, max_range_km
        self.rejected = 0
        self.planes = {}
        self.lock = threading.Lock()
        self.rows = 0
        self.last_t = 0.0
        self.offset = 0
        self.partial = b""

    @staticmethod
    def _num(x):
        try:
            return float(x) if x != "" else None
        except ValueError:
            return None

    def ingest(self, row):
        if len(row) < 10:
            return
        t = self._num(row[0])
        icao = row[1].strip()
        if t is None or not icao:
            return
        n = self._num
        lat0, lon0 = n(row[3]), n(row[4])
        if lat0 is not None and lon0 is not None and \
                haversine_km(self.home[0], self.home[1], lat0, lon0) > self.max_range_km:
            self.rejected += 1   # physically impossible to receive: decoder glitch
            return
        with self.lock:
            p = self.planes.setdefault(icao, {
                "icao": icao, "callsign": "", "latitude": None, "longitude": None,
                "altitude": None, "speed": None, "heading": None,
                "vertical_rate": None, "snr": None, "trail": [], "seen": 0})
            if row[2].strip():
                p["callsign"] = row[2].strip()
            lat, lon = n(row[3]), n(row[4])
            if lat is not None and lon is not None:
                p["latitude"], p["longitude"] = lat, lon
                p["trail"].append([lat, lon])
                p["trail"] = p["trail"][-60:]
            for key, idx in (("altitude", 5), ("speed", 6), ("heading", 7),
                             ("vertical_rate", 8), ("snr", 9)):
                v = n(row[idx])
                if v is not None:
                    p[key] = v
            p["seen"] = t
            self.last_t = max(self.last_t, t)
            self.rows += 1

    def poll_once(self):
        if not os.path.exists(self.path):
            return
        size = os.path.getsize(self.path)
        if size < self.offset:  # file was recreated
            self.offset, self.partial = 0, b""
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        if not chunk:
            return
        self.offset += len(chunk)
        lines = (self.partial + chunk).split(b"\n")
        self.partial = lines.pop()
        for line in lines:
            line = line.decode("utf-8", "ignore").strip()
            if not line or line.startswith("t,"):
                continue
            for row in csv.reader([line]):
                self.ingest(row)

    def poll_loop(self):
        while True:
            try:
                self.poll_once()
            except Exception as e:
                print("read error:", repr(e))
            time.sleep(0.5)

    def now(self):
        return self.last_t if self.replay else time.time()

    def snapshot(self):
        now = self.now()
        with self.lock:
            return [dict(p) for p in self.planes.values() if now - p["seen"] <= self.timeout]

    def status(self):
        if self.replay:
            note = "replay mode"
        elif self.rows == 0:
            note = "waiting for data in " + os.path.basename(self.path)
        else:
            note = "last row %.0fs ago" % (time.time() - self.last_t)
        if self.rejected:
            note += " | dropped %d bad fixes" % self.rejected
        return {"rows": self.rows, "note": note}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="adsb_tracks.csv")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--home", default="12.95,77.62", help="receiver lat,lon")
    ap.add_argument("--max-range-km", type=float, default=450.0)
    a = ap.parse_args()

    home = tuple(float(x) for x in a.home.split(","))
    tr = Tracker(os.path.abspath(a.csv), a.timeout, a.replay, home, a.max_range_km)
    tr.poll_once()
    threading.Thread(target=tr.poll_loop, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/data"):
                body, ctype = json.dumps(tr.snapshot()).encode(), "application/json"
            elif self.path.startswith("/status"):
                body, ctype = json.dumps(tr.status()).encode(), "application/json"
            else:
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        allow_reuse_address = False
        daemon_threads = True

    try:
        srv = Server(("127.0.0.1", a.port), Handler)
    except OSError as e:
        raise SystemExit("Cannot use port %d (%s). Try --port %d" % (a.port, e, a.port + 1))
    print("Reading %s" % tr.path)
    print("Open http://127.0.0.1:%d   (Ctrl+C to stop)" % a.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
