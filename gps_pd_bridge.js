(function () {
    "use strict";
    let _pd = null, _watchId = null, _currentZone = -1, _currentFamily = -1, _onUpdate = null;
    let _pendingZone = -1, _zoneTimer = null;
    var ZONE_DEBOUNCE_MS = 3000;

    function sendToPd(receiver, value) {
        if (!_pd) return;
        try { if (typeof _pd.sendFloat === "function") _pd.sendFloat(receiver, value); }
        catch (e) { console.error("[gps_pd_bridge] sendFloat error:", e); }
    }

    function emitZoneChange(zone, family, label, detail) {
        window.dispatchEvent(new CustomEvent("gps-zone-change", {
            detail: Object.assign({
                zone: zone,
                family: family,
                label: label
            }, detail || {})
        }));
    }

    function commitZone(zone) {
        _currentZone = zone;
        window.currentZone = zone;
        var family = GeoLogic.zoneToFamily(zone);
        var label = GeoLogic.zoneToTrackLabel(zone);
        sendToPd("zone", zone);
        console.log("[gps_pd_bridge] ZONE", zone, "(" + label + ") | debounced");
        if (family !== _currentFamily) {
            _currentFamily = family;
            sendToPd("family", family);
            console.log("[gps_pd_bridge] FAMILY", family, "(" + label + ")");
        }
        emitZoneChange(zone, family, label, { committed: true });
    }

    function onPosition(pos) {
        var lat = pos.coords.latitude, lon = pos.coords.longitude, acc = pos.coords.accuracy;
        if (!GeoLogic.nearRoute(lon, lat)) {
            console.log("[gps_pd_bridge] off-route | lat:", lat.toFixed(6), "lon:", lon.toFixed(6), "±" + Math.round(acc) + "m");
            if (_onUpdate) _onUpdate({ lat: lat, lon: lon, acc: acc, onRoute: false, zone: _currentZone });
            return;
        }
        var progress = GeoLogic.projectOntoPolyline(lon, lat, GeoLogic.ROUTE_LINE);
        var zone = GeoLogic.progressToZone(progress);
        var label = GeoLogic.zoneToTrackLabel(zone);

        if (zone !== _currentZone) {
            if (zone !== _pendingZone) {
                _pendingZone = zone;
                clearTimeout(_zoneTimer);
                _zoneTimer = setTimeout(function () {
                    if (_pendingZone === zone) {
                        commitZone(zone);
                    }
                }, ZONE_DEBOUNCE_MS);
            }
        } else {
            _pendingZone = -1;
            clearTimeout(_zoneTimer);
        }

        if (_onUpdate) _onUpdate({ lat: lat, lon: lon, acc: acc, onRoute: true, progress: progress, zone: zone, family: GeoLogic.zoneToFamily(zone), label: label });
    }

    function onError(err) { console.warn("[gps_pd_bridge] GPS error:", err.code, err.message); }

    function startGpsBridge(pd4web, onUpdateCb) {
        _pd = pd4web;
        _onUpdate = onUpdateCb || null;
        if (!window.GeoLogic) { console.error("[gps_pd_bridge] GeoLogic not found!"); return; }
        if (!navigator.geolocation) { console.warn("[gps_pd_bridge] Geolocation API not available"); return; }
        if (_watchId !== null) { console.warn("[gps_pd_bridge] Already tracking"); return; }
        _watchId = navigator.geolocation.watchPosition(onPosition, onError, { enableHighAccuracy: true, maximumAge: 2000, timeout: 10000 });
        console.log("[gps_pd_bridge] GPS tracking started");
    }

    function stopGpsBridge() {
        if (_watchId !== null) { navigator.geolocation.clearWatch(_watchId); _watchId = null; console.log("[gps_pd_bridge] GPS tracking stopped"); }
    }

    function setZone(z) {
        z = Math.max(1, Math.min(35, Math.floor(Number(z))));
        _pendingZone = -1;
        clearTimeout(_zoneTimer);
        _currentZone = z;
        _currentFamily = GeoLogic.zoneToFamily(z);
        window.currentZone = z;
        sendToPd("zone", z);
        sendToPd("family", _currentFamily);
        console.log("[gps_pd_bridge] manual zone:", z, "family:", _currentFamily, "(" + GeoLogic.zoneToTrackLabel(z) + ")");
        emitZoneChange(z, _currentFamily, GeoLogic.zoneToTrackLabel(z), { committed: true, manual: true });
    }

    // Live tuning from console: setZoneDebounce(2000)
    window.setZoneDebounce = function (ms) {
        ZONE_DEBOUNCE_MS = Math.max(500, Math.floor(Number(ms)));
        console.log("[gps_pd_bridge] debounce set to", ZONE_DEBOUNCE_MS, "ms");
    };

    window.startGpsBridge = startGpsBridge;
    window.stopGpsBridge = stopGpsBridge;
    window.setZone = setZone;
})();
