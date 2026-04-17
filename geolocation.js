/**
 * geolocation.js — Alexandria Pedestrian Soundwalk (Peripheral.Mode)
 * Pure geolocation logic: route projection, weighted zone mapping, polygon detection.
 * No UI, no pd4web dependencies — fully testable in isolation.
 *
 * UPDATED for Peripheral.Mode — new route from user GeoJSON (Feature 2 + Feature 7).
 *
 * Track 1:      zones 1-6   →  0% to 10%   (compact museum plaza)
 * Trans 1→2:    zones 7-10  → 10% to 16%   (heading onto Safeya Zaghloul)
 * Track 2:      zones 11-25 → 16% to 45%   (Safeya Zaghloul, ends before seafront corner)
 * Hinge 2→3:    zones 26-27 → 45% to 50%   (seafront corner → mid-seafront)
 * Track 3:      zones 28-31 → 50% to 74%   (mid-seafront → south down El Naby Danial)
 * Track 4:      zones 32-35 → 74% to 100%  (lower El Naby + side streets back)
 */

/* ────────────────────────────────────────────────────────────────────────
   ROUTE COORDINATES
   ROUTE_LINE: inner / right_side from user GeoJSON Feature 2 (53 points)
   [longitude, latitude] — walk order
   ──────────────────────────────────────────────────────────────────────── */
const ROUTE_LINE = [
    [29.9068798, 31.1997637], [29.9062710, 31.1995091], [29.9062334, 31.1994953],
    [29.9067001, 31.1987153], [29.9069442, 31.1982679], [29.9068128, 31.1982013],
    [29.9063140, 31.1979950], [29.9054422, 31.1976577], [29.9050801, 31.1975086],
    [29.9047234, 31.1973847], [29.9045062, 31.1973021], [29.9043264, 31.1974558],
    [29.9040448, 31.1977449], [29.9037927, 31.1979835], [29.9028110, 31.1988852],
    [29.9017891, 31.1998180], [29.9010863, 31.2004627], [29.9007162, 31.2008068],
    [29.9006894, 31.2009399], [29.9004265, 31.2011808], [29.9000912, 31.2014652],
    [29.8999169, 31.2015983], [29.8997774, 31.2017245], [29.8996728, 31.2017750],
    [29.8983827, 31.2012244], [29.8984873, 31.2009490], [29.8987448, 31.2004787],
    [29.8989245, 31.2000635], [29.8990103, 31.1998799], [29.8991444, 31.1997790],
    [29.8994100, 31.1996138], [29.8995012, 31.1995197], [29.8996111, 31.1992995],
    [29.8998552, 31.1987970], [29.9000215, 31.1984541], [29.9002280, 31.1980044],
    [29.9005124, 31.1974193], [29.9006196, 31.1971692], [29.9007644, 31.1968479],
    [29.9009012, 31.1965405], [29.9009924, 31.1963042], [29.9011131, 31.1960495],
    [29.9011802, 31.1959050], [29.9016281, 31.1960862], [29.9018910, 31.1961918],
    [29.9022423, 31.1963111], [29.9026579, 31.1964716], [29.9029583, 31.1965841],
    [29.9032319, 31.1966850], [29.9034170, 31.1967470], [29.9035350, 31.1966781],
    [29.9038086, 31.1965795], [29.9041573, 31.1964762]
];

/* ────────────────────────────────────────────────────────────────────────
   OUTER BOUNDARY
   OUTER_BORDER: outer / left_side from user GeoJSON Feature 7 (38 points)
   ──────────────────────────────────────────────────────────────────────── */
const OUTER_BORDER = [
    [29.8999869, 31.2006768], [29.8999367, 31.2006290], [29.8995573, 31.2002376],
    [29.8993145, 31.1999560], [29.8991862, 31.1998486], [29.8993145, 31.1997627],
    [29.8994429, 31.1996720], [29.8995182, 31.1996243], [29.8996382, 31.1994024],
    [29.8997777, 31.1991494], [29.8998781, 31.1989107], [29.9000706, 31.1985098],
    [29.9002353, 31.1981447], [29.9004333, 31.1977389], [29.9005896, 31.1974239],
    [29.9009104, 31.1967581], [29.9009914, 31.1965385], [29.9012173, 31.1960564],
    [29.9014498, 31.1961414], [29.9020089, 31.1963440], [29.9022807, 31.1964536],
    [29.9025603, 31.1965599], [29.9028282, 31.1966596], [29.9030845, 31.1967791],
    [29.9033951, 31.1969087], [29.9035077, 31.1969386], [29.9036126, 31.1967891],
    [29.9037951, 31.1967260], [29.9040552, 31.1966596], [29.9043426, 31.1965964],
    [29.9045367, 31.1965666], [29.9048746, 31.1965499], [29.9050920, 31.1966230],
    [29.9054570, 31.1967692], [29.9054997, 31.1968057], [29.9052357, 31.1972707],
    [29.9051852, 31.1973770], [29.9051774, 31.1973803]
];

const ZONE_POLYGON = OUTER_BORDER
    .concat(ROUTE_LINE.slice().reverse())
    .concat([OUTER_BORDER[0]]);

/* ────────────────────────────────────────────────────────────────────────
   ARTISTIC ZONE BREAKPOINTS v2
   ──────────────────────────────────────────────────────────────────────── */
const ZONE_BREAKPOINTS = (function() {
    var bp = [];
    for (var i = 0; i < 6; i++) bp.push((i + 1) * 0.10 / 6);
    for (var i = 0; i < 4; i++) bp.push(0.10 + (i + 1) * 0.06 / 4);
    for (var i = 0; i < 15; i++) bp.push(0.16 + (i + 1) * 0.29 / 15);
    for (var i = 0; i < 2; i++) bp.push(0.45 + (i + 1) * 0.05 / 2);
    for (var i = 0; i < 4; i++) bp.push(0.50 + (i + 1) * 0.24 / 4);
    for (var i = 0; i < 4; i++) bp.push(0.74 + (i + 1) * 0.26 / 4);
    return bp;
})();

const ZONE_FAMILIES = [
    0,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5,
    6, 6, 6, 6,
    7, 7, 7, 7
];

const FAMILY_LABELS = [
    'Track 1', 'Transition', 'Track 2', 'Track 2', 'Track 2', 'Transition', 'Track 3', 'Track 4'
];

function distanceToRoute(lon, lat, line) {
    const toRad = d => d * Math.PI / 180;
    const cosLat = Math.cos(toRad((lat + line[0][1]) / 2));
    const mPerDegLat = 111319;
    const mPerDegLon = 111319 * cosLat;
    let minDist = Infinity;
    for (let i = 0; i < line.length - 1; i++) {
        const ax = line[i][0], ay = line[i][1];
        const bx = line[i+1][0], by = line[i+1][1];
        const dx = bx - ax, dy = by - ay;
        const lsq = dx * dx + dy * dy;
        const t = lsq > 0 ? Math.max(0, Math.min(1, ((lon-ax)*dx + (lat-ay)*dy) / lsq)) : 0;
        const px = ax + t*dx, py = ay + t*dy;
        const dlonM = (lon-px) * mPerDegLon;
        const dlatM = (lat-py) * mPerDegLat;
        const d = Math.sqrt(dlonM*dlonM + dlatM*dlatM);
        if (d < minDist) minDist = d;
    }
    return minDist;
}

function nearRoute(lon, lat, thresholdM = 45) {
    return distanceToRoute(lon, lat, ROUTE_LINE) <= thresholdM;
}

function pointInPolygon(lon, lat, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
        const xi = poly[i][0], yi = poly[i][1];
        const xj = poly[j][0], yj = poly[j][1];
        if (((yi > lat) !== (yj > lat)) && (lon < (xj-xi)*(lat-yi)/(yj-yi) + xi))
            inside = !inside;
    }
    return inside;
}

function projectOntoPolyline(lon, lat, line) {
    const segs = [];
    let total = 0;
    for (let i = 0; i < line.length - 1; i++) {
        const dx = line[i+1][0] - line[i][0];
        const dy = line[i+1][1] - line[i][1];
        const len = Math.sqrt(dx*dx + dy*dy);
        segs.push(len);
        total += len;
    }
    let best = Infinity, bestAccum = 0, accum = 0;
    for (let i = 0; i < line.length - 1; i++) {
        const ax = line[i][0], ay = line[i][1];
        const bx = line[i+1][0], by = line[i+1][1];
        const dx = bx - ax, dy = by - ay;
        const lsq = dx * dx + dy * dy;
        const t = lsq > 0 ? Math.max(0, Math.min(1, ((lon-ax)*dx + (lat-ay)*dy) / lsq)) : 0;
        const px = ax + t*dx, py = ay + t*dy;
        const d = (lon-px)*(lon-px) + (lat-py)*(lat-py);
        if (d < best) { best = d; bestAccum = accum + t*segs[i]; }
        accum += segs[i];
    }
    return total > 0 ? bestAccum / total : 0;
}

function progressToZone(p) {
    if (p <= 0) return 1;
    if (p >= 1) return 35;
    let lo = 0, hi = ZONE_BREAKPOINTS.length - 1;
    while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (ZONE_BREAKPOINTS[mid] < p) lo = mid + 1;
        else hi = mid;
    }
    return lo + 1;
}

function zoneToTrackLabel(zone) {
    if (zone < 1 || zone > 35) return '';
    return FAMILY_LABELS[ZONE_FAMILIES[zone]] || '';
}

function zoneToFamily(zone) {
    if (zone < 1 || zone > 35) return -1;
    return ZONE_FAMILIES[zone];
}

const GeoLogic = {
    ROUTE_LINE, OUTER_BORDER, ZONE_POLYGON, ZONE_BREAKPOINTS,
    ZONE_FAMILIES, FAMILY_LABELS,
    distanceToRoute, nearRoute, pointInPolygon, projectOntoPolyline,
    progressToZone, zoneToTrackLabel, zoneToFamily
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = GeoLogic;
} else if (typeof window !== 'undefined') {
    window.GeoLogic = GeoLogic;
}