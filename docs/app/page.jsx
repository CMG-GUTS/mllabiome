'use client';
import React, { useEffect, useRef } from 'react';
import Link from 'next/link';
function rRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}
function makeLCG(seed) {
    let s = seed >>> 0;
    return () => { s = Math.imul(1664525, s) + 1013904223 >>> 0; return s / 4294967296; };
}
export function HeroCanvas() {
    const ref = useRef(null);
    useEffect(() => {
        const cv = ref.current;
        if (!cv)
            return;
        const ctx = cv.getContext('2d');
        const dpr = Math.max(3, window.devicePixelRatio || 1);
        let raf, tick = 0;
        const resize = () => {
            cv.width = cv.offsetWidth * dpr;
            cv.height = cv.offsetHeight * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        };
        resize();
        window.addEventListener('resize', resize);
        const sm = (t) => t * t * (3 - 2 * t);
        const cl = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
        const sl = (p, s, e) => sm(cl((p - s) / (e - s + 1e-9), 0, 1));
        const lrp = (a, b, t) => a + (b - a) * t;
        const CASE_C = '#2563eb';
        const CTRL_C = '#64748b';
        const FOLD_C = ['#2563eb', '#0891b2', '#7c3aed', '#059669', '#b45309'];
        const countRGB = (v) => {
            const m = cl(v, 0, 1);
            return [Math.round(lrp(241, 2, m)), Math.round(lrp(245, 155, m)), Math.round(lrp(249, 190, m))];
        };
        const clrRGB = (v) => {
            const m = cl(Math.abs(v) / 4.0, 0, 1);
            if (v < 0)
                return [Math.round(lrp(255, 217, m)), Math.round(lrp(251, 119, m)), Math.round(lrp(235, 6, m))];
            return [Math.round(lrp(246, 29, m)), Math.round(lrp(248, 78, m)), Math.round(lrp(255, 216, m))];
        };
        const lumFn = ([r, g, b]) => (0.299 * r + 0.587 * g + 0.114 * b) / 255;
        const ALL_DOTS = (() => {
            const rng = makeLCG(7);
            return Array.from({ length: 169 }, (_, i) => ({
                c: i < 91, fold: i % 5,
                px: rng() * 0.88 + 0.06, py: rng() * 0.82 + 0.06,
            }));
        })();
        const FOLD_REPS = [0, 1, 2, 3, 4].flatMap(f => [
            ...ALL_DOTS.filter(d => d.fold === f && d.c).slice(0, 2),
            ...ALL_DOTS.filter(d => d.fold === f && !d.c).slice(0, 2),
        ]);
        const N_ROW = 18, N_COL = 22;
        const COUNT_VALS = (() => {
            const rng = makeLCG(42);
            return Array.from({ length: N_ROW }, () => Array.from({ length: N_COL }, () => { const v = rng(); return v > 0.40 ? (v - 0.40) / 0.60 : 0; }));
        })();
        const SEL_LO = 5, SEL_HI = 18;
        const N_SEL = SEL_HI - SEL_LO + 1;
        const SEL_VALS = COUNT_VALS.map(r => r.slice(SEL_LO, SEL_HI + 1));
        const CLR_VALS = (() => {
            const rng = makeLCG(99);
            return SEL_VALS.map(row => {
                const geo = Math.exp(row.reduce((s, v) => s + Math.log(v + 0.001), 0) / row.length);
                return row.map(v => Math.log((v + 0.001) / geo) + (rng() - 0.5) * 0.4);
            });
        })();
        const GEN_LBLS = ['Lachnospiraceae', 'Ruminococcaceae', 'Bacteroidaceae', 'Prevotellaceae', 'Erysipelotrichaceae', 'Alistipes', 'Faecalibacterium', 'Blautia', 'Roseburia', 'Akkermansia', 'Coprococcus', 'Dialister', 'Megamonas', 'Clostridium'];
        const BIN_VALS = SEL_VALS.map(row => row.map(v => v > 0.28 ? 1 : 0));
        const GENUS_OFFSET = 5;
        const N_GENUS = SEL_HI - (SEL_LO + GENUS_OFFSET) + 1;
        const GENUS_SEL_VALS = SEL_VALS.map(r => r.slice(GENUS_OFFSET));
        const GENUS_CLR_VALS = CLR_VALS.map(r => r.slice(GENUS_OFFSET));
        const GENUS_BIN_VALS = BIN_VALS.map(r => r.slice(GENUS_OFFSET));
        const GENUS_LBLS = GEN_LBLS.slice(GENUS_OFFSET);
        const RADAR_LBLS = ['AUC', 'nMCC', 'F1', 'Prec', 'Rec'];
        const ENS_V = [0.9320, 0.8210, 0.8840, 0.8790, 0.9100];
        const BASE_V = [0.8838, 0.7800, 0.8400, 0.8300, 0.8680];
        const LEARNERS = ['LightGBM', 'Random Forest', 'MLP', 'XGBoost', 'Logistic Reg', 'SVC'];
        const CYCLE = 1800;
        const rr = (x, y, w, h, r) => rRect(ctx, x, y, w, h, r);
        function panel(px, py, pw, ph, title, tag, a) {
            if (a < 0.01)
                return;
            ctx.globalAlpha = a;
            ctx.shadowColor = 'rgba(15,23,42,0.06)';
            ctx.shadowBlur = 8;
            ctx.shadowOffsetY = 2;
            rr(px, py, pw, ph, 7);
            ctx.fillStyle = '#ffffff';
            ctx.fill();
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
            ctx.shadowOffsetY = 0;
            rr(px, py, pw, ph, 7);
            ctx.strokeStyle = '#e2e8f0';
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.fillStyle = '#f8fafc';
            ctx.fillRect(px + 1, py + 1, pw - 2, 22);
            ctx.fillStyle = '#e2e8f0';
            ctx.fillRect(px + 1, py + 22, pw - 2, 1);
            ctx.textAlign = 'left';
            ctx.textBaseline = 'middle';
            ctx.font = '500 9px Inter,sans-serif';
            ctx.fillStyle = '#475569';
            ctx.save();
            ctx.beginPath();
            ctx.rect(px + 1, py + 1, pw - 10, 22);
            ctx.clip();
            ctx.fillText(title, px + 8, py + 11);
            ctx.restore();
            if (tag) {
                const tw = tag.length * 6.5 + 8;
                rr(px + pw - tw - 5, py + 5, tw, 12, 3);
                ctx.fillStyle = 'rgba(37,99,235,0.09)';
                ctx.fill();
                ctx.textAlign = 'right';
                ctx.font = '700 7.5px "JetBrains Mono",monospace';
                ctx.fillStyle = '#2563eb';
                ctx.fillText(tag, px + pw - 8, py + 11);
            }
            ctx.globalAlpha = 1;
        }
        function panelOp(px, py, pw, ph, title, tag, a) {
            if (a < 0.01)
                return;
            ctx.globalAlpha = a * 0.06;
            rr(px, py, pw, ph, 7);
            ctx.fillStyle = '#e2e8f0';
            ctx.fill();
            ctx.globalAlpha = a * 0.38;
            ctx.setLineDash([5, 4]);
            rr(px, py, pw, ph, 7);
            ctx.strokeStyle = '#94a3b8';
            ctx.lineWidth = 1.2;
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;
        }
        function dot(x, y, r, color, a) {
            if (a < 0.01)
                return;
            ctx.globalAlpha = a;
            ctx.beginPath();
            ctx.arc(x, y, r, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.globalAlpha = 1;
        }
        function tunnel(x1, y1, x2, y2, t, color, a, nP = 18) {
            if (a < 0.01 || t <= 0 || t >= 1)
                return;
            const dx = x2 - x1, dy = y2 - y1, len = Math.sqrt(dx * dx + dy * dy);
            if (len < 1)
                return;
            const nx = -dy / len, ny = dx / len;
            for (let i = 0; i < nP; i++) {
                const t0 = ((t * 1.4 - 0.2 + i / nP) % 1 + 1) % 1;
                if (t0 < 0.02 || t0 > 0.98)
                    continue;
                const wave = Math.sin(tick * 0.09 + i * 0.70 + t * 3.0) * cl(len * 0.035, 2, 6);
                const pa = sm(cl(t0 * 3, 0, 1)) * sm(cl((1 - t0) * 3, 0, 1));
                dot(lrp(x1, x2, t0) + nx * wave, lrp(y1, y2, t0) + ny * wave, cl(2.8 - t0 * 0.9, 1.0, 2.8), color, a * pa * 0.80);
            }
        }
        function axisLabels(px, py, pw, ph, xLbl, yLbl, a) {
            if (a < 0.01)
                return;
            const fs = cl(Math.floor(Math.min(pw, ph) * 0.048), 6, 9);
            ctx.globalAlpha = a * 0.40;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.font = `400 ${fs}px Inter,sans-serif`;
            ctx.fillStyle = '#94a3b8';
            if (xLbl)
                ctx.fillText(xLbl, px + pw / 2, py + ph - fs - 1);
            if (yLbl) {
                ctx.save();
                ctx.translate(px + fs, py + ph / 2);
                ctx.rotate(-Math.PI / 2);
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';
                ctx.fillText(yLbl, 0, 0);
                ctx.restore();
            }
            ctx.globalAlpha = 1;
        }
        function otuMatrix(px, py, pw, ph, phase1, a, bracketLabel = 'family–genus', brkLo = SEL_LO, brkHi = SEL_HI) {
            if (a < 0.01)
                return;
            const AX = 14;
            const BX = 10;
            const GW = pw - AX - 6, GH = ph - BX - 6;
            const CW = Math.max(2, Math.floor(GW / N_COL));
            const CH = Math.max(2, Math.floor(GH / N_ROW));
            const x0 = px + AX + Math.floor((GW - N_COL * CW) / 2);
            const y0 = py + 10 + Math.floor((GH - N_ROW * CH) / 2);
            const selT = sm(cl((phase1 - 0.58) / 0.28, 0, 1));
            for (let ri = 0; ri < N_ROW; ri++) {
                const rowT = sm(cl((phase1 * 1.5 - ri * (0.80 / N_ROW)) / 0.12, 0, 1));
                if (rowT < 0.01)
                    continue;
                ctx.globalAlpha = a * rowT * 0.72;
                ctx.beginPath();
                ctx.arc(px + AX - 5, y0 + ri * CH + CH / 2, 1.8, 0, 2 * Math.PI);
                ctx.fillStyle = ri < 9 ? CASE_C : CTRL_C;
                ctx.fill();
                ctx.globalAlpha = 1;
                for (let ci = 0; ci < N_COL; ci++) {
                    const v = COUNT_VALS[ri][ci];
                    const inSel = ci >= brkLo && ci <= brkHi;
                    const [r, g, b] = countRGB(v);
                    ctx.globalAlpha = a * rowT * (inSel ? cl(0.70 + selT * 0.25, 0, 1) : cl(0.35 + v * 0.35, 0.08, 0.80));
                    const fx = x0 + ci * CW, fy = y0 + ri * CH;
                    rr(fx + 0.3, fy + 0.3, CW - 0.6, CH - 0.6, 1);
                    if (inSel && selT > 0.25)
                        ctx.fillStyle = `rgba(37,99,235,${0.12 + v * 0.65})`;
                    else
                        ctx.fillStyle = v < 0.04 ? '#f1f5f9' : `rgb(${r},${g},${b})`;
                    ctx.fill();
                }
            }
            ctx.globalAlpha = 1;
            {
                const fadeA = sm(cl(phase1 / 0.15, 0, 1));
                if (fadeA > 0.01) {
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(px, py, pw, ph);
                    ctx.clip();
                    const eW = cl(N_COL * CW * 0.18, 3, 10);
                    const gL = ctx.createLinearGradient(x0, 0, x0 + eW, 0);
                    gL.addColorStop(0, `rgba(255,255,255,${(fadeA * 0.78).toFixed(2)})`);
                    gL.addColorStop(1, 'rgba(255,255,255,0)');
                    ctx.fillStyle = gL;
                    ctx.fillRect(x0, y0, eW, N_ROW * CH);
                    const gR = ctx.createLinearGradient(x0 + N_COL * CW - eW, 0, x0 + N_COL * CW, 0);
                    gR.addColorStop(0, 'rgba(255,255,255,0)');
                    gR.addColorStop(1, `rgba(255,255,255,${(fadeA * 0.78).toFixed(2)})`);
                    ctx.fillStyle = gR;
                    ctx.fillRect(x0 + N_COL * CW - eW, y0, eW, N_ROW * CH);
                    ctx.restore();
                }
            }
            if (selT > 0.01) {
                const bx1 = x0 + brkLo * CW, bx2 = x0 + (brkHi + 1) * CW, byT = y0 - 4;
                ctx.globalAlpha = a * selT * 0.82;
                ctx.beginPath();
                ctx.moveTo(bx1, byT + 5);
                ctx.lineTo(bx1, byT);
                ctx.lineTo(bx2, byT);
                ctx.lineTo(bx2, byT + 5);
                ctx.strokeStyle = '#2563eb';
                ctx.lineWidth = 1.4;
                ctx.lineJoin = 'round';
                ctx.stroke();
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';
                ctx.fillStyle = '#2563eb';
                ctx.font = `600 ${cl(CW * 2.2, 5, 8)}px "JetBrains Mono",monospace`;
                ctx.fillText(bracketLabel, (bx1 + bx2) / 2, byT - 1);
                ctx.globalAlpha = 1;
            }
            axisLabels(px + AX, py + 4, N_COL * CW, N_ROW * CH + BX, '', '', a * sm(cl(phase1 / 0.15, 0, 1)));
        }
        function taxoTree(px, py, pw, ph, phase1, a, selLo = 4, selHi = 5) {
            if (a < 0.01)
                return;
            const RANKS = ['domain', 'phylum', 'class', 'order', 'family', 'genus', 'species', 'strain'];
            const SEL_LO2 = selLo, SEL_HI2 = selHi;
            const rowH = cl(Math.floor((ph - 2) / RANKS.length), 10, 52);
            const treeTop = py - Math.round(rowH * 0.38);
            const indent = cl(pw * 0.065, 4, 10);
            const bracketT = sm(cl((phase1 - 0.50) / 0.28, 0, 1));
            const labelT = sm(cl((phase1 - 0.80) / 0.20, 0, 1));
            RANKS.forEach((rank, ri) => {
                const ry = treeTop + ri * rowH, rc = ry + rowH / 2;
                const rev = sm(cl((phase1 / 0.50 - ri * 0.10) / 0.15, 0, 1));
                if (rev < 0.01)
                    return;
                const inSel = ri >= SEL_LO2 && ri <= SEL_HI2;
                const isGrey = ri > SEL_HI2;
                const lx = px + 5 + ri * indent;
                if (inSel && bracketT > 0.01) {
                    ctx.globalAlpha = a * bracketT * 0.12;
                    rr(px + 3, ry + 1, pw - 6, rowH - 2, 3);
                    ctx.fillStyle = '#2563eb';
                    ctx.fill();
                    ctx.globalAlpha = 1;
                }
                if (ri > 0) {
                    const pRc = treeTop + (ri - 1) * rowH + rowH / 2, pLx = px + 5 + (ri - 1) * indent;
                    ctx.globalAlpha = a * rev * (isGrey ? 0.35 : inSel ? (0.28 + bracketT * 0.32) : 0.28);
                    ctx.beginPath();
                    ctx.moveTo(pLx + 5, pRc);
                    ctx.lineTo(pLx + 5, rc);
                    ctx.lineTo(lx + 5, rc);
                    ctx.strokeStyle = inSel && bracketT > 0.3 ? '#3b82f6' : '#cbd5e1';
                    ctx.lineWidth = inSel && bracketT > 0.3 ? 1.0 : 0.7;
                    ctx.stroke();
                    ctx.globalAlpha = 1;
                }
                ctx.globalAlpha = a * rev * (isGrey ? 0.45 : inSel ? (0.42 + bracketT * 0.52) : 0.50);
                ctx.beginPath();
                ctx.arc(lx + 5, rc, inSel ? 3.5 : 2.5, 0, 2 * Math.PI);
                ctx.fillStyle = inSel && bracketT > 0.2 ? '#2563eb' : isGrey ? '#e2e8f0' : '#94a3b8';
                ctx.fill();
                ctx.globalAlpha = 1;
                ctx.globalAlpha = a * rev * (isGrey ? 0.48 : inSel ? (0.45 + bracketT * 0.50) : 0.55);
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.font = `${inSel && bracketT > 0.4 ? '700' : '400'} ${cl(rowH * 0.50, 7, 10)}px ${inSel && bracketT > 0.4 ? '"JetBrains Mono",monospace' : 'Inter,sans-serif'}`;
                ctx.fillStyle = inSel && bracketT > 0.4 ? '#1d4ed8' : isGrey ? '#94a3b8' : '#64748b';
                ctx.fillText(rank, lx + 12, rc);
                ctx.globalAlpha = 1;
            });
            if (bracketT > 0.01) {
                const topY = treeTop + SEL_LO2 * rowH, botY = treeTop + SEL_HI2 * rowH + rowH;
                ctx.globalAlpha = a * bracketT * 0.80;
                ctx.beginPath();
                ctx.moveTo(px + 9, topY + 2);
                ctx.lineTo(px + 3, topY + 2);
                ctx.lineTo(px + 3, botY - 2);
                ctx.lineTo(px + 9, botY - 2);
                ctx.strokeStyle = '#2563eb';
                ctx.lineWidth = 1.5;
                ctx.lineJoin = 'round';
                ctx.stroke();
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = '#2563eb';
                ctx.font = `700 ${cl(rowH * 0.46, 7, 9)}px "JetBrains Mono",monospace`;
                ctx.globalAlpha = 1;
            }
        }
        function reducedOTU(px, py, pw, ph, phase1, a, selVals = SEL_VALS, nSel = N_SEL, genLbls = GEN_LBLS) {
            if (a < 0.01)
                return;
            const AX = 12, BX = 10;
            const GW = pw - AX - 6, GH = ph - BX - 6;
            const CH = Math.max(2, Math.floor(GH / N_ROW));
            const CW = Math.max(2, Math.floor(GW / nSel));
            const x0 = px + AX + Math.max(0, Math.floor((GW - nSel * CW) / 2));
            const y0 = py + 10 + Math.floor((GH - N_ROW * CH) / 2);
            const lblT = sm(cl(phase1 * 3, 0, 1));
            ctx.save();
            ctx.beginPath();
            ctx.rect(px, py, pw, ph);
            ctx.clip();
            if (lblT > 0.01) {
                ctx.save();
                ctx.beginPath();
                ctx.rect(px, py, pw, y0 - 1);
                ctx.clip();
                ctx.globalAlpha = a * lblT * 0.72;
                genLbls.forEach((g, ci) => {
                    ctx.save();
                    ctx.translate(x0 + ci * CW + CW / 2, y0 - 4);
                    ctx.rotate(-Math.PI / 3);
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    ctx.font = `400 4px Inter,sans-serif`;
                    ctx.fillStyle = '#475569';
                    ctx.fillText(g, 0, 0);
                    ctx.restore();
                });
                ctx.globalAlpha = 1;
                ctx.restore();
            }
            for (let ri = 0; ri < N_ROW; ri++) {
                const rowT = sm(cl((phase1 * 1.5 - ri * (0.80 / N_ROW)) / 0.12, 0, 1));
                if (rowT < 0.01)
                    continue;
                ctx.globalAlpha = a * rowT * 0.72;
                ctx.beginPath();
                ctx.arc(px + AX - 4, y0 + ri * CH + CH / 2, 1.8, 0, 2 * Math.PI);
                ctx.fillStyle = ri < 9 ? CASE_C : CTRL_C;
                ctx.fill();
                ctx.globalAlpha = 1;
                for (let ci = 0; ci < nSel; ci++) {
                    const v = selVals[ri][ci];
                    const [r, g, b] = countRGB(v);
                    ctx.globalAlpha = a * rowT * (0.65 + v * 0.30);
                    rr(x0 + ci * CW + 0.3, y0 + ri * CH + 0.3, CW - 0.6, CH - 0.6, 1);
                    ctx.fillStyle = v < 0.04 ? '#f1f5f9' : `rgb(${r},${g},${b})`;
                    ctx.fill();
                }
            }
            ctx.globalAlpha = 1;
            ctx.restore();
        }
        function foldPanel(px, py, pw, ph, phase1, a, selVals = SEL_VALS, nSel = N_SEL) {
            if (a < 0.01)
                return;
            const N = 5;
            const ACC = '#2563eb';
            const dT = sm(cl(phase1 * 4, 0, 1));
            if (dT < 0.01)
                return;
            const diagY = py + 6;
            const legH = 16;
            const availH = ph - 6 - legH - 4;
            const rowH = availH / N;
            const numW = 14;
            const gX = px + numW + 8;
            const gW = pw - numW - 18;
            const gap = cl(gW * 0.018, 2, 4);
            const segW = (gW - gap * (N - 1)) / N;
            const segBounds = (si) => {
                const r0 = Math.floor(si * N_ROW / N), r1 = Math.floor((si + 1) * N_ROW / N);
                return { r0, nR: r1 - r0 };
            };
            for (let fi = 0; fi < N; fi++) {
                const rowT = sm(cl((dT - fi * 0.07) * 5, 0, 1));
                if (rowT < 0.01)
                    continue;
                const midY = diagY + fi * rowH + rowH / 2;
                const barH = cl(rowH * 0.82, 12, 32);
                ctx.globalAlpha = a * rowT * 0.28;
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                ctx.font = '400 6.5px "JetBrains Mono",monospace';
                ctx.fillStyle = '#64748b';
                ctx.fillText(`${fi + 1}`, px + numW + 2, midY);
                ctx.globalAlpha = 1;
                for (let si = 0; si < N; si++) {
                    const sx = gX + si * (segW + gap);
                    const isTest = si === fi;
                    const { r0, nR } = segBounds(si);
                    if (nR < 1)
                        continue;
                    const cellW = segW / nSel;
                    const cellH = barH / nR;
                    const hmX = sx;
                    const hmY = midY - barH / 2;
                    for (let ri = 0; ri < nR; ri++) {
                        const row = selVals[r0 + ri];
                        for (let ci = 0; ci < nSel; ci++) {
                            const v = row[ci];
                            const [r, g, b] = countRGB(v);
                            const cellA = isTest
                                ? a * rowT * (0.72 + v * 0.22)
                                : a * rowT * (0.30 + v * 0.12);
                            ctx.globalAlpha = cellA;
                            if (isTest) {
                                ctx.fillStyle = v < 0.04 ? '#f8fafc' : `rgb(${r},${g},${b})`;
                            }
                            else {
                                const gr = Math.round(r * 0.35 + 241 * 0.65);
                                const gg = Math.round(g * 0.35 + 245 * 0.65);
                                const gb = Math.round(b * 0.35 + 249 * 0.65);
                                ctx.fillStyle = v < 0.04 ? '#f8fafc' : `rgb(${gr},${gg},${gb})`;
                            }
                            ctx.fillRect(hmX + ci * cellW, hmY + ri * cellH, cellW - 0.4, cellH - 0.4);
                        }
                    }
                    ctx.globalAlpha = 1;
                    if (isTest) {
                        ctx.globalAlpha = a * rowT * 0.70;
                        rr(sx, hmY, segW, barH, 1.5);
                        ctx.strokeStyle = ACC;
                        ctx.lineWidth = 1.0;
                        ctx.stroke();
                        ctx.globalAlpha = 1;
                    }
                }
            }
            const lT = sm(cl((dT - 0.30) * 4, 0, 1));
            if (lT > 0) {
                const ly = py + ph - legH + 5;
                const sw = 8, sh = 8;
                ctx.globalAlpha = a * lT * 0.55;
                rr(px + 10, ly, sw, sh, 1.5);
                ctx.fillStyle = ACC;
                ctx.fill();
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.font = '400 6.5px Inter,sans-serif';
                ctx.fillStyle = '#94a3b8';
                ctx.fillText('test', px + sw + 13, ly + sh / 2);
                rr(px + 50, ly, sw, sh, 1.5);
                ctx.fillStyle = '#c4d4e8';
                ctx.fill();
                ctx.fillStyle = '#94a3b8';
                ctx.fillText('train', px + sw + 53, ly + sh / 2);
                ctx.globalAlpha = 1;
            }
        }
        function clrMatrix(px, py, pw, ph, phase1, a, clrVals = CLR_VALS, nSel = N_SEL) {
            if (a < 0.01)
                return;
            const N = 5, ACC = '#2563eb';
            const dT = sm(cl(phase1 * 4, 0, 1));
            if (dT < 0.01)
                return;
            const diagY = py + 6;
            const legH = 14;
            const availH = ph - 6 - legH - 4;
            const rowH = availH / N;
            const numW = 14;
            const gX = px + numW + 8;
            const gW = pw - numW - 18;
            const gap = cl(gW * 0.018, 2, 4);
            const segW = (gW - gap * (N - 1)) / N;
            const segBounds = (si) => {
                const r0 = Math.floor(si * N_ROW / N), r1 = Math.floor((si + 1) * N_ROW / N);
                return { r0, nR: r1 - r0 };
            };
            for (let fi = 0; fi < N; fi++) {
                const rowT = sm(cl((dT - fi * 0.07) * 5, 0, 1));
                if (rowT < 0.01)
                    continue;
                const midY = diagY + fi * rowH + rowH / 2;
                const barH = cl(rowH * 0.82, 12, 32);
                ctx.globalAlpha = a * rowT * 0.28;
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                ctx.font = '400 6.5px "JetBrains Mono",monospace';
                ctx.fillStyle = '#64748b';
                ctx.fillText(`${fi + 1}`, px + numW + 2, midY);
                ctx.globalAlpha = 1;
                for (let si = 0; si < N; si++) {
                    const sx = gX + si * (segW + gap);
                    const isTest = si === fi;
                    const { r0, nR } = segBounds(si);
                    if (nR < 1)
                        continue;
                    const cellW = segW / nSel, cellH = barH / nR;
                    const hmX = sx, hmY = midY - barH / 2;
                    for (let ri = 0; ri < nR; ri++) {
                        const row = clrVals[r0 + ri];
                        for (let ci = 0; ci < nSel; ci++) {
                            const v = row[ci];
                            const rgb = clrRGB(v);
                            const m = cl(Math.abs(v) / 4.0, 0, 1);
                            if (isTest) {
                                ctx.globalAlpha = a * rowT * (0.65 + m * 0.28);
                                ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
                            }
                            else {
                                const gr = Math.round(rgb[0] * 0.30 + 241 * 0.70);
                                const gg = Math.round(rgb[1] * 0.30 + 245 * 0.70);
                                const gb = Math.round(rgb[2] * 0.30 + 249 * 0.70);
                                ctx.globalAlpha = a * rowT * (0.25 + m * 0.12);
                                ctx.fillStyle = `rgb(${gr},${gg},${gb})`;
                            }
                            ctx.fillRect(hmX + ci * cellW, hmY + ri * cellH, cellW - 0.4, cellH - 0.4);
                        }
                    }
                    ctx.globalAlpha = 1;
                    if (isTest) {
                        ctx.globalAlpha = a * rowT * 0.65;
                        rr(sx, midY - barH / 2, segW, barH, 1.5);
                        ctx.strokeStyle = ACC;
                        ctx.lineWidth = 1.0;
                        ctx.stroke();
                        ctx.globalAlpha = 1;
                    }
                }
            }
            const lT = sm(cl((dT - 0.3) * 4, 0, 1));
            if (lT > 0) {
                const ly = py + ph - legH + 4, sw = 8, sh = 8;
                ctx.globalAlpha = a * lT * 0.55;
                rr(px + 10, ly, sw, sh, 1.5);
                ctx.fillStyle = ACC;
                ctx.fill();
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.font = '400 6.5px Inter,sans-serif';
                ctx.fillStyle = '#94a3b8';
                ctx.fillText('test', px + sw + 13, ly + sh / 2);
                rr(px + 50, ly, sw, sh, 1.5);
                ctx.fillStyle = '#c4d4e8';
                ctx.fill();
                ctx.fillStyle = '#94a3b8';
                ctx.fillText('train', px + sw + 53, ly + sh / 2);
                ctx.globalAlpha = 1;
            }
        }
        function binMatrix(px, py, pw, ph, phase1, a, binVals = BIN_VALS, nSel = N_SEL) {
            if (a < 0.01)
                return;
            const N = 5, ACC = '#2563eb';
            const dT = sm(cl(phase1 * 4, 0, 1));
            if (dT < 0.01)
                return;
            const diagY = py + 6, legH = 14, availH = ph - 6 - legH - 4;
            const rowH = availH / N, numW = 14, gX = px + numW + 8, gW = pw - numW - 18;
            const gap = cl(gW * 0.018, 2, 4), segW = (gW - gap * (N - 1)) / N;
            const segBounds = (si) => {
                const r0 = Math.floor(si * N_ROW / N), r1 = Math.floor((si + 1) * N_ROW / N);
                return { r0, nR: r1 - r0 };
            };
            for (let fi = 0; fi < N; fi++) {
                const rowT = sm(cl((dT - fi * 0.07) * 5, 0, 1));
                if (rowT < 0.01)
                    continue;
                const midY = diagY + fi * rowH + rowH / 2, barH = cl(rowH * 0.82, 12, 32);
                ctx.globalAlpha = a * rowT * 0.28;
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                ctx.font = '400 6.5px "JetBrains Mono",monospace';
                ctx.fillStyle = '#64748b';
                ctx.fillText(`${fi + 1}`, px + numW + 2, midY);
                ctx.globalAlpha = 1;
                for (let si = 0; si < N; si++) {
                    const sx = gX + si * (segW + gap), isTest = si === fi;
                    const { r0, nR } = segBounds(si);
                    if (nR < 1)
                        continue;
                    const cellW = segW / nSel, cellH = barH / nR;
                    for (let ri = 0; ri < nR; ri++) {
                        const row = binVals[r0 + ri];
                        for (let ci = 0; ci < nSel; ci++) {
                            const v = row[ci];
                            if (isTest) {
                                ctx.globalAlpha = a * rowT * (v ? 0.82 : 0.28);
                                ctx.fillStyle = v ? 'rgba(37,99,235,0.75)' : '#f1f5f9';
                            }
                            else {
                                ctx.globalAlpha = a * rowT * (v ? 0.26 : 0.10);
                                ctx.fillStyle = v ? 'rgba(37,99,235,0.35)' : '#e8f0fb';
                            }
                            ctx.fillRect(sx + ci * cellW, midY - barH / 2 + ri * cellH, cellW - 0.4, cellH - 0.4);
                        }
                    }
                    ctx.globalAlpha = 1;
                    if (isTest) {
                        ctx.globalAlpha = a * rowT * 0.65;
                        rr(sx, midY - barH / 2, segW, barH, 1.5);
                        ctx.strokeStyle = ACC;
                        ctx.lineWidth = 1.0;
                        ctx.stroke();
                        ctx.globalAlpha = 1;
                    }
                }
            }
            const lT = sm(cl((dT - 0.3) * 4, 0, 1));
            if (lT > 0) {
                const ly = py + ph - legH + 4, sw = 8, sh = 8;
                ctx.globalAlpha = a * lT * 0.55;
                rr(px + 10, ly, sw, sh, 1.5);
                ctx.fillStyle = ACC;
                ctx.fill();
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.font = '400 6.5px Inter,sans-serif';
                ctx.fillStyle = '#94a3b8';
                ctx.fillText('test', px + sw + 13, ly + sh / 2);
                rr(px + 90, ly, sw, sh, 1.5);
                ctx.fillStyle = '#dde6f5';
                ctx.fill();
                ctx.fillStyle = '#94a3b8';
                ctx.fillText('train', px + sw + 93, ly + sh / 2);
                ctx.globalAlpha = 1;
            }
        }
        function treeStub(tx, ty, tw, th, act, seed, ts) {
            const nr = cl(Math.min(tw * 0.12, th * 0.09, 5), 2, 5.5);
            const lY = [ty + th * 0.16, ty + th * 0.50, ty + th * 0.82];
            const lX = [[tx + tw * 0.5], [tx + tw * 0.25, tx + tw * 0.75],
                [tx + tw * 0.10, tx + tw * 0.40, tx + tw * 0.60, tx + tw * 0.90]];
            lX[0].forEach(px => lX[1].forEach(qx => {
                ctx.beginPath();
                ctx.moveTo(px, lY[0] + nr);
                ctx.lineTo(qx, lY[1] - nr);
                ctx.strokeStyle = `rgba(147,197,253,${0.14 + act * 0.28})`;
                ctx.lineWidth = 0.9;
                ctx.stroke();
            }));
            lX[1].forEach((px, pi) => lX[2].slice(pi * 2, pi * 2 + 2).forEach(qx => {
                ctx.beginPath();
                ctx.moveTo(px, lY[1] + nr);
                ctx.lineTo(qx, lY[2] - nr);
                ctx.strokeStyle = `rgba(147,197,253,${0.10 + act * 0.20})`;
                ctx.lineWidth = 0.65;
                ctx.stroke();
            }));
            lX.forEach((row, lv) => row.forEach((nx, ni) => {
                const na = 0.30 + act * (0.55 + Math.sin(ts * 0.09 + seed + lv * 1.4 + ni) * 0.15);
                if (lv < 2) {
                    ctx.beginPath();
                    ctx.arc(nx, lY[lv], nr, 0, 2 * Math.PI);
                    ctx.fillStyle = `rgba(147,197,253,${na})`;
                    ctx.fill();
                }
                else {
                    ctx.fillStyle = `rgba(147,197,253,${na * 0.60})`;
                    ctx.fillRect(nx - nr * 0.8, lY[lv] - nr * 0.8, nr * 1.6, nr * 1.6);
                }
            }));
        }
        function modelRF(mx, my, mw, mh, act, ts) {
            const n = 3, g = 6, tw = (mw - g * (n - 1)) / n, treeH = mh * 0.62;
            for (let i = 0; i < n; i++)
                treeStub(mx + i * (tw + g), my, tw, treeH, act, i * 2.1, ts);
            const IMPS = [[0.42, 0.28, 0.18, 0.12], [0.16, 0.44, 0.24, 0.16], [0.28, 0.18, 0.40, 0.14]];
            const impT = sm(cl((act - 0.30) * 3, 0, 1));
            if (impT > 0.01) {
                const bY = my + treeH + 3, bH = cl(mh * 0.050, 2, 5.5), barW = (tw - 5) / 4;
                IMPS.forEach((imps, ti) => imps.forEach((v, fi) => {
                    const bx = mx + ti * (tw + g) + fi * (barW + 1) + 2;
                    ctx.globalAlpha = act * impT * 0.16;
                    rr(bx, bY, barW, bH, 1);
                    ctx.fillStyle = 'rgba(147,197,253,0.12)';
                    ctx.fill();
                    ctx.globalAlpha = act * impT * 0.72;
                    rr(bx, bY, barW * v, bH, 1);
                    ctx.fillStyle = `rgba(147,197,253,${0.48 + v * 0.42})`;
                    ctx.fill();
                    ctx.globalAlpha = 1;
                }));
            }
            const voteT = sm(cl((act - 0.50) * 4, 0, 1));
            if (voteT > 0.01) {
                const bY = my + mh * 0.87, bH = cl(mh * 0.056, 2.5, 6);
                ctx.globalAlpha = act * voteT * 0.12;
                rr(mx, bY, mw, bH, 2);
                ctx.fillStyle = 'rgba(255,255,255,0.05)';
                ctx.fill();
                ctx.globalAlpha = act * voteT * 0.85;
                rr(mx, bY, mw * 0.79 * voteT, bH, 2);
                ctx.fillStyle = 'rgba(37,99,235,0.85)';
                ctx.fill();
                ctx.globalAlpha = 1;
            }
            ctx.globalAlpha = act * 0.28;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'bottom';
            ctx.font = `500 ${cl(mw * 0.055, 5.5, 8)}px "JetBrains Mono",monospace`;
            ctx.fillStyle = 'rgba(147,197,253,0.75)';
            ctx.fillText('bagging  ·  majority vote', mx + mw, my + mh);
            ctx.globalAlpha = 1;
        }
        function modelGBM(mx, my, mw, mh, act, ts) {
            const n = 3, g = 5, tw = (mw - g * (n - 1)) / n, tH = mh * 0.48;
            for (let i = 0; i < n; i++) {
                treeStub(mx + i * (tw + g), my + 2, tw, tH - 2, act * (0.55 + i * 0.15), i * 1.2, ts);
                if (i < n - 1) {
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.font = `700 ${cl(mw * 0.088, 8, 13)}px Inter,sans-serif`;
                    ctx.fillStyle = `rgba(147,197,253,${0.40 + act * 0.45})`;
                    ctx.fillText('+', mx + i * (tw + g) + tw + g / 2, my + tH * 0.38);
                }
                ctx.textAlign = 'center';
                ctx.font = `400 ${cl(mw * 0.052, 5, 7.5)}px "JetBrains Mono",monospace`;
                ctx.fillStyle = `rgba(147,197,253,${0.28 + act * 0.32})`;
                ctx.fillText(`rₜ${i + 1}`, mx + i * (tw + g) + tw / 2, my + tH + 8);
            }
            const curveT = sm(cl((act - 0.22) * 3, 0, 1));
            if (curveT > 0.01) {
                const cY = my + tH + 16, cH = cl(mh * 0.26, 14, 30), cW = mw * 0.90, cX = mx + mw * 0.05;
                ctx.globalAlpha = act * curveT * 0.18;
                ctx.beginPath();
                ctx.moveTo(cX, cY);
                ctx.lineTo(cX, cY + cH);
                ctx.lineTo(cX + cW, cY + cH);
                ctx.strokeStyle = 'rgba(147,197,253,0.40)';
                ctx.lineWidth = 0.6;
                ctx.stroke();
                ctx.globalAlpha = 1;
                ctx.globalAlpha = act * curveT * 0.80;
                ctx.beginPath();
                for (let i = 0; i <= 36; i++) {
                    const t = i / 36;
                    const decay = Math.exp(-t * 3.4) * 0.86 + Math.sin(ts * 0.04 + t * 6.5) * 0.025 * (1 - t);
                    const sx = cX + t * cW * curveT, sy = cY + cH - cl(decay, 0, 1) * cH;
                    i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                }
                ctx.strokeStyle = 'rgba(99,140,255,0.90)';
                ctx.lineWidth = 1.4;
                ctx.stroke();
                ctx.globalAlpha = act * curveT * 0.22;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.font = `400 ${cl(mw * 0.046, 4.5, 6.5)}px "JetBrains Mono",monospace`;
                ctx.fillStyle = 'rgba(147,197,253,0.65)';
                ctx.fillText('residual ↓', cX + 2, cY);
                ctx.globalAlpha = 1;
            }
        }
        function modelMLP(mx, my, mw, mh, act, ts) {
            const L = [4, 6, 4, 2];
            const lx = L.map((_, i) => mx + (i + 0.5) / L.length * mw);
            const ly = L.map(n => Array.from({ length: n }, (_, j) => my + (j + 0.5) / n * mh * 0.86 + mh * 0.07));
            const nr = cl(Math.min(mw * 0.044, mh * 0.065, 4.5), 2, 5.5);
            const WSIGNS = [[1, -1, 1, 1, -1, 1], [-1, 1, 1, -1, 1, 1], [1, -1, 1, 1], [1, 1]];
            for (let l = 0; l < L.length - 1; l++)
                ly[l].forEach((fy, fi) => ly[l + 1].forEach((ty2, ti) => {
                    const wa = sm(cl((Math.sin(ts * 0.08 + l * 1.9 + fi * 0.6 + ti * 0.4) * 0.5 + 0.5) * act, 0, 1));
                    const wsign = WSIGNS[l][fi % WSIGNS[l].length];
                    ctx.beginPath();
                    ctx.moveTo(lx[l], fy);
                    ctx.lineTo(lx[l + 1], ty2);
                    ctx.strokeStyle = wsign > 0 ? `rgba(99,140,255,${wa * 0.30})` : `rgba(251,146,60,${wa * 0.22})`;
                    ctx.lineWidth = 0.65;
                    ctx.stroke();
                }));
            const pulse = (ts % 110) / 110;
            L.forEach((_, l) => ly[l].forEach((ny, ni) => {
                const na = sm(cl((Math.sin(ts * 0.10 + l * 2.2 + ni * 1.3) * 0.5 + 0.5) * act, 0, 1));
                const pDist = Math.abs(l / (L.length - 1) - pulse), pGlow = Math.exp(-pDist * 14) * act * 0.55;
                const rg = ctx.createRadialGradient(lx[l], ny, 0, lx[l], ny, nr * 2.5);
                rg.addColorStop(0, `rgba(99,140,255,${0.15 + na * 0.42 + pGlow})`);
                rg.addColorStop(1, 'rgba(99,140,255,0)');
                ctx.beginPath();
                ctx.arc(lx[l], ny, nr * 2.5, 0, 2 * Math.PI);
                ctx.fillStyle = rg;
                ctx.fill();
                ctx.beginPath();
                ctx.arc(lx[l], ny, nr, 0, 2 * Math.PI);
                ctx.fillStyle = `rgba(165,200,255,${0.32 + na * 0.62 + pGlow * 0.8})`;
                ctx.fill();
            }));
            ctx.globalAlpha = act * 0.30;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.font = `500 ${cl(mw * 0.053, 5, 7.5)}px "JetBrains Mono",monospace`;
            ctx.fillStyle = 'rgba(147,197,253,0.72)';
            ctx.fillText('4 → 6 → 4 → 2', mx + mw * 0.5, my + mh);
            ctx.globalAlpha = 1;
        }
        function radar(cx, cy, r, ens, base, act, a) {
            if (a < 0.01)
                return;
            const N = ens.length, ang = (i) => (i / N) * 2 * Math.PI - Math.PI / 2;
            const pt = (v, i) => [cx + r * v * Math.cos(ang(i)), cy + r * v * Math.sin(ang(i))];
            ctx.globalAlpha = a * 0.28;
            for (let ring = 0.25; ring <= 1.01; ring += 0.25) {
                ctx.beginPath();
                for (let i = 0; i < N; i++) {
                    const [x, y] = pt(ring, i);
                    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
                }
                ctx.closePath();
                ctx.strokeStyle = '#e2e8f0';
                ctx.lineWidth = 0.6;
                ctx.stroke();
            }
            for (let i = 0; i < N; i++) {
                const [x, y] = pt(1, i);
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(x, y);
                ctx.strokeStyle = '#e2e8f0';
                ctx.lineWidth = 0.6;
                ctx.stroke();
            }
            ctx.globalAlpha = 1;
            ctx.beginPath();
            base.forEach((v, i) => { const [x, y] = pt(v, i); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
            ctx.closePath();
            ctx.globalAlpha = a * 0.50;
            ctx.fillStyle = 'rgba(148,163,184,0.18)';
            ctx.fill();
            ctx.strokeStyle = 'rgba(148,163,184,0.45)';
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.globalAlpha = 1;
            const t = cl(act, 0, 1);
            ctx.beginPath();
            ens.forEach((v, i) => { const [x, y] = pt(v * t, i); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
            ctx.closePath();
            ctx.globalAlpha = a;
            ctx.fillStyle = `rgba(37,99,235,${t * 0.16})`;
            ctx.fill();
            ctx.strokeStyle = `rgba(37,99,235,${t * 0.80})`;
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.globalAlpha = 1;
            ctx.globalAlpha = a * 0.72;
            RADAR_LBLS.forEach((l, i) => {
                const [x, y] = pt((r + 13) / r, i);
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.font = `600 ${cl(Math.floor(r * 0.18), 6, 9)}px "JetBrains Mono",monospace`;
                ctx.fillStyle = '#475569';
                ctx.fillText(l, x, y);
            });
            ctx.globalAlpha = 1;
        }
        function modelLR(mx, my, mw, mh, act, ts) {
            const coefW = mw * 0.52, sigW = mw * 0.44, sigX = mx + coefW + 4;
            const COEFS = [0.72, 0.55, -0.48, 0.38, -0.31, 0.25, -0.18, 0.12];
            const FLBLS = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8'];
            const bY0 = my + mh * 0.10, barAreaH = mh * 0.80, bRowH = barAreaH / COEFS.length;
            const zeroX = mx + coefW * 0.50, maxC = 0.72;
            ctx.globalAlpha = act * 0.22;
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(zeroX, bY0 - 2);
            ctx.lineTo(zeroX, bY0 + barAreaH + 2);
            ctx.strokeStyle = 'rgba(147,197,253,0.7)';
            ctx.lineWidth = 0.6;
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;
            COEFS.forEach((c, ci) => {
                const barT = sm(cl((act - ci * 0.08) * 4, 0, 1));
                if (barT < 0.01)
                    return;
                const by = bY0 + ci * bRowH + bRowH * 0.5, bH = cl(bRowH * 0.44, 1.5, 6);
                ctx.globalAlpha = act * barT * 0.44;
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                ctx.font = `400 ${cl(bRowH * 0.50, 4, 6.5)}px "JetBrains Mono",monospace`;
                ctx.fillStyle = 'rgba(147,197,253,0.65)';
                ctx.fillText(FLBLS[ci], zeroX - 3, by);
                ctx.globalAlpha = 1;
                const bw = coefW * 0.48 * Math.abs(c) / maxC * barT;
                const osc = Math.sin(ts * 0.05 + ci * 1.1) * 0.5 + 0.5;
                ctx.globalAlpha = act * barT * (0.62 + osc * 0.26);
                ctx.fillStyle = c >= 0 ? 'rgba(99,140,255,0.88)' : 'rgba(251,146,60,0.78)';
                ctx.fillRect(c >= 0 ? zeroX : zeroX - bw, by - bH / 2, bw, bH);
                const wX = c >= 0 ? zeroX + bw : zeroX - bw;
                ctx.globalAlpha = act * barT * 0.35;
                ctx.strokeStyle = c >= 0 ? 'rgba(147,197,253,0.70)' : 'rgba(253,186,116,0.65)';
                ctx.lineWidth = 0.7;
                ctx.beginPath();
                ctx.moveTo(wX, by - bH * 0.7);
                ctx.lineTo(wX, by + bH * 0.7);
                ctx.stroke();
                ctx.globalAlpha = 1;
            });
            ctx.globalAlpha = act * 0.32;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.font = `400 ${cl(mw * 0.054, 5.5, 7.5)}px "JetBrains Mono",monospace`;
            ctx.fillStyle = 'rgba(147,197,253,0.80)';
            ctx.fillText('β', mx, my);
            ctx.globalAlpha = 1;
            const sigH = mh * 0.72, sigY0 = my + mh * 0.14, sigCY = sigY0 + sigH * 0.5;
            ctx.globalAlpha = act * 0.88;
            ctx.beginPath();
            for (let i = 0; i <= 44; i++) {
                const xv = (i / 44 - 0.5) * 6, yv = 1 / (1 + Math.exp(-xv));
                const sx = sigX + (i / 44) * sigW, sy = sigY0 + sigH - yv * sigH;
                i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
            }
            ctx.strokeStyle = `rgba(99,140,255,${0.62 + act * 0.28})`;
            ctx.lineWidth = 1.6;
            ctx.stroke();
            ctx.globalAlpha = 1;
            ctx.globalAlpha = act * 0.12;
            ctx.beginPath();
            ctx.moveTo(sigX + sigW * 0.5, sigCY);
            for (let i = 22; i <= 44; i++) {
                const xv = (i / 44 - 0.5) * 6, yv = 1 / (1 + Math.exp(-xv));
                ctx.lineTo(sigX + (i / 44) * sigW, sigY0 + sigH - yv * sigH);
            }
            ctx.lineTo(sigX + sigW, sigY0 + sigH);
            ctx.lineTo(sigX + sigW * 0.5, sigY0 + sigH);
            ctx.closePath();
            ctx.fillStyle = 'rgba(99,140,255,0.45)';
            ctx.fill();
            ctx.globalAlpha = 1;
            const thrT = sm(cl((act - 0.5) * 4, 0, 1));
            ctx.globalAlpha = act * thrT * 0.30;
            ctx.setLineDash([2, 3]);
            ctx.beginPath();
            ctx.moveTo(sigX, sigCY);
            ctx.lineTo(sigX + sigW, sigCY);
            ctx.strokeStyle = 'rgba(147,197,253,0.80)';
            ctx.lineWidth = 0.7;
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;
            if (thrT > 0.5) {
                const pulse = Math.sin(ts * 0.05) * 0.5 + 0.5;
                const xv = (pulse - 0.5) * 6, yv = 1 / (1 + Math.exp(-xv));
                const dx = sigX + pulse * sigW, dy = sigY0 + sigH - yv * sigH;
                const rg = ctx.createRadialGradient(dx, dy, 0, dx, dy, 5);
                rg.addColorStop(0, `rgba(99,140,255,${act * 0.75})`);
                rg.addColorStop(1, 'rgba(99,140,255,0)');
                ctx.beginPath();
                ctx.arc(dx, dy, 5, 0, 2 * Math.PI);
                ctx.fillStyle = rg;
                ctx.fill();
                ctx.beginPath();
                ctx.arc(dx, dy, 2, 0, 2 * Math.PI);
                ctx.fillStyle = `rgba(165,200,255,${act * 0.95})`;
                ctx.fill();
            }
            ctx.globalAlpha = act * 0.32;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.font = `400 ${cl(mw * 0.054, 5.5, 7.5)}px "JetBrains Mono",monospace`;
            ctx.fillStyle = 'rgba(147,197,253,0.80)';
            ctx.fillText('σ(z)', sigX, my);
            ctx.globalAlpha = 1;
        }
        function modelBNB(mx, my, mw, mh, act, ts) {
            const NF = 6, colW = mw * 0.27, rowH = mh * 0.62 / NF;
            const tblX = mx + mw * 0.16, tblY = my + mh * 0.14;
            const LIKS = [[0.18, 0.72], [0.55, 0.28], [0.22, 0.68], [0.62, 0.35], [0.15, 0.60], [0.70, 0.30]];
            const hT = sm(cl(act * 5, 0, 1));
            if (hT > 0.01) {
                ctx.globalAlpha = act * hT * 0.48;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';
                ctx.font = `500 ${cl(mw * 0.054, 5, 7.5)}px "JetBrains Mono",monospace`;
                ctx.fillStyle = 'rgba(147,197,253,0.78)';
                ctx.fillText('ctrl', tblX + colW * 0.5, tblY - 2);
                ctx.fillStyle = 'rgba(99,140,255,0.88)';
                ctx.fillText('case', tblX + colW * 1.5 + 3, tblY - 2);
                ctx.globalAlpha = 1;
            }
            LIKS.forEach((lik, fi) => {
                const rT = sm(cl((act - fi * 0.07) * 5, 0, 1));
                if (rT < 0.01)
                    return;
                const fy = tblY + fi * rowH;
                lik.forEach((v, ci) => {
                    const osc = Math.sin(ts * 0.06 + fi * 1.1 + ci * 2.3) * 0.5 + 0.5;
                    ctx.globalAlpha = act * rT * (0.55 + osc * 0.24);
                    rr(tblX + ci * (colW + 2) + 0.5, fy + 0.5, colW - 1, rowH - 1, 1.5);
                    ctx.fillStyle = ci === 0 ? `rgba(147,197,253,${0.18 + v * 0.60})` : `rgba(99,140,255,${0.20 + v * 0.64})`;
                    ctx.fill();
                    ctx.globalAlpha = act * rT * 0.80;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.font = `600 ${cl(rowH * 0.42, 4.5, 7)}px "JetBrains Mono",monospace`;
                    ctx.fillStyle = 'rgba(224,242,254,0.92)';
                    ctx.fillText(v.toFixed(2), tblX + ci * (colW + 2) + colW * 0.5, fy + rowH * 0.5);
                    ctx.globalAlpha = 1;
                });
            });
            const postT = sm(cl((act - 0.60) * 4, 0, 1));
            if (postT > 0.01) {
                const bX = mx + 4, bW = mw - 8, bH = cl(mh * 0.054, 2.5, 6.5);
                const bY0b = my + mh * 0.80, bY1 = my + mh * 0.80 + bH + 4;
                const lLP = [0, 1].map(ci => LIKS.reduce((s, lik) => s + Math.log(cl(lik[ci], 1e-4, 1 - 1e-4)), 0));
                const lMx = Math.max(lLP[0], lLP[1]), lRw = lLP.map(lp => Math.exp(lp - lMx)), lSm = lRw[0] + lRw[1];
                const POSTS = [[lRw[0] / lSm, 'rgba(147,197,253,0.62)', 'ctrl'], [lRw[1] / lSm, 'rgba(99,140,255,0.90)', 'case']];
                POSTS.forEach(([p, c, lbl], vi) => {
                    const y = vi === 0 ? bY0b : bY1;
                    ctx.globalAlpha = act * postT * 0.12;
                    rr(bX, y, bW, bH, 2);
                    ctx.fillStyle = 'rgba(255,255,255,0.05)';
                    ctx.fill();
                    const fw = bW * p * postT;
                    if (fw > 1) {
                        ctx.globalAlpha = act * postT * 0.88;
                        rr(bX, y, fw, bH, 2);
                        ctx.fillStyle = c;
                        ctx.fill();
                    }
                    ctx.globalAlpha = act * postT * 0.48;
                    ctx.textAlign = 'right';
                    ctx.textBaseline = 'middle';
                    ctx.font = `400 ${cl(mw * 0.047, 4.5, 6.5)}px "JetBrains Mono",monospace`;
                    ctx.fillStyle = 'rgba(147,197,253,0.78)';
                    ctx.fillText(`P(c|x) ${lbl}`, bX - 2, y + bH * 0.5);
                    ctx.globalAlpha = 1;
                });
            }
            ctx.globalAlpha = act * 0.24;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'bottom';
            ctx.font = `400 ${cl(mw * 0.045, 4, 6.5)}px "JetBrains Mono",monospace`;
            ctx.fillStyle = 'rgba(147,197,253,0.65)';
            ctx.fillText('P(c|x) ∝ P(c) ∏P(xᵢ|c)', mx + mw, my + mh);
            ctx.globalAlpha = 1;
        }
        function modelNC(mx, my, mw, mh, act, ts) {
            const rng = makeLCG(113);
            const PTS = Array.from({ length: 20 }, (_, i) => {
                const c = i < 10;
                return { x: rng() * 0.40 + (c ? 0.06 : 0.52), y: rng() * 0.70 + 0.14, c };
            });
            const ptR = cl(Math.min(mw, mh) * 0.024, 1.5, 4);
            const C0 = { x: PTS.filter(p => !p.c).reduce((s, p) => s + p.x, 0) / 10, y: PTS.filter(p => !p.c).reduce((s, p) => s + p.y, 0) / 10 };
            const C1 = { x: PTS.filter(p => p.c).reduce((s, p) => s + p.x, 0) / 10, y: PTS.filter(p => p.c).reduce((s, p) => s + p.y, 0) / 10 };
            const toS = (p) => ({ sx: mx + p.x * mw, sy: my + p.y * mh });
            PTS.forEach((p, pi) => {
                const pT = sm(cl((act - pi * 0.038) * 5, 0, 1));
                if (pT < 0.01)
                    return;
                const { sx, sy } = toS(p);
                ctx.globalAlpha = act * pT * 0.68;
                ctx.beginPath();
                ctx.arc(sx, sy, ptR, 0, 2 * Math.PI);
                ctx.fillStyle = p.c ? 'rgba(99,140,255,0.82)' : 'rgba(147,197,253,0.55)';
                ctx.fill();
                ctx.globalAlpha = 1;
            });
            const cT = sm(cl((act - 0.45) * 4, 0, 1));
            if (cT > 0.01) {
                ;
                [{ c: C0, col0: 'rgba(147,197,253,0.55)', col1: 'rgba(147,197,253,0.92)' }, { c: C1, col0: 'rgba(99,140,255,0.68)', col1: 'rgba(99,140,255,0.96)' }].forEach(({ c, col0, col1 }) => {
                    const { sx, sy } = toS(c);
                    const rg = ctx.createRadialGradient(sx, sy, 0, sx, sy, ptR * 4);
                    rg.addColorStop(0, col0);
                    rg.addColorStop(1, 'rgba(99,140,255,0)');
                    ctx.beginPath();
                    ctx.arc(sx, sy, ptR * 4, 0, 2 * Math.PI);
                    ctx.fillStyle = rg;
                    ctx.fill();
                    ctx.beginPath();
                    ctx.arc(sx, sy, ptR * 2, 0, 2 * Math.PI);
                    ctx.fillStyle = col1;
                    ctx.fill();
                    ctx.beginPath();
                    ctx.arc(sx, sy, ptR * 2, 0, 2 * Math.PI);
                    ctx.strokeStyle = 'rgba(224,242,254,0.72)';
                    ctx.lineWidth = 1.2;
                    ctx.stroke();
                });
                const { sx: c0sx, sy: c0sy } = toS(C0), { sx: c1sx, sy: c1sy } = toS(C1);
                ctx.globalAlpha = act * cT * 0.62;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';
                ctx.font = `600 ${cl(Math.min(mw, mh) * 0.054, 4.5, 7)}px "JetBrains Mono",monospace`;
                ctx.fillStyle = 'rgba(147,197,253,0.90)';
                ctx.fillText('ctrl', c0sx, c0sy - ptR * 2.8);
                ctx.fillStyle = 'rgba(99,140,255,0.96)';
                ctx.fillText('case', c1sx, c1sy - ptR * 2.8);
                ctx.globalAlpha = 1;
                const bdT = sm(cl((act - 0.62) * 4, 0, 1));
                if (bdT > 0.01) {
                    const { sx: s0x, sy: s0y } = toS(C0), { sx: s1x, sy: s1y } = toS(C1);
                    const midX = (s0x + s1x) / 2, midY = (s0y + s1y) / 2;
                    const dx = s1x - s0x, dy = s1y - s0y, len = Math.sqrt(dx * dx + dy * dy);
                    const nx = -dy / len, ny2 = dx / len, ext = Math.max(mw, mh) * 0.72;
                    ctx.globalAlpha = act * bdT * 0.42;
                    ctx.setLineDash([4, 3]);
                    ctx.beginPath();
                    ctx.moveTo(midX - nx * ext, midY - ny2 * ext);
                    ctx.lineTo(midX + nx * ext, midY + ny2 * ext);
                    ctx.strokeStyle = 'rgba(224,242,254,0.58)';
                    ctx.lineWidth = 0.8;
                    ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.globalAlpha = 1;
                }
            }
            const qT = sm(cl((act - 0.70) * 5, 0, 1));
            if (qT > 0.01) {
                const qx = mx + mw * (0.25 + Math.sin(ts * 0.024) * 0.19), qy = my + mh * (0.50 + Math.cos(ts * 0.019) * 0.17);
                const rg = ctx.createRadialGradient(qx, qy, 0, qx, qy, ptR * 3);
                rg.addColorStop(0, `rgba(250,204,21,${act * 0.68})`);
                rg.addColorStop(1, 'rgba(250,204,21,0)');
                ctx.beginPath();
                ctx.arc(qx, qy, ptR * 3, 0, 2 * Math.PI);
                ctx.fillStyle = rg;
                ctx.fill();
                ctx.beginPath();
                ctx.arc(qx, qy, ptR * 1.1, 0, 2 * Math.PI);
                ctx.fillStyle = `rgba(253,224,71,${act * 0.95})`;
                ctx.fill();
            }
            ctx.globalAlpha = act * 0.24;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'bottom';
            ctx.font = `400 ${cl(mw * 0.047, 4.5, 6.5)}px "JetBrains Mono",monospace`;
            ctx.fillStyle = 'rgba(147,197,253,0.65)';
            ctx.fillText('nearest centroid', mx + mw, my + mh);
            ctx.globalAlpha = 1;
        }
        function modelKNN(mx, my, mw, mh, act, ts) {
            const rng = makeLCG(227);
            const PTS = Array.from({ length: 22 }, (_, i) => ({ x: rng() * 0.82 + 0.09, y: rng() * 0.68 + 0.14, c: i < 11 }));
            const ptR = cl(Math.min(mw, mh) * 0.022, 1.5, 3.5);
            const QX = mx + mw * 0.50, QY = my + mh * 0.44;
            const K = 5;
            const DISTS = PTS.map((p, i) => {
                const dx = (mx + p.x * mw) - QX, dy = (my + p.y * mh) - QY;
                return { i, d: Math.sqrt(dx * dx + dy * dy) };
            }).sort((a, b) => a.d - b.d);
            const kRadius = DISTS[K - 1].d, kSet = new Set(DISTS.slice(0, K).map(d => d.i));
            PTS.forEach((p, pi) => {
                const pT = sm(cl((act - pi * 0.034) * 5, 0, 1));
                if (pT < 0.01)
                    return;
                const sx = mx + p.x * mw, sy = my + p.y * mh, isK = kSet.has(pi);
                ctx.globalAlpha = act * pT * (isK ? 0.92 : 0.42);
                ctx.beginPath();
                ctx.arc(sx, sy, ptR * (isK ? 1.45 : 1), 0, 2 * Math.PI);
                ctx.fillStyle = p.c ? 'rgba(99,140,255,0.88)' : 'rgba(147,197,253,0.65)';
                ctx.fill();
                if (isK) {
                    ctx.globalAlpha = act * pT * 0.55;
                    ctx.strokeStyle = 'rgba(224,242,254,0.82)';
                    ctx.lineWidth = 0.9;
                    ctx.stroke();
                }
                ctx.globalAlpha = 1;
            });
            const radT = sm(cl((act - 0.50) * 4, 0, 1));
            if (radT > 0.01) {
                ctx.globalAlpha = act * radT * 0.22;
                ctx.setLineDash([3, 2]);
                ctx.beginPath();
                ctx.arc(QX, QY, kRadius * (0.38 + radT * 0.62), 0, 2 * Math.PI);
                ctx.strokeStyle = 'rgba(224,242,254,0.78)';
                ctx.lineWidth = 0.8;
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.globalAlpha = 1;
            }
            const rg = ctx.createRadialGradient(QX, QY, 0, QX, QY, ptR * 3.2);
            rg.addColorStop(0, `rgba(250,204,21,${act * 0.66})`);
            rg.addColorStop(1, 'rgba(250,204,21,0)');
            ctx.beginPath();
            ctx.arc(QX, QY, ptR * 3.2, 0, 2 * Math.PI);
            ctx.fillStyle = rg;
            ctx.fill();
            ctx.beginPath();
            ctx.arc(QX, QY, ptR * 1.4, 0, 2 * Math.PI);
            ctx.fillStyle = `rgba(253,224,71,${act * 0.96})`;
            ctx.fill();
            const voteT = sm(cl((act - 0.65) * 4, 0, 1));
            if (voteT > 0.01) {
                const kNeigh = Array.from(kSet).map(ki => PTS[ki]);
                const votes = [kNeigh.filter(p => !p.c).length, kNeigh.filter(p => p.c).length];
                const bY = my + mh * 0.83, bH = cl(mh * 0.052, 2.5, 6), bX = mx + 4, halfW = (mw - 12) / 2;
                const BAR_DATA = [[votes[0], 'rgba(147,197,253,0.68)', 'ctrl'], [votes[1], 'rgba(99,140,255,0.92)', 'case']];
                BAR_DATA.forEach(([v, c, lbl], vi) => {
                    const x = bX + vi * (halfW + 4);
                    ctx.globalAlpha = act * voteT * 0.12;
                    rr(x, bY, halfW, bH, 2);
                    ctx.fillStyle = 'rgba(255,255,255,0.05)';
                    ctx.fill();
                    const fw = halfW * (v / K) * voteT;
                    if (fw > 1) {
                        ctx.globalAlpha = act * voteT * 0.88;
                        rr(x, bY, fw, bH, 2);
                        ctx.fillStyle = c;
                        ctx.fill();
                    }
                    ctx.globalAlpha = act * voteT * 0.50;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    ctx.font = `400 ${cl(mw * 0.047, 4.5, 6.5)}px "JetBrains Mono",monospace`;
                    ctx.fillStyle = 'rgba(147,197,253,0.75)';
                    ctx.fillText(`${lbl} ${v}/${K}`, x + halfW / 2, bY + bH + 2);
                    ctx.globalAlpha = 1;
                });
            }
            ctx.globalAlpha = act * 0.24;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'bottom';
            ctx.font = `400 ${cl(mw * 0.047, 4.5, 6.5)}px "JetBrains Mono",monospace`;
            ctx.fillStyle = 'rgba(147,197,253,0.65)';
            ctx.fillText('k=5  ·  distance vote', mx + mw, my + mh);
            ctx.globalAlpha = 1;
        }
        function modelSVC(mx, my, mw, mh, act, ts) {
            const rng = makeLCG(339);
            const PTS = Array.from({ length: 20 }, (_, i) => {
                const c = i < 10;
                return { x: cl((c ? 0.28 : 0.62) + rng() * 0.26 - 0.13, 0.05, 0.94), y: cl(0.50 + rng() * 0.50 - 0.25, 0.10, 0.90), c };
            });
            const ptR = cl(Math.min(mw, mh) * 0.022, 1.5, 3.5);
            const midX = mx + mw * 0.50, midY = my + mh * 0.50;
            const bAngle = 0.22;
            const cos = Math.cos(Math.PI / 2 + bAngle), sin = Math.sin(Math.PI / 2 + bAngle);
            const ext = Math.max(mw, mh) * 0.68;
            const marginPx = cl(Math.min(mw, mh) * 0.17, 16, 34);
            const dtb = (p) => Math.abs((mx + p.x * mw - midX) * cos + (my + p.y * mh - midY) * sin);
            const svSet = new Set([...PTS].sort((a, b) => dtb(a) - dtb(b)).slice(0, 4).map(p => PTS.indexOf(p)));
            const bdT = sm(cl((act - 0.55) * 4, 0, 1));
            if (bdT > 0.01) {
                ctx.globalAlpha = act * bdT * 0.08;
                [-1, 1].forEach(side => {
                    const ox = cos * marginPx * side, oy = sin * marginPx * side;
                    ctx.beginPath();
                    ctx.moveTo(midX + ox - sin * ext, midY + oy + cos * ext);
                    ctx.lineTo(midX + ox + sin * ext, midY + oy - cos * ext);
                    ctx.lineTo(midX + sin * ext, midY - cos * ext);
                    ctx.lineTo(midX - sin * ext, midY + cos * ext);
                    ctx.closePath();
                    ctx.fillStyle = 'rgba(147,197,253,0.28)';
                    ctx.fill();
                });
                ctx.globalAlpha = act * bdT * 0.32;
                ctx.setLineDash([3, 3]);
                [-1, 1].forEach(side => {
                    const ox = cos * marginPx * side, oy = sin * marginPx * side;
                    ctx.beginPath();
                    ctx.moveTo(midX + ox - sin * ext, midY + oy + cos * ext);
                    ctx.lineTo(midX + ox + sin * ext, midY + oy - cos * ext);
                    ctx.strokeStyle = 'rgba(147,197,253,0.60)';
                    ctx.lineWidth = 0.7;
                    ctx.stroke();
                });
                ctx.setLineDash([]);
                ctx.globalAlpha = 1;
                ctx.globalAlpha = act * bdT * 0.80;
                ctx.beginPath();
                ctx.moveTo(midX - sin * ext, midY + cos * ext);
                ctx.lineTo(midX + sin * ext, midY - cos * ext);
                ctx.strokeStyle = 'rgba(224,242,254,0.88)';
                ctx.lineWidth = 1.5;
                ctx.stroke();
                ctx.globalAlpha = 1;
            }
            if (bdT > 0.01) {
                const GX = 24, GY = 16, cw2 = mw / GX, ch2 = mh / GY;
                for (let gi = 0; gi < GX; gi++)
                    for (let gj = 0; gj < GY; gj++) {
                        const cx2 = mx + (gi + 0.5) * cw2, cy2 = my + (gj + 0.5) * ch2;
                        const d = ((cx2 - midX) * cos + (cy2 - midY) * sin) / marginPx;
                        const t2 = cl((d + 1.5) * 0.333, 0, 1);
                        const ri = Math.round(147 + (99 - 147) * t2), gi2 = Math.round(197 + (140 - 197) * t2), bi = Math.round(253 + (255 - 253) * t2);
                        ctx.globalAlpha = act * bdT * cl(0.065 - Math.abs(d) * 0.010, 0, 0.065);
                        ctx.fillStyle = `rgb(${ri},${gi2},${bi})`;
                        ctx.fillRect(cx2 - cw2 * 0.5, cy2 - ch2 * 0.5, cw2, ch2);
                        ctx.globalAlpha = 1;
                    }
            }
            PTS.forEach((p, pi) => {
                const pT = sm(cl((act - pi * 0.035) * 5, 0, 1));
                if (pT < 0.01)
                    return;
                const sx = mx + p.x * mw, sy = my + p.y * mh, isSV = svSet.has(pi);
                if (isSV && bdT > 0.01) {
                    const rg = ctx.createRadialGradient(sx, sy, 0, sx, sy, ptR * 3.2);
                    rg.addColorStop(0, p.c ? `rgba(99,140,255,${act * 0.42})` : `rgba(147,197,253,${act * 0.32})`);
                    rg.addColorStop(1, 'rgba(99,140,255,0)');
                    ctx.beginPath();
                    ctx.arc(sx, sy, ptR * 3.2, 0, 2 * Math.PI);
                    ctx.fillStyle = rg;
                    ctx.fill();
                }
                ctx.globalAlpha = act * pT * (isSV ? 1.0 : 0.52);
                ctx.beginPath();
                ctx.arc(sx, sy, ptR * (isSV ? 1.55 : 1), 0, 2 * Math.PI);
                ctx.fillStyle = p.c ? 'rgba(99,140,255,0.90)' : 'rgba(147,197,253,0.68)';
                ctx.fill();
                if (isSV) {
                    ctx.globalAlpha = act * pT * 0.72;
                    ctx.strokeStyle = 'rgba(224,242,254,0.88)';
                    ctx.lineWidth = 1.0;
                    ctx.stroke();
                }
                ctx.globalAlpha = 1;
            });
            ctx.globalAlpha = act * 0.24;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'bottom';
            ctx.font = `400 ${cl(mw * 0.047, 4.5, 6.5)}px "JetBrains Mono",monospace`;
            ctx.fillStyle = 'rgba(147,197,253,0.65)';
            ctx.fillText('max margin  ·  support vectors', mx + mw, my + mh);
            ctx.globalAlpha = 1;
        }
        function modelGPC(mx, my, mw, mh, act, ts) {
            const splitX = mx + mw * 0.42;
            const kmW = mw * 0.38, postW = mw * 0.54, postX = splitX + 4;
            const N = 8;
            const rng = makeLCG(451);
            const XS = Array.from({ length: N }, () => rng());
            const ell = 0.28;
            const cellS = Math.min(kmW / N, mh * 0.76 / N);
            const kmX0 = mx + (kmW - N * cellS) / 2;
            const kmY0 = my + mh * 0.12;
            ctx.globalAlpha = act * 0.22;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.font = `400 ${cl(mw * 0.048, 4.5, 7)}px "JetBrains Mono",monospace`;
            ctx.fillStyle = 'rgba(147,197,253,0.78)';
            ctx.fillText('K(x,x′)', mx, my);
            ctx.globalAlpha = 1;
            for (let i = 0; i < N; i++) {
                const rT = sm(cl((act - i * 0.06) * 5, 0, 1));
                if (rT < 0.01)
                    continue;
                for (let j = 0; j < N; j++) {
                    const cT = sm(cl((act - (i + j) * 0.030) * 4, 0, 1));
                    if (cT < 0.01)
                        continue;
                    const dx = XS[i] - XS[j];
                    const k = Math.exp(-(dx * dx) / (2 * ell * ell));
                    const osc = i !== j ? Math.sin(ts * 0.04 + i * 0.9 + j * 1.3) * 0.5 + 0.5 : 1;
                    const v = i === j ? 1.0 : k * (0.65 + osc * 0.35);
                    const r = Math.round(i === j ? 224 : 147 + (99 - 147) * v);
                    const g = Math.round(i === j ? 242 : 197 + (140 - 197) * v);
                    const b = Math.round(i === j ? 254 : 253 + (255 - 253) * v);
                    ctx.globalAlpha = act * cT * (0.45 + v * 0.50);
                    rr(kmX0 + j * cellS + 0.4, kmY0 + i * cellS + 0.4, cellS - 0.8, cellS - 0.8, 1.2);
                    ctx.fillStyle = `rgb(${r},${g},${b})`;
                    ctx.fill();
                    ctx.globalAlpha = 1;
                }
            }
            const lblT = sm(cl((act - 0.45) * 4, 0, 1));
            if (lblT > 0.01) {
                ctx.globalAlpha = act * lblT * 0.30;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.font = `400 ${cl(mw * 0.044, 4, 6.5)}px "JetBrains Mono",monospace`;
                ctx.fillStyle = 'rgba(147,197,253,0.70)';
                ctx.fillText(`RBF  ℓ=${ell}`, kmX0 + N * cellS / 2, kmY0 + N * cellS + 3);
                ctx.globalAlpha = 1;
            }
            const postH = mh * 0.68, postY0 = my + mh * 0.15;
            const nPts = 52;
            const gpMean = (t) => 3.2 * Math.tanh(3.0 * (t - 0.46));
            const gpStd = (t) => 0.28 + 0.62 * Math.pow(Math.abs(t - 0.5) * 2.0, 1.3);
            const fToY = (f) => postY0 + postH * (0.5 - cl(f, -3, 3) / 6);
            const postCT = sm(cl((act - 0.35) * 3, 0, 1));
            if (postCT > 0.01) {
                ctx.globalAlpha = act * postCT * 0.14;
                ctx.beginPath();
                for (let i = 0; i <= nPts; i++) {
                    const t = i / nPts;
                    const f = gpMean(t), s = gpStd(t) * 2;
                    const sx = postX + t * postW;
                    i === 0 ? ctx.moveTo(sx, fToY(f + s)) : ctx.lineTo(sx, fToY(f + s));
                }
                for (let i = nPts; i >= 0; i--) {
                    const t = i / nPts;
                    const f = gpMean(t), s = gpStd(t) * 2;
                    ctx.lineTo(postX + t * postW, fToY(f - s));
                }
                ctx.closePath();
                ctx.fillStyle = 'rgba(99,140,255,0.50)';
                ctx.fill();
                ctx.globalAlpha = 1;
                ctx.globalAlpha = act * postCT * 0.22;
                ctx.beginPath();
                for (let i = 0; i <= nPts; i++) {
                    const t = i / nPts;
                    const f = gpMean(t), s = gpStd(t);
                    const sx = postX + t * postW;
                    i === 0 ? ctx.moveTo(sx, fToY(f + s)) : ctx.lineTo(sx, fToY(f + s));
                }
                for (let i = nPts; i >= 0; i--) {
                    const t = i / nPts;
                    const f = gpMean(t), s = gpStd(t);
                    ctx.lineTo(postX + t * postW, fToY(f - s));
                }
                ctx.closePath();
                ctx.fillStyle = 'rgba(99,140,255,0.60)';
                ctx.fill();
                ctx.globalAlpha = 1;
                ctx.globalAlpha = act * postCT * 0.92;
                ctx.beginPath();
                for (let i = 0; i <= nPts; i++) {
                    const t = i / nPts, sx = postX + t * postW, sy = fToY(gpMean(t));
                    i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                }
                ctx.strokeStyle = 'rgba(99,140,255,0.95)';
                ctx.lineWidth = 1.6;
                ctx.stroke();
                ctx.globalAlpha = 1;
                const zeroY = fToY(0);
                ctx.globalAlpha = act * postCT * 0.22;
                ctx.setLineDash([2, 3]);
                ctx.beginPath();
                ctx.moveTo(postX, zeroY);
                ctx.lineTo(postX + postW, zeroY);
                ctx.strokeStyle = 'rgba(147,197,253,0.70)';
                ctx.lineWidth = 0.7;
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.globalAlpha = 1;
            }
            const probH = mh * 0.14, probY0 = my + mh * 0.86;
            const sigT = sm(cl((act - 0.62) * 4, 0, 1));
            if (sigT > 0.01) {
                ctx.globalAlpha = act * sigT * 0.18;
                ctx.fillStyle = 'rgba(15,23,42,0.30)';
                ctx.fillRect(postX, probY0, postW, probH);
                ctx.globalAlpha = act * sigT * 0.85;
                ctx.beginPath();
                for (let i = 0; i <= nPts; i++) {
                    const t = i / nPts, f = gpMean(t);
                    const p = 1 / (1 + Math.exp(-f));
                    const sx = postX + t * postW, sy = probY0 + probH * (1 - p);
                    i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
                }
                ctx.strokeStyle = 'rgba(165,200,255,0.90)';
                ctx.lineWidth = 1.2;
                ctx.stroke();
                ctx.globalAlpha = 1;
                ctx.globalAlpha = act * sigT * 0.25;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.font = `400 ${cl(mw * 0.044, 4, 6.5)}px "JetBrains Mono",monospace`;
                ctx.fillStyle = 'rgba(147,197,253,0.72)';
                ctx.fillText('σ(f) Laplace', postX + 1, probY0 + 1);
                ctx.globalAlpha = 1;
            }
            const obsT = sm(cl((act - 0.55) * 4, 0, 1));
            if (obsT > 0.5) {
                const t = (Math.sin(ts * 0.026) * 0.5 + 0.5) * 0.88 + 0.06;
                const fx = postX + t * postW, fy = fToY(gpMean(t));
                const std2 = gpStd(t) * 2;
                ctx.globalAlpha = act * obsT * 0.38;
                ctx.beginPath();
                ctx.moveTo(fx, fToY(gpMean(t) + std2));
                ctx.lineTo(fx, fToY(gpMean(t) - std2));
                ctx.strokeStyle = 'rgba(224,242,254,0.62)';
                ctx.lineWidth = 0.8;
                ctx.stroke();
                ctx.globalAlpha = 1;
                const rg = ctx.createRadialGradient(fx, fy, 0, fx, fy, 5);
                rg.addColorStop(0, `rgba(165,200,255,${act * 0.72})`);
                rg.addColorStop(1, 'rgba(99,140,255,0)');
                ctx.beginPath();
                ctx.arc(fx, fy, 5, 0, 2 * Math.PI);
                ctx.fillStyle = rg;
                ctx.fill();
                ctx.beginPath();
                ctx.arc(fx, fy, 2, 0, 2 * Math.PI);
                ctx.fillStyle = `rgba(224,242,254,${act * 0.96})`;
                ctx.fill();
            }
            ctx.globalAlpha = act * 0.22;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.font = `400 ${cl(mw * 0.048, 4.5, 7)}px "JetBrains Mono",monospace`;
            ctx.fillStyle = 'rgba(147,197,253,0.78)';
            ctx.fillText('f(x)', postX, my);
            ctx.globalAlpha = 1;
            ctx.globalAlpha = act * 0.22;
            ctx.textAlign = 'right';
            ctx.textBaseline = 'bottom';
            ctx.font = `400 ${cl(mw * 0.044, 4, 6.5)}px "JetBrains Mono",monospace`;
            ctx.fillStyle = 'rgba(147,197,253,0.65)';
            ctx.fillText('GP posterior  ·  Laplace approx', mx + mw, my + mh);
            ctx.globalAlpha = 1;
        }
        function draw() {
            tick++;
            const W = cv.offsetWidth, H = cv.offsetHeight;
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = '#f1f5f9';
            ctx.fillRect(0, 0, W, H);
            const rawPhase = (tick % CYCLE) / CYCLE;
            const cycleNum = Math.floor(tick / CYCLE);
            const iterN = cycleNum % 2;
            const r1Freeze = iterN === 1;
            const taxoIterN = cycleNum % 3;
            const taxoLo = taxoIterN === 2 ? 5 : 4;
            const taxoLabel = taxoIterN === 2 ? 'genus' : 'family – genus';
            const otuBrkLo = taxoIterN === 2 ? 10 : SEL_LO;
            const otuBrkHi = SEL_HI;
            const activeSelVals = taxoIterN === 2 ? GENUS_SEL_VALS : SEL_VALS;
            const activeClrVals = taxoIterN === 2 ? GENUS_CLR_VALS : CLR_VALS;
            const activeBinVals = taxoIterN === 2 ? GENUS_BIN_VALS : BIN_VALS;
            const activeNSel = taxoIterN === 2 ? N_GENUS : N_SEL;
            const activeGenLbls = taxoIterN === 2 ? GENUS_LBLS : GEN_LBLS;
            const phase = Math.min(rawPhase, 0.830);
            const rawPhase2 = rawPhase;
            const phase2 = Math.min(rawPhase2, 0.830);
            const masterA = 1 - sm(cl((rawPhase - 0.860) / 0.040, 0, 1));
            const TRANSFORM = iterN === 0 ? 'CLR' : 'BINARY';
            const TX_CHOSEN = iterN === 0 ? 0 : 4;
            const PAD = Math.max(7, W * 0.011);
            const GAP = Math.max(10, W * 0.018);
            const RGAP = Math.max(10, H * 0.035);
            const nC = 4;
            const pW = Math.floor((W - PAD * 2 - GAP * (nC - 1)) / nC);
            const rH = Math.floor((H - PAD * 2 - RGAP) / 2);
            const r1Y = PAD, r2Y = PAD + rH + RGAP;
            const pX = (i) => PAD + i * (pW + GAP);
            const HDR = 22;
            const contY = (ry) => ry + HDR + 6;
            const contH = rH - HDR - 8;
            const P6_WIN_START = 0.335, P6_WIN_END = 0.830;
            const p6MP = cl((rawPhase2 - P6_WIN_START) / (P6_WIN_END - P6_WIN_START), 0, 1);
            const lrnRaw = p6MP * LEARNERS.length;
            const lrnI = Math.min(Math.floor(lrnRaw), LEARNERS.length - 1);
            const lrnPhase = lrnI < LEARNERS.length - 1 ? lrnRaw % 1 : Math.min(lrnRaw - (LEARNERS.length - 1), 1);
            const lrn = LEARNERS[lrnI];
            const lrnFam = lrn === 'MLP' ? 'mlp' : lrn === 'Random Forest' ? 'rf' : lrn === 'Logistic Reg' ? 'lr' : lrn === 'BernoulliNB' ? 'bnb' : lrn === 'NearestCentroid' ? 'nc' : lrn === 'KNeighbors' ? 'knn' : lrn === 'SVC' ? 'svc' : lrn === 'GaussianProcess' ? 'gpc' : 'gbm';
            const a0 = cycleNum > 0 ? 1 : sl(phase, 0.00, 0.03);
            const a1 = cycleNum > 0 ? 1 : sl(phase, 0.02, 0.05);
            const a2 = cycleNum > 0 ? 1 : sl(phase, 0.04, 0.07);
            const a3 = cycleNum > 0 ? 1 : sl(phase, 0.06, 0.09);
            const a4 = sl(phase2, 0.10, 0.16) * masterA;
            const a5 = sl(phase2, 0.19, 0.25) * masterA;
            const a6 = sl(phase2, 0.31, 0.35) * masterA;
            const a7 = sl(phase2, 0.31, 0.35) * masterA;
            panelOp(pX(0), r1Y, pW, rH, 'Feature selection', '', a0);
            panel(pX(1), r1Y, pW, rH, 'Abundance matrix  · N×M', '', a1);
            panel(pX(2), r1Y, pW, rH, 'Abundance matrix  · selected features', '', a2);
            panel(pX(3), r1Y, pW, rH, 'Evaluation protocol  ·  Stratified nested CV', '', a3);
            panelOp(pX(3), r2Y, pW, rH, 'Compositional transform', '', a4);
            panel(pX(2), r2Y, pW, rH, 'Transformed abundance  · MPDR', '', a5);
            panel(pX(1), r2Y, pW, rH, `${lrn}  ·  MPMA-B`, '', a6);
            panel(pX(0), r2Y, pW, rH, 'Scoring', '', a7);
            const midY1 = r1Y + rH / 2, midY2 = r2Y + rH / 2;
            [0, 1, 2].forEach(gi => {
                const ab = Math.max([a0, a1, a2, a3][gi], [a0, a1, a2, a3][gi + 1]) * 0.6;
                if (ab < 0.02)
                    return;
                const ax = pX(gi) + pW + GAP / 2;
                ctx.globalAlpha = ab * 0.35;
                ctx.beginPath();
                ctx.moveTo(ax - 5, midY1 - 4);
                ctx.lineTo(ax, midY1);
                ctx.lineTo(ax - 5, midY1 + 4);
                ctx.strokeStyle = '#94a3b8';
                ctx.lineWidth = 1.2;
                ctx.lineJoin = 'round';
                ctx.stroke();
                ctx.globalAlpha = 1;
            });
            {
                const ax = pX(3) + pW / 2, ay1 = r1Y + rH + 2, ay2 = r2Y - 2;
                ctx.globalAlpha = Math.min(a3, a4) * 0.38;
                ctx.setLineDash([3, 3]);
                ctx.beginPath();
                ctx.moveTo(ax, ay1);
                ctx.lineTo(ax, ay2);
                ctx.strokeStyle = '#94a3b8';
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.beginPath();
                ctx.moveTo(ax - 4, ay2 - 6);
                ctx.lineTo(ax, ay2);
                ctx.lineTo(ax + 4, ay2 - 6);
                ctx.strokeStyle = '#94a3b8';
                ctx.lineWidth = 1.2;
                ctx.lineJoin = 'round';
                ctx.stroke();
                ctx.globalAlpha = 1;
            }
            ;
            [[a4, a5, pX(2) + pW + GAP / 2], [a5, a6, pX(1) + pW + GAP / 2], [a6, a7, pX(0) + pW + GAP / 2]].forEach(([aA, aB, ax]) => {
                const ab = Math.max(aA, aB) * 0.6;
                if (ab < 0.02)
                    return;
                ctx.globalAlpha = ab * 0.35;
                ctx.beginPath();
                ctx.moveTo(ax + 5, midY2 - 4);
                ctx.lineTo(ax, midY2);
                ctx.lineTo(ax + 5, midY2 + 4);
                ctx.strokeStyle = '#94a3b8';
                ctx.lineWidth = 1.2;
                ctx.lineJoin = 'round';
                ctx.stroke();
                ctx.globalAlpha = 1;
            });
            if (a0 > 0.01) {
                const ph0 = cycleNum > 0 ? 1.0 : sl(phase, 0.00, 0.06);
                taxoTree(pX(0) + 4, contY(r1Y), pW - 8, contH, ph0, a0, taxoLo, 5);
            }
            if (a1 > 0.01) {
                const ph1 = cycleNum > 0 ? 1.0 : sl(phase, 0.02, 0.08);
                otuMatrix(pX(1), contY(r1Y), pW, contH, ph1, a1, taxoLabel, otuBrkLo, otuBrkHi);
            }
            if (a2 > 0.01) {
                const ph2 = cycleNum > 0 ? 1.0 : sl(phase, 0.04, 0.10);
                reducedOTU(pX(2), contY(r1Y), pW, contH, ph2, a2, activeSelVals, activeNSel, activeGenLbls);
            }
            if (a3 > 0.01) {
                const ph3 = cycleNum > 0 ? 1.0 : sl(phase, 0.06, 0.12);
                foldPanel(pX(3), contY(r1Y), pW, contH, ph3, a3, activeSelVals, activeNSel);
            }
            if (a4 > 0.01) {
                const TX_NAMES = ['CLR', 'ALR', 'ILR', 'LOG', 'BINARY', 'RANK-STD', 'RANK-UNIT', 'ARCSIN-SQRT', 'HELLINGER', 'ZSCORE', 'TSS', 'NONE'];
                const CHOSEN = TX_CHOSEN;
                const fT = sm(cl((sl(rawPhase2, 0.12, 0.28) - 0.0) / 0.5, 0, 1));
                if (fT > 0.01) {
                    const opX = pX(3), opY = r2Y + HDR;
                    const rowH = cl(Math.floor(contH / TX_NAMES.length), 7, 36);
                    const startY = opY;
                    const SCAN_END = 0.52;
                    const scanning = fT < SCAN_END;
                    const scanIdx = scanning
                        ? Math.floor((fT / SCAN_END) * (TX_NAMES.length * 2)) % TX_NAMES.length
                        : CHOSEN;
                    const landT = sm(cl((fT - SCAN_END) * 8, 0, 1)) * sm(cl(1 - (fT - SCAN_END) * 2.5, 0, 1));
                    TX_NAMES.forEach((name, ti) => {
                        const itemT = sm(cl((fT - ti * 0.05) * 6, 0, 1));
                        if (itemT < 0.01)
                            return;
                        const iy = startY + ti * rowH + rowH / 2;
                        const isSel = ti === scanIdx;
                        const isLock = !scanning && ti === CHOSEN;
                        const bgA = isLock ? 0.10 + landT * 0.08 : (isSel && scanning ? 0.05 : 0);
                        if (bgA > 0.005) {
                            ctx.globalAlpha = a4 * itemT * bgA;
                            rr(opX + 8, iy - rowH * 0.48, pW - 16, rowH * 0.96, 3);
                            ctx.fillStyle = '#2563eb';
                            ctx.fill();
                            ctx.globalAlpha = 1;
                        }
                        const dotX = opX + 18, dotR = cl(rowH * 0.18, 2, 4);
                        const dotA = isLock ? 0.92 : (isSel && scanning ? 0.60 : 0.32);
                        ctx.globalAlpha = a4 * itemT * dotA;
                        ctx.beginPath();
                        ctx.arc(dotX, iy, dotR, 0, 2 * Math.PI);
                        if (isSel || isLock) {
                            ctx.fillStyle = '#2563eb';
                            ctx.fill();
                        }
                        else {
                            ctx.strokeStyle = '#94a3b8';
                            ctx.lineWidth = 0.8;
                            ctx.stroke();
                        }
                        ctx.globalAlpha = 1;
                        const lblA = isLock ? 0.92 : (isSel && scanning ? 0.65 : 0.38);
                        ctx.globalAlpha = a4 * itemT * lblA;
                        ctx.textAlign = 'left';
                        ctx.textBaseline = 'middle';
                        ctx.font = `${isLock ? '600' : isSel ? '500' : '400'} ${cl(rowH * 0.44, 7, 11)}px ${isLock ? '"JetBrains Mono",monospace' : 'Inter,sans-serif'}`;
                        ctx.fillStyle = isLock ? '#1d4ed8' : isSel && scanning ? '#2563eb' : '#64748b';
                        ctx.fillText(name, opX + 26, iy);
                        ctx.globalAlpha = 1;
                    });
                    const overflowT = sm(cl((fT - 0.55) * 4, 0, 1));
                    if (overflowT > 0.01) {
                        const listBot = startY + TX_NAMES.length * rowH;
                        const fadeH = cl(rowH * 2.4, 12, 24);
                        ctx.save();
                        ctx.beginPath();
                        ctx.rect(opX, r2Y, pW, rH);
                        ctx.clip();
                        const gBot = ctx.createLinearGradient(0, listBot - fadeH, 0, listBot);
                        gBot.addColorStop(0, 'rgba(241,245,249,0)');
                        gBot.addColorStop(1, `rgba(241,245,249,${(a4 * overflowT * 0.90).toFixed(2)})`);
                        ctx.fillStyle = gBot;
                        ctx.fillRect(opX + 4, listBot - fadeH, pW - 8, fadeH);
                        ctx.globalAlpha = a4 * overflowT * 0.36;
                        ctx.textAlign = 'left';
                        ctx.textBaseline = 'middle';
                        ctx.font = `400 ${cl(rowH * 0.44, 7, 9)}px Inter,sans-serif`;
                        ctx.fillStyle = '#94a3b8';
                        ctx.fillText('···', opX + 26, listBot - rowH * 0.55);
                        ctx.globalAlpha = 1;
                        ctx.restore();
                    }
                    if (landT > 0.02) {
                        const ly = startY + CHOSEN * rowH + rowH / 2;
                        ctx.globalAlpha = a4 * landT * 0.55;
                        rr(opX + 8, ly - rowH * 0.48, pW - 16, rowH * 0.96, 3);
                        ctx.strokeStyle = '#2563eb';
                        ctx.lineWidth = 1;
                        ctx.stroke();
                        ctx.globalAlpha = 1;
                    }
                }
            }
            if (a5 > 0.01) {
                const ph5 = sl(phase2, 0.20, 0.32);
                if (iterN === 0)
                    clrMatrix(pX(2), contY(r2Y), pW, contH, ph5, a5, activeClrVals, activeNSel);
                else
                    binMatrix(pX(2), contY(r2Y), pW, contH, ph5, a5, activeBinVals, activeNSel);
            }
            if (a6 > 0.02) {
                ctx.globalAlpha = a6;
                rr(pX(1) + 2, r2Y + HDR + 2, pW - 4, contH - 2, 5);
                ctx.fillStyle = '#0f172a';
                ctx.fill();
                ctx.save();
                ctx.beginPath();
                ctx.rect(pX(1) + 2, r2Y + HDR + 2, pW - 4, contH - 2);
                ctx.clip();
                const actML = sl(rawPhase2, 0.320, 0.335) * masterA;
                const modelTs = Math.min(lrnPhase, 0.44) / 0.44 * 200;
                const lrnBuild = sl(lrnPhase, 0.00, 0.46) * actML;
                const trainDone = sl(lrnPhase, 0.44, 0.52);
                if (actML > 0.04) {
                    rr(pX(1) + 1, r2Y + HDR + 1, pW - 2, contH, 6);
                    const bR = Math.round(37 + trainDone * (34 - 37));
                    const bG = Math.round(99 + trainDone * (197 - 99));
                    const bB = Math.round(235 + trainDone * (94 - 235));
                    ctx.strokeStyle = `rgba(${bR},${bG},${bB},${actML * 0.35})`;
                    ctx.lineWidth = 1.5;
                    ctx.stroke();
                }
                ctx.textAlign = 'right';
                ctx.textBaseline = 'top';
                ctx.font = `700 ${cl(pW * 0.065, 9, 13)}px Inter,sans-serif`;
                ctx.fillStyle = `rgba(147,197,253,${0.50 + actML * 0.45})`;
                ctx.fillText(lrn, pX(1) + pW - 6, r2Y + HDR + 6);
                if (trainDone > 0.05) {
                    ctx.globalAlpha = a6 * trainDone * 0.80;
                    ctx.font = `500 ${cl(pW * 0.052, 7, 10)}px "JetBrains Mono",monospace`;
                    ctx.fillStyle = '#4ade80';
                    ctx.fillText('trained \u2713', pX(1) + pW - 6, r2Y + HDR + 18);
                }
                ctx.globalAlpha = 1;
                const vP = 6;
                if (lrnFam === 'rf')
                    modelRF(pX(1) + vP + 2, r2Y + HDR + vP + 18, pW - vP * 2 - 4, contH - vP * 2 - 22, lrnBuild, modelTs);
                else if (lrnFam === 'gbm')
                    modelGBM(pX(1) + vP + 2, r2Y + HDR + vP + 18, pW - vP * 2 - 4, contH - vP * 2 - 22, lrnBuild, modelTs);
                else if (lrnFam === 'lr')
                    modelLR(pX(1) + vP + 2, r2Y + HDR + vP + 18, pW - vP * 2 - 4, contH - vP * 2 - 22, lrnBuild, modelTs);
                else if (lrnFam === 'bnb')
                    modelBNB(pX(1) + vP + 2, r2Y + HDR + vP + 18, pW - vP * 2 - 4, contH - vP * 2 - 22, lrnBuild, modelTs);
                else if (lrnFam === 'nc')
                    modelNC(pX(1) + vP + 2, r2Y + HDR + vP + 18, pW - vP * 2 - 4, contH - vP * 2 - 22, lrnBuild, modelTs);
                else if (lrnFam === 'knn')
                    modelKNN(pX(1) + vP + 2, r2Y + HDR + vP + 18, pW - vP * 2 - 4, contH - vP * 2 - 22, lrnBuild, modelTs);
                else if (lrnFam === 'svc')
                    modelSVC(pX(1) + vP + 2, r2Y + HDR + vP + 18, pW - vP * 2 - 4, contH - vP * 2 - 22, lrnBuild, modelTs);
                else if (lrnFam === 'gpc')
                    modelGPC(pX(1) + vP + 2, r2Y + HDR + vP + 18, pW - vP * 2 - 4, contH - vP * 2 - 22, lrnBuild, modelTs);
                else
                    modelMLP(pX(1) + vP + 2, r2Y + HDR + vP + 18, pW - vP * 2 - 4, contH - vP * 2 - 22, lrnBuild, modelTs);
                ctx.restore();
            }
            if (a7 > 0.01) {
                const LRN_METRICS = [
                    [0.9320, 0.7140, 0.9070, 0.9250, 0.9130],
                    [0.9180, 0.6890, 0.8920, 0.9100, 0.8980],
                    [0.9050, 0.6620, 0.8810, 0.8960, 0.8850],
                    [0.9350, 0.7210, 0.9110, 0.9280, 0.9160],
                    [0.8820, 0.6210, 0.8540, 0.8680, 0.8600],
                    [0.9120, 0.6780, 0.8860, 0.9020, 0.8920],
                ];
                const VALS = LRN_METRICS[lrnI];
                const actML = sl(rawPhase2, 0.335, 0.350) * masterA;
                const act = sl(lrnPhase, 0.44, 0.86) * actML;
                const bT = sm(cl(act * 2.5, 0, 1));
                const ox = pX(0), oy = contY(r2Y);
                const METRICS = ['AUC', 'nMCC', 'F1w', 'Prec', 'Rec'];
                const lW = cl(pW * 0.175, 20, 34);
                const valW = cl(pW * 0.22, 28, 44);
                const bX = ox + lW + 5;
                const bTW = pW - lW - valW - 10;
                const mRows = METRICS.length;
                const mRowH = Math.max(9, Math.floor((contH - 9) / 6));
                const mStart = oy + 2;
                METRICS.forEach((m, mi) => {
                    const mT = sm(cl((bT - mi * 0.07) * 6, 0, 1));
                    if (mT < 0.01)
                        return;
                    const my = mStart + mi * mRowH + mRowH * 0.5;
                    const barH = cl(mRowH * 0.32, 2, 6);
                    ctx.globalAlpha = a7 * mT * 0.48;
                    ctx.textAlign = 'right';
                    ctx.textBaseline = 'middle';
                    ctx.font = `600 ${cl(mRowH * 0.40, 5.5, 8.5)}px "JetBrains Mono",monospace`;
                    ctx.fillStyle = '#94a3b8';
                    ctx.fillText(m, ox + lW, my);
                    ctx.globalAlpha = a7 * mT * 0.10;
                    rr(bX, my - barH / 2, bTW, barH, barH / 2);
                    ctx.fillStyle = '#cbd5e1';
                    ctx.fill();
                    const barW = bTW * VALS[mi] * cl(act * 3, 0, 1);
                    if (barW > 1) {
                        ctx.globalAlpha = a7 * mT * 0.88;
                        rr(bX, my - barH / 2, barW, barH, barH / 2);
                        ctx.fillStyle = '#2563eb';
                        ctx.fill();
                    }
                    ctx.globalAlpha = 1;
                    const valX = ox + pW - 4;
                    ctx.globalAlpha = a7 * mT * 0.85;
                    ctx.textAlign = 'right';
                    ctx.textBaseline = 'middle';
                    ctx.font = `700 ${cl(mRowH * 0.42, 6, 9)}px "JetBrains Mono",monospace`;
                    ctx.fillStyle = '#2563eb';
                    ctx.fillText((VALS[mi] * 100).toFixed(1) + '%', valX, my);
                    ctx.globalAlpha = 1;
                });
                const divY = mStart + mRows * mRowH + 5;
                ctx.globalAlpha = a7 * bT * 0.14;
                ctx.beginPath();
                ctx.moveTo(ox + 4, divY);
                ctx.lineTo(ox + pW - 4, divY);
                ctx.strokeStyle = '#94a3b8';
                ctx.lineWidth = 0.6;
                ctx.stroke();
                ctx.globalAlpha = 1;
                const SELECTION_SCORES = [0.879, 0.857, 0.841, 0.883, 0.832, 0.864];
                const selectionScore = SELECTION_SCORES[lrnI];
                const hT2 = sm(cl((act - 0.35) * 4, 0, 1));
                if (hT2 > 0.01) {
                    const hRowH = mRowH;
                    const hY = divY + 2;
                    const hBarH = cl(mRowH * 0.32, 2, 6);
                    const hBarY = hY + hRowH / 2 - hBarH / 2;
                    ctx.globalAlpha = a7 * hT2 * 0.55;
                    ctx.textAlign = 'right';
                    ctx.textBaseline = 'middle';
                    ctx.font = `600 ${cl(hRowH * 0.40, 5.5, 8.5)}px "JetBrains Mono",monospace`;
                    ctx.fillStyle = '#94a3b8';
                    ctx.fillText('selection', ox + lW, hY + hRowH / 2);
                    ctx.globalAlpha = a7 * hT2 * 0.10;
                    rr(bX, hBarY, bTW, hBarH, hBarH / 2);
                    ctx.fillStyle = '#cbd5e1';
                    ctx.fill();
                    const hBarW = bTW * selectionScore * cl(act * 3, 0, 1);
                    if (hBarW > 1) {
                        const hG = ctx.createLinearGradient(bX, 0, bX + hBarW, 0);
                        hG.addColorStop(0, '#3b82f6');
                        hG.addColorStop(1, '#6366f1');
                        ctx.globalAlpha = a7 * hT2 * 0.92;
                        rr(bX, hBarY, hBarW, hBarH, hBarH / 2);
                        ctx.fillStyle = hG;
                        ctx.fill();
                        ctx.globalAlpha = 1;
                    }
                    const valX2 = ox + pW - 4;
                    ctx.globalAlpha = a7 * hT2 * 0.92;
                    ctx.textAlign = 'right';
                    ctx.textBaseline = 'middle';
                    ctx.font = `700 ${cl(hRowH * 0.42, 6, 9)}px "JetBrains Mono",monospace`;
                    ctx.fillStyle = '#6366f1';
                    ctx.fillText((selectionScore * 100).toFixed(1) + '%', valX2, hY + hRowH / 2);
                    ctx.globalAlpha = 1;
                }
            }
            raf = requestAnimationFrame(draw);
        }
        draw();
        return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize); };
    }, []);
    return <canvas ref={ref} style={{ width: '100%', height: '100%', display: 'block' }}/>;
}
export function EnsembleSweepCanvas() {
    const ref = useRef(null);
    useEffect(() => {
        const cv = ref.current;
        if (!cv)
            return;
        const dpr = Math.max(3, window.devicePixelRatio || 1);
        let raf, tick = 0;
        const resize = () => { cv.width = cv.offsetWidth * dpr; cv.height = cv.offsetHeight * dpr; };
        resize();
        window.addEventListener('resize', resize);
        const CYCLE = 3000;
        const sm = (x) => 1 / (1 + Math.exp(-7 * (x - 0.5)));
        const cl = (v, a, b) => Math.max(a, Math.min(b, v));
        const sl = (p, a, b) => sm(cl((p - a) / (b - a), 0, 1));
        const lrp = (a, b, t) => a + (b - a) * t;
        const INK = '#0f172a';
        const MID = '#64748b';
        const DIM = '#94a3b8';
        const TRACK = '#e9eef5';
        const ACC = '#2563eb';
        const BG = '#f8fafc';
        const MPMAS = [
            { cfg: 'genus  ·  CLR  ·  Random Forest', score: 0.8790, qual: true },
            { cfg: 'species  ·  CLR  ·  CatBoost', score: 0.8830, qual: true },
            { cfg: 'family  ·  CLR  ·  LightGBM', score: 0.8640, qual: true },
            { cfg: 'genus  ·  binary  ·  MLP', score: 0.8020, qual: false },
            { cfg: 'phylum–genus  ·  CLR  ·  XGBoost', score: 0.8580, qual: true },
            { cfg: 'domain–species  ·  binary  ·  Log Reg', score: 0.8120, qual: false },
            { cfg: 'genus  ·  CLR  ·  SVC', score: 0.8510, qual: true },
            { cfg: 'family  ·  rank-std  ·  LightGBM', score: 0.8360, qual: false },
            { cfg: 'phylum–species  ·  CLR  ·  XGBoost', score: 0.8690, qual: true },
            { cfg: 'genus  ·  binary  ·  Random Forest', score: 0.8040, qual: false },
            { cfg: 'order–genus  ·  CLR  ·  LightGBM', score: 0.8170, qual: false },
            { cfg: 'class–family  ·  rank-std  ·  MLP', score: 0.7930, qual: false },
            { cfg: 'phylum  ·  CLR  ·  Gaussian Process', score: 0.8280, qual: false },
            { cfg: 'genus  ·  rank-std  ·  KNN', score: 0.7870, qual: false },
        ];
        const ENS_MEMBERS = MPMAS.filter(m => m.qual);
        const WEIGHTS = [0.24, 0.22, 0.18, 0.16, 0.12, 0.08];
        const STRATS = [
            'top_k', 'stratified', 'diverse', 'diverse_families',
            'clustered', 'iterative', 'interpretable', 'maximal_diversity',
            'borda_ranking', 'greedy_forward', 'superlearner', 'shapley_value',
            'best_per_family',
        ];
        const STRAT_SCORE = [0.868, 0.878, 0.881, 0.893, 0.876, 0.872, 0.851, 0.875, 0.869, 0.874, 0.871, 0.870, 0.862];
        const BEST_SI = 3;
        const BEST_SCORE = 0.893;
        const THR = 0.820;
        const AGG_STRATS = [
            'voting', 'weighted_voting',
            'probability_averaging', 'weighted_prob_averaging', 'geometric_mean', 'trimmed_mean', 'median_probability',
            'stacking_lr', 'hierarchical_stacking', 'bayesian_model_averaging', 'superlearner',
            'confidence_weighted', 'temperature_scaled',
            'rank_aggregation',
        ];
        const AGG_SCORE = [0.871, 0.876, 0.884, 0.888, 0.881, 0.879, 0.877, 0.901, 0.892, 0.889, 0.895, 0.886, 0.883, 0.874];
        const BEST_AGG_I = 7;
        const BEST_AGG_SCORE = 0.901;
        function draw() {
            tick++;
            const ctx = cv.getContext('2d');
            const W = cv.offsetWidth, H = cv.offsetHeight;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = BG;
            ctx.fillRect(0, 0, W, H);
            const SPLIT = 500 / CYCLE;
            const splitP = (tick % CYCLE) / CYCLE;
            const rawP = splitP < SPLIT
                ? splitP * (0.84 / SPLIT)
                : 0.84 + (splitP - SPLIT) * (0.16 / (1 - SPLIT));
            const phase = Math.min(rawP, 0.960);
            const masterA = 1 - sm(cl((rawP - 0.990) / 0.010, 0, 1));
            const PX = 20, PY = 18;
            const cW = W - PX * 2, cH = H - PY * 2;
            const C0W = Math.floor(cW * 0.19);
            const C3W = Math.floor(cW * 0.25);
            const C12 = cW - C0W - C3W - 48;
            const C1W = Math.floor(C12 * 0.53);
            const C2W = C12 - C1W;
            const c0 = PX, c1 = PX + C0W + 16, c2 = c1 + C1W + 16, c3 = c2 + C2W + 16;
            const sepA = sl(phase, 0.01, 0.07) * masterA;
            [c1 - 8, c2 - 8, c3 - 8].forEach(sx => {
                ctx.globalAlpha = sepA * 0.15;
                ctx.beginPath();
                ctx.moveTo(sx, PY + 14);
                ctx.lineTo(sx, PY + cH);
                ctx.strokeStyle = MID;
                ctx.lineWidth = 0.6;
                ctx.stroke();
                ctx.globalAlpha = 1;
            });
            const lblA = sl(phase, 0.01, 0.08) * masterA;
            [
                [c0, 'candidate pool'],
                [c1, 'member selection'],
                [c2, 'aggregation strategy'],
                [c3, 'optimal ensemble'],
            ].forEach(([x, t]) => {
                ctx.globalAlpha = lblA * 0.32;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.font = `400 7px "JetBrains Mono",monospace`;
                ctx.fillStyle = MID;
                ctx.fillText(t, x, PY);
                ctx.globalAlpha = 1;
            });
            const TOP = PY + 16;
            const C_PH = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33, 0.36, 0.39, 0.42];
            const C_QPH = [0.08, 0.11, 0.14, 0.17, 0.20, 0.23, 0.26, 0.29, 0.32, 0.35, 0.38, 0.41, 0.44, 0.47];
            const rowGap = 4;
            const slotH = Math.max(10, Math.floor((cH - 16) / MPMAS.length));
            const rowH = slotH - rowGap;
            MPMAS.forEach((m, mi) => {
                const cT = sl(phase, C_PH[mi], C_PH[mi] + 0.08) * masterA;
                if (cT < 0.01)
                    return;
                const ry = TOP + mi * (rowH + rowGap);
                const qT = sl(phase, C_QPH[mi], C_QPH[mi] + 0.10) * masterA;
                const cfgY = ry + Math.round(rowH * 0.38);
                const barY = ry + Math.round(rowH * 0.78);
                const barH = 1.5;
                const barW = C0W;
                const scoreY = barY + barH / 2;
                const cfgT = sl(phase, C_PH[mi] + 0.01, C_PH[mi] + 0.07) * masterA;
                if (cfgT > 0.01) {
                    ctx.globalAlpha = cfgT * (m.qual ? 0.78 : 0.35);
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    ctx.font = `${m.qual ? '500' : '400'} 6.5px Inter,sans-serif`;
                    ctx.fillStyle = m.qual ? INK : DIM;
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(c0, ry, C0W - 36, rowH * 0.65);
                    ctx.clip();
                    ctx.fillText(m.cfg, c0, cfgY);
                    ctx.restore();
                    ctx.globalAlpha = 1;
                }
                if (qT > 0.01) {
                    ctx.globalAlpha = qT * (m.qual ? 0.82 : 0.38);
                    ctx.textAlign = 'right';
                    ctx.textBaseline = 'middle';
                    ctx.font = `${m.qual ? '500' : '400'} 6px "JetBrains Mono",monospace`;
                    ctx.fillStyle = m.qual ? ACC : '#94a3b8';
                    ctx.fillText(m.score.toFixed(3), c0 + C0W, scoreY);
                    ctx.globalAlpha = 1;
                }
                if (qT > 0.01) {
                    ctx.globalAlpha = qT * (m.qual ? 0.12 : 0.07);
                    ctx.fillStyle = TRACK;
                    ctx.fillRect(c0, barY, barW, barH);
                    ctx.globalAlpha = 1;
                    const fw = barW * m.score * cl(sl(phase, C_QPH[mi], C_QPH[mi] + 0.07), 0, 1);
                    if (fw > 0.5) {
                        ctx.globalAlpha = qT * (m.qual ? 0.88 : 0.40);
                        ctx.fillStyle = m.qual ? ACC : '#94a3b8';
                        ctx.fillRect(c0, barY, fw, barH);
                        ctx.globalAlpha = 1;
                    }
                    const thrX = c0 + barW * THR;
                    ctx.globalAlpha = qT * 0.18;
                    ctx.beginPath();
                    ctx.moveTo(thrX, barY);
                    ctx.lineTo(thrX, barY + barH);
                    ctx.strokeStyle = MID;
                    ctx.lineWidth = 0.6;
                    ctx.stroke();
                    ctx.globalAlpha = 1;
                }
            });
            const poolReadyT = sl(phase, 0.58, 0.64) * masterA;
            if (poolReadyT > 0.01) {
                const pRy = TOP + MPMAS.length * (rowH + rowGap) - rowGap / 2;
                ctx.globalAlpha = poolReadyT * 0.18;
                ctx.beginPath();
                ctx.moveTo(c0, pRy);
                ctx.lineTo(c0 + C0W, pRy);
                ctx.strokeStyle = MID;
                ctx.lineWidth = 0.6;
                ctx.stroke();
                ctx.globalAlpha = poolReadyT * 0.40;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.font = `400 6.5px "JetBrains Mono",monospace`;
                ctx.fillStyle = ACC;
                ctx.fillText('6 qualified  ·  sweep ready', c0, pRy + 4);
                ctx.globalAlpha = 1;
            }
            const swStart = 0.64, swEnd = 0.84;
            const sweepFrac = cl((phase - swStart) / (swEnd - swStart), 0, 1);
            const locked = sweepFrac >= 0.999;
            const activeSI = locked ? BEST_SI : Math.floor(sweepFrac * STRATS.length);
            const nSt = STRATS.length;
            const stRowH = cl((cH - 16) / nSt, 11, 26);
            const stValW = 34;
            const stBarW = Math.round(C1W * 0.22);
            const stLblW = C1W - 2 - 5 - 6 - stBarW - 4 - stValW;
            const stAccX = c1;
            const stTxtX = c1 + 7;
            const stBarX = c1 + 2 + 5 + stLblW + 6;
            const stValX = stBarX + stBarW + 4;
            STRATS.forEach((s, si) => {
                const sT = sl(phase, swStart + si * (swEnd - swStart) / nSt * 0.45, swStart + si * (swEnd - swStart) / nSt * 0.45 + 0.025) * masterA;
                if (sT < 0.01)
                    return;
                const sy = TOP + si * stRowH + stRowH / 2;
                const isBest = locked && si === BEST_SI;
                const isActive = !locked && si === activeSI;
                const rowA = locked ? (isBest ? 1.0 : 0.28) : (isActive ? 1.0 : 0.50);
                if (isBest) {
                    ctx.globalAlpha = sT * 0.85;
                    ctx.fillStyle = ACC;
                    ctx.fillRect(stAccX, sy - stRowH * 0.44, 1.5, stRowH * 0.88);
                    ctx.globalAlpha = 1;
                }
                const lblFS = cl(stRowH * 0.44, 6, 9);
                ctx.globalAlpha = sT * rowA * (isBest ? 0.90 : 0.65);
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.font = `${isBest ? '500' : '400'} ${lblFS}px ${isBest ? '"JetBrains Mono",monospace' : 'Inter,sans-serif'}`;
                ctx.fillStyle = isBest ? INK : MID;
                ctx.save();
                ctx.beginPath();
                ctx.rect(stTxtX, sy - stRowH * 0.5, stLblW, stRowH);
                ctx.clip();
                ctx.fillText(s, stTxtX, sy);
                ctx.restore();
                ctx.globalAlpha = 1;
                ctx.globalAlpha = sT * rowA * 0.12;
                ctx.fillStyle = TRACK;
                ctx.fillRect(stBarX, sy - 1.5, stBarW, 3);
                ctx.globalAlpha = 1;
                const barFillT = locked ? 1.0 : (isActive ? cl(sweepFrac * nSt - si, 0, 1) : (si < activeSI ? 1.0 : 0));
                if (barFillT > 0.001) {
                    ctx.globalAlpha = sT * rowA * (isBest ? 0.88 : 0.50);
                    ctx.fillStyle = isBest ? ACC : DIM;
                    ctx.fillRect(stBarX, sy - 1.5, stBarW * barFillT, 3);
                    ctx.globalAlpha = 1;
                }
                if (barFillT > 0.8) {
                    const valFS = cl(stRowH * 0.40, 5.5, 8);
                    ctx.globalAlpha = sT * rowA * (isBest ? 0.88 : 0.48);
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    ctx.font = `${isBest ? '600' : '400'} ${valFS}px "JetBrains Mono",monospace`;
                    ctx.fillStyle = isBest ? ACC : DIM;
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(stValX, sy - stRowH * 0.5, stValW, stRowH);
                    ctx.clip();
                    ctx.fillText(STRAT_SCORE[si].toFixed(3), stValX, sy);
                    ctx.restore();
                    ctx.globalAlpha = 1;
                }
            });
            const nAgg = AGG_STRATS.length;
            const aggRowH = cl((cH - 16) / nAgg, 13, 30);
            const aggRowGap = 0;
            const aggSwStart = 0.66, aggSwEnd = 0.84;
            const aggSweepFrac = cl((phase - aggSwStart) / (aggSwEnd - aggSwStart), 0, 1);
            const aggLocked = aggSweepFrac >= 0.999;
            const activeAI = aggLocked ? BEST_AGG_I : Math.floor(aggSweepFrac * nAgg);
            const aValW = 34;
            const aBarW = Math.round(C2W * 0.22);
            const aLblW = C2W - 2 - 5 - 6 - aBarW - 4 - aValW;
            const aAccX = c2;
            const aTxtX = c2 + 7;
            const aBarX = c2 + 2 + 5 + aLblW + 6;
            const aValX = aBarX + aBarW + 4;
            AGG_STRATS.forEach((aName, ai) => {
                const aPhStart = aggSwStart + (ai / nAgg) * (aggSwEnd - aggSwStart) * 0.48;
                const aT = sl(phase, aPhStart, aPhStart + 0.024) * masterA;
                if (aT < 0.01)
                    return;
                const ay = TOP + ai * aggRowH + aggRowH / 2;
                const isBestA = aggLocked && ai === BEST_AGG_I;
                const isActiveA = !aggLocked && ai === activeAI;
                const aRowA = aggLocked ? (isBestA ? 1.0 : 0.28) : (isActiveA ? 1.0 : 0.50);
                if (isBestA) {
                    ctx.globalAlpha = aT * 0.80;
                    ctx.fillStyle = 'rgba(167,139,250,0.90)';
                    ctx.fillRect(aAccX, ay - aggRowH * 0.44, 1.5, aggRowH * 0.88);
                    ctx.globalAlpha = 1;
                }
                const aLblFS = cl(aggRowH * 0.38, 6, 9);
                ctx.globalAlpha = aT * aRowA * (isBestA ? 0.92 : 0.60);
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.font = `${isBestA ? '500' : '400'} ${aLblFS}px ${isBestA ? '"JetBrains Mono",monospace' : 'Inter,sans-serif'}`;
                ctx.fillStyle = isBestA ? 'rgba(167,139,250,0.95)' : MID;
                ctx.save();
                ctx.beginPath();
                ctx.rect(aTxtX, ay - aggRowH * 0.5, aLblW, aggRowH);
                ctx.clip();
                ctx.fillText(aName, aTxtX, ay);
                ctx.restore();
                ctx.globalAlpha = 1;
                ctx.globalAlpha = aT * aRowA * 0.10;
                ctx.fillStyle = TRACK;
                ctx.fillRect(aBarX, ay - 1.5, aBarW, 3);
                ctx.globalAlpha = 1;
                const aBarFill = aggLocked ? 1.0
                    : (isActiveA ? cl(aggSweepFrac * nAgg - ai, 0, 1) : (ai < activeAI ? 1.0 : 0));
                if (aBarFill > 0.001) {
                    ctx.globalAlpha = aT * aRowA * (isBestA ? 0.85 : 0.45);
                    ctx.fillStyle = isBestA ? 'rgba(167,139,250,0.90)' : 'rgba(196,181,253,0.55)';
                    ctx.fillRect(aBarX, ay - 1.5, aBarW * aBarFill, 3);
                    ctx.globalAlpha = 1;
                }
                if (aBarFill > 0.8) {
                    const aValFS = cl(aggRowH * 0.34, 5.5, 8);
                    ctx.globalAlpha = aT * aRowA * (isBestA ? 0.88 : 0.45);
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    ctx.font = `${isBestA ? '600' : '400'} ${aValFS}px "JetBrains Mono",monospace`;
                    ctx.fillStyle = isBestA ? 'rgba(167,139,250,0.95)' : DIM;
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(aValX, ay - aggRowH * 0.5, aValW, aggRowH);
                    ctx.clip();
                    ctx.fillText(AGG_SCORE[ai].toFixed(3), aValX, ay);
                    ctx.restore();
                    ctx.globalAlpha = 1;
                }
            });
            if (rawP > 0.84) {
                let ry2 = TOP;
                const a = masterA;
                ctx.globalAlpha = a * 0.88;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.font = `500 7.5px "JetBrains Mono",monospace`;
                ctx.fillStyle = ACC;
                ctx.fillText(`selection: ${STRATS[BEST_SI]}`, c3, ry2);
                ry2 += 12;
                ctx.globalAlpha = a * 0.80;
                ctx.font = `500 7.5px "JetBrains Mono",monospace`;
                ctx.fillStyle = 'rgba(167,139,250,0.90)';
                ctx.fillText(`aggregation: ${AGG_STRATS[BEST_AGG_I]}`, c3, ry2);
                ry2 += 16;
                const scoreFS = cl(cH * 0.11, 18, 34);
                const hairY = cH - PY - scoreFS - 10;
                const rowsAvail = hairY - ry2 - 8;
                const mGap = 3;
                const mRowH = cl(Math.floor((rowsAvail - mGap * 5) / 6), 14, 44);
                ENS_MEMBERS.forEach((m, mi) => {
                    const mry = ry2 + mi * (mRowH + mGap);
                    const [taxon, transform, learner] = m.cfg.split('·').map((p) => p.trim());
                    const wBarH = Math.round(mRowH * WEIGHTS[mi] * 0.9);
                    ctx.globalAlpha = a * 0.15;
                    ctx.fillStyle = ACC;
                    ctx.fillRect(c3, mry, 2, mRowH);
                    ctx.globalAlpha = a * 0.70;
                    ctx.fillStyle = ACC;
                    ctx.fillRect(c3, mry + mRowH - wBarH, 2, wBarH);
                    ctx.globalAlpha = a * 0.38;
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'top';
                    ctx.font = `400 5.5px "JetBrains Mono",monospace`;
                    ctx.fillStyle = DIM;
                    ctx.fillText(`MPMA-B ${mi + 1}`, c3 + 7, mry + 1);
                    ctx.globalAlpha = a * 0.85;
                    ctx.textAlign = 'right';
                    ctx.font = `600 7px "JetBrains Mono",monospace`;
                    ctx.fillStyle = ACC;
                    ctx.fillText(`${(WEIGHTS[mi] * 100).toFixed(0)}%`, c3 + C3W, mry + 1);
                    const z2y = mry + Math.round(mRowH * 0.32);
                    ctx.globalAlpha = a * 0.78;
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(c3 + 7, z2y, C3W - 36, Math.round(mRowH * 0.36));
                    ctx.clip();
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'top';
                    ctx.font = `400 ${cl(mRowH * 0.27, 5.5, 7.5)}px Inter,sans-serif`;
                    ctx.fillStyle = INK;
                    ctx.fillText(`${taxon}  ·  ${transform}`, c3 + 7, z2y);
                    ctx.restore();
                    const z3y = mry + Math.round(mRowH * 0.66);
                    ctx.globalAlpha = a * 0.90;
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(c3 + 7, z3y, C3W - 36, Math.round(mRowH * 0.36));
                    ctx.clip();
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'top';
                    ctx.font = `500 ${cl(mRowH * 0.30, 6, 8.5)}px "JetBrains Mono",monospace`;
                    ctx.fillStyle = ACC;
                    ctx.fillText(learner ?? '', c3 + 7, z3y);
                    ctx.restore();
                    ctx.globalAlpha = 1;
                });
                ry2 += ENS_MEMBERS.length * (mRowH + mGap) + 10;
                ctx.globalAlpha = a * 0.18;
                ctx.beginPath();
                ctx.moveTo(c3, hairY);
                ctx.lineTo(c3 + C3W, hairY);
                ctx.strokeStyle = MID;
                ctx.lineWidth = 0.6;
                ctx.stroke();
                ctx.globalAlpha = 1;
                ctx.globalAlpha = a * 0.92;
                ctx.textAlign = 'right';
                ctx.textBaseline = 'bottom';
                ctx.font = `700 ${scoreFS}px "JetBrains Mono",monospace`;
                ctx.fillStyle = INK;
                ctx.fillText(BEST_AGG_SCORE.toFixed(3), c3 + C3W, cH - PY);
                ctx.globalAlpha = 1;
            }
            raf = requestAnimationFrame(draw);
        }
        draw();
        return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize); };
    }, []);
    return <canvas ref={ref} style={{ width: '100%', height: '100%', display: 'block' }}/>;
}
export function GlobalExplainCanvas() {
    const ref = useRef(null);
    useEffect(() => {
        const cv = ref.current;
        if (!cv)
            return;
        const dpr = Math.max(3, window.devicePixelRatio || 1);
        let raf, tick = 0;
        const resize = () => { cv.width = cv.offsetWidth * dpr; cv.height = cv.offsetHeight * dpr; };
        resize();
        window.addEventListener('resize', resize);
        const CYCLE = 1800;
        const sm = (x) => 1 / (1 + Math.exp(-7 * (x - 0.5)));
        const cl = (v, a, b) => Math.max(a, Math.min(b, v));
        const sl = (p, a, b) => sm(cl((p - a) / (b - a), 0, 1));
        const INK = '#0f172a';
        const MID = '#64748b';
        const DIM = '#94a3b8';
        const TRACK = '#e2e8f0';
        const BG = '#f8fafc';
        const METHODS = ['SHAP', 'LIME', 'Permutation', 'ALE'];
        const M_COL = ['#2563eb', '#0891b2', '#7c3aed', '#059669'];
        const M_START = [0.06, 0.30, 0.52, 0.72];
        const M_END = [0.26, 0.48, 0.68, 0.86];
        const PHY_S = ['#2563eb', '#059669', '#7c3aed', '#d97706', '#dc2626', '#db2777', '#475569', '#0891b2', '#94a3b8'];
        const PHY_F = ['#dbeafe', '#d1fae5', '#ede9fe', '#fef3c7', '#ffe4e6', '#fce7f3', '#f1f5f9', '#cffafe', '#f1f5f9'];
        const PHY_N = ['Bacteroidota', 'Firmicutes', 'Proteobacteria', 'Actinobacteriota', 'Verrucomicrobiota', 'Fusobacteriota', 'Spirochaetota', 'Cyanobacteria', 'Other'];
        const NET_NODES = [
            { label: 'o. Oscillospirales', phy: 1 },
            { label: 'f. Prevotellaceae', phy: 0 },
            { label: 'f. Bacteroidaceae', phy: 0 },
            { label: 'o. Bacteroidales', phy: 0 },
            { label: 'o. Lachnospirales', phy: 1 },
            { label: 'o. Bifidobacteriales', phy: 3 },
            { label: 's. Parabacteroides johnsonii', phy: 0 },
            { label: 's. Parabacteroides merdae', phy: 0 },
            { label: 's. Bacteroides vulgatus', phy: 0 },
            { label: 's. Slackia piriformis', phy: 3 },
            { label: 'p. Synergistota', phy: 8 },
            { label: 'p. Verrucomicrobiota', phy: 4 },
            { label: 's. Coprococcus eutactus', phy: 1 },
            { label: 'o. Monoglobales', phy: 1 },
            { label: 'f. Monoglobaceae', phy: 1 },
            { label: 'p. Fusobacteriota', phy: 5 },
            { label: 'p. Spirochaetota', phy: 6 },
            { label: 'p. Patescibacteria', phy: 8 },
            { label: 'g. Prevotella', phy: 0 },
            { label: 's. Parabacteroides distasonis', phy: 0 },
            { label: 'f. Ruminococcaceae', phy: 1 },
            { label: 's. Bacteroides coprocola', phy: 0 },
            { label: 'f. Burkholderiaceae', phy: 2 },
            { label: 'f. Acidaminococcaceae', phy: 1 },
            { label: 'f. Eubacteriaceae', phy: 1 },
            { label: 'f. Saccharimonadales', phy: 8 },
            { label: 'o. Thermomicrobiales', phy: 8 },
            { label: 's. Bacteroides uniformis', phy: 0 },
            { label: 'p. Acidobacteriota', phy: 8 },
            { label: 'p. Campilobacterota', phy: 8 },
            { label: 's. Paraprevotella clara', phy: 0 },
            { label: 's. Schaalia odontolytica', phy: 3 },
            { label: 'g. Bacteroides', phy: 0 },
            { label: 'g. UCG-002', phy: 1 },
            { label: 'p. Desulfobacterota', phy: 8 },
            { label: 'p. Elusimicrobiota', phy: 8 },
            { label: 'f. Verrucomicrobiaceae', phy: 4 },
            { label: 's. Ruminococcus lactaris', phy: 1 },
            { label: 'p. Chloroflexi', phy: 8 },
            { label: 'g. Clostridia UCG-014', phy: 1 },
            { label: 'f. RF39', phy: 1 },
            { label: 'o. RF39', phy: 1 },
            { label: 'c. Bacteroidia', phy: 0 },
            { label: 'f. Lachnospiraceae', phy: 1 },
            { label: 'g. CAG-352', phy: 1 },
            { label: 's. Prevotella stercorea', phy: 0 },
            { label: 's. Prevotella timonensis', phy: 0 },
            { label: 's. Streptococcus mutans', phy: 1 },
            { label: 'f. DTU014', phy: 1 },
            { label: 'g. Ruminococcus', phy: 1 },
            { label: 's. Bacteroides salyersiae', phy: 0 },
        ];
        const NP3 = [
            [-0.69617, +0.53504, -0.47862],
            [-0.90074, +0.19731, -0.38696],
            [-0.53865, +0.27855, -0.79515],
            [-0.54307, +0.82647, -0.14837],
            [-0.22724, +0.78103, -0.58168],
            [+0.05471, +0.93743, -0.34385],
            [+0.88877, +0.45540, -0.05200],
            [+0.73320, +0.58267, +0.35059],
            [+0.28676, +0.31705, +0.90402],
            [+0.35217, +0.66228, +0.66133],
            [+0.51743, -0.19128, -0.83407],
            [+0.77711, +0.16413, -0.60758],
            [-0.08964, +0.74360, +0.66259],
            [-0.71178, +0.45655, +0.53379],
            [-0.92205, +0.37158, +0.10840],
            [+0.56228, +0.77435, -0.29019],
            [+0.42659, +0.60723, -0.67029],
            [+0.39008, +0.19065, -0.90083],
            [-0.52753, +0.79913, +0.28828],
            [+0.40985, -0.15443, +0.89899],
            [-0.13535, +0.98559, +0.10141],
            [+0.69756, +0.21668, +0.68298],
            [-0.98164, -0.19016, -0.01467],
            [-0.78173, -0.61579, -0.09851],
            [+0.16035, -0.98315, +0.08776],
            [+0.03011, -0.86994, +0.49224],
            [+0.55601, -0.83098, -0.01787],
            [-0.06299, -0.00110, +0.99801],
            [-0.02686, +0.40678, -0.91313],
            [+0.79403, -0.41979, -0.43965],
            [-0.30373, +0.39236, +0.86822],
            [+0.97627, -0.00057, -0.21654],
            [-0.37119, -0.90380, -0.21297],
            [-0.38643, -0.68801, -0.61426],
            [+0.43915, -0.66010, -0.60943],
            [-0.03388, -0.08482, -0.99582],
            [+0.10340, -0.92106, -0.37544],
            [+0.96884, +0.06407, +0.23924],
            [+0.04695, -0.49847, -0.86563],
            [+0.34992, +0.90784, +0.23105],
            [-0.80585, -0.44933, +0.38562],
            [-0.81771, -0.31508, -0.48175],
            [-0.50562, -0.19716, -0.83993],
            [-0.88174, +0.04486, +0.46959],
            [+0.48240, -0.64357, +0.59424],
            [+0.03956, -0.48223, +0.87515],
            [-0.51441, -0.10853, +0.85065],
            [-0.43325, -0.56118, +0.70524],
            [+0.86360, -0.49431, +0.09928],
            [-0.42728, -0.86794, +0.25322],
            [+0.77573, -0.27482, +0.56809],
        ];
        const NET_EDGES = [
            [0, 1, 0.0345],
            [0, 2, 0.0322],
            [3, 1, 0.0310],
            [4, 5, 0.0286],
            [6, 7, 0.0269],
            [8, 9, 0.0258],
            [3, 2, 0.0252],
            [10, 11, 0.0250],
            [8, 12, 0.0244],
            [0, 3, 0.0242],
            [13, 14, 0.0236],
            [15, 16, 0.0234],
            [10, 17, 0.0232],
            [7, 8, 0.0229],
            [3, 18, 0.0226],
            [8, 19, 0.0214],
            [3, 20, 0.0214],
            [8, 21, 0.0186],
            [22, 23, 0.0184],
            [24, 25, 0.0183],
            [24, 26, 0.0180],
            [8, 27, 0.0175],
            [10, 28, 0.0175],
            [10, 29, 0.0172],
            [8, 30, 0.0167],
            [15, 11, 0.0163],
            [6, 31, 0.0160],
            [32, 33, 0.0160],
            [10, 34, 0.0159],
            [24, 15, 0.0157],
            [10, 35, 0.0157],
            [24, 36, 0.0155],
            [6, 37, 0.0153],
            [32, 8, 0.0150],
            [16, 38, 0.0148],
            [8, 39, 0.0146],
            [22, 40, 0.0141],
            [22, 41, 0.0141],
            [3, 4, 0.0140],
            [2, 42, 0.0136],
            [14, 22, 0.0136],
            [3, 43, 0.0132],
            [8, 44, 0.0132],
            [8, 45, 0.0132],
            [46, 47, 0.0130],
            [8, 46, 0.0129],
            [48, 40, 0.0129],
            [32, 49, 0.0127],
            [3, 5, 0.0126],
            [8, 50, 0.0126],
        ];
        const NET_MAX = 0.0345, NET_MIN = 0.0126;
        const NET_DEG = new Array(NET_NODES.length).fill(0);
        NET_EDGES.forEach(([a, b]) => { NET_DEG[a]++; NET_DEG[b]++; });
        const MAX_DEG = Math.max(...NET_DEG);
        let rotX = 0.30, rotY = 0.45, vRotX = 0.0, vRotY = 0.0;
        let isDrag = false, dragX = 0, dragY = 0, startDragX = 0, startDragY = 0, didDrag = false;
        let isPaused = false, isVisible = true;
        let pillCache = [], pillCacheRX = NaN, pillCacheRY = NaN;
        cv.style.cursor = 'grab';
        cv.style.willChange = 'contents';
        const onMD = (e) => { isDrag = true; didDrag = false; startDragX = e.clientX; startDragY = e.clientY; dragX = e.clientX; dragY = e.clientY; vRotX = 0; vRotY = 0; cv.style.cursor = 'grabbing'; };
        const onMM = (e) => { if (!isDrag)
            return; if (Math.hypot(e.clientX - startDragX, e.clientY - startDragY) > 5)
            didDrag = true; vRotY = (e.clientX - dragX) * 0.0055; vRotX = (e.clientY - dragY) * 0.0055; rotY += (e.clientX - dragX) * 0.0055; rotX += (e.clientY - dragY) * 0.0055; dragX = e.clientX; dragY = e.clientY; };
        const onMU = () => { isDrag = false; cv.style.cursor = isPaused ? 'default' : 'grab'; };
        const onCK = () => { if (didDrag)
            return; isPaused = !isPaused; cv.style.cursor = isPaused ? 'default' : 'grab'; if (!isPaused && isVisible)
            raf = requestAnimationFrame(draw); };
        const onTD = (e) => { e.preventDefault(); isDrag = true; didDrag = false; startDragX = e.touches[0].clientX; startDragY = e.touches[0].clientY; dragX = e.touches[0].clientX; dragY = e.touches[0].clientY; vRotX = 0; vRotY = 0; };
        const onTM = (e) => { e.preventDefault(); if (!isDrag)
            return; didDrag = true; vRotY = (e.touches[0].clientX - dragX) * 0.0055; vRotX = (e.touches[0].clientY - dragY) * 0.0055; rotY += (e.touches[0].clientX - dragX) * 0.0055; rotX += (e.touches[0].clientY - dragY) * 0.0055; dragX = e.touches[0].clientX; dragY = e.touches[0].clientY; };
        const onTE = () => { isDrag = false; };
        cv.addEventListener('mousedown', onMD);
        window.addEventListener('mousemove', onMM);
        window.addEventListener('mouseup', onMU);
        cv.addEventListener('click', onCK);
        cv.addEventListener('touchstart', onTD, { passive: false });
        cv.addEventListener('touchmove', onTM, { passive: false });
        cv.addEventListener('touchend', onTE);
        const observer = new IntersectionObserver(entries => {
            const was = isVisible;
            isVisible = entries[0].isIntersecting;
            if (isVisible && !was && !isPaused)
                raf = requestAnimationFrame(draw);
        }, { threshold: 0.1 });
        observer.observe(cv);
        const GFEATS = [
            { name: 'g. Prevotella', vals: [0.124, 0.118, 0.131, 0.109] },
            { name: 'g. Bacteroides', vals: [0.097, 0.089, 0.102, 0.094] },
            { name: 'g. Lachnospira', vals: [0.083, 0.091, 0.079, 0.086] },
            { name: 'g. Faecalibacterium', vals: [0.071, 0.063, 0.068, 0.075] },
            { name: 'g. Ruminococcus', vals: [0.062, 0.055, 0.058, 0.064] },
            { name: 'g. Blautia', vals: [0.055, 0.061, 0.052, 0.049] },
            { name: 'g. Alistipes', vals: [0.048, 0.044, 0.051, 0.046] },
            { name: 'g. Dorea', vals: [0.041, 0.038, 0.045, 0.039] },
        ];
        const MAX_IMP = 0.131;
        GFEATS.forEach(f => { f.cons = f.vals.reduce((a, b) => a + b, 0) / 4; });
        function draw() {
            tick++;
            const ctx = cv.getContext('2d');
            const W = cv.offsetWidth, H = cv.offsetHeight;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = BG;
            ctx.fillRect(0, 0, W, H);
            const rawP = (tick % CYCLE) / CYCLE;
            const phase = Math.min(rawP < 0.55 ? rawP * (0.970 / 0.55) : 0.970, 0.970);
            const masterA = 1 - sm(cl((rawP - 0.930) / 0.040, 0, 1));
            const PX = 18, PY = 14;
            const cW = W - PX * 2;
            const NF = GFEATS.length;
            const TOP = PY + 18, BOT = H - PY - 6, contH = BOT - TOP;
            const fH = cl(contH / NF, 14, 38);
            const mTop = TOP + cl((contH - NF * fH) * 0.3, 0, 24);
            const BARS_W = Math.floor(cW * 0.50);
            const NET_GAP = 14;
            const NET_X = PX + BARS_W + NET_GAP;
            const NET_W = cW - BARS_W - NET_GAP;
            const LBLW = cl(BARS_W * 0.34, 70, 108);
            const CONSW = 28;
            const BARX = PX + 16 + 4 + LBLW + 8;
            const BARTW = BARS_W - 16 - 4 - LBLW - 8 - 4 - CONSW;
            const MB_H = cl(fH * 0.13, 1.5, 3.2);
            const MB_G = cl(fH * 0.04, 0.5, 1.0);
            const MB_TOT = 4 * MB_H + 3 * MB_G;
            const lblA = sl(phase, 0.01, 0.07) * masterA;
            ctx.globalAlpha = lblA * 0.32;
            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.font = `400 7px "JetBrains Mono",monospace`;
            ctx.fillStyle = MID;
            ctx.fillText('multi-method importance  ·  SHAP · LIME · Perm · ALE', PX, PY);
            ctx.globalAlpha = 1;
            const legA = sl(phase, 0.02, 0.08) * masterA;
            if (legA > 0.01) {
                ctx.font = `400 6.5px "JetBrains Mono",monospace`;
                const li = METHODS.map((m, mi) => ({ m, col: M_COL[mi], w: ctx.measureText(m).width + 6 + 10 }));
                const lt = li.reduce((a, b) => a + b.w, 0) - 10;
                let lx = PX + BARS_W - lt;
                li.forEach(({ m, col, w }) => {
                    ctx.globalAlpha = legA * 0.85;
                    ctx.beginPath();
                    ctx.arc(lx + 3, PY + 3.5, 2.5, 0, 2 * Math.PI);
                    ctx.fillStyle = col;
                    ctx.fill();
                    ctx.globalAlpha = legA * 0.62;
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'top';
                    ctx.font = `400 6.5px "JetBrains Mono",monospace`;
                    ctx.fillStyle = MID;
                    ctx.fillText(m, lx + 8, PY);
                    ctx.globalAlpha = 1;
                    lx += w;
                });
            }
            GFEATS.forEach((f, fi) => {
                const fT = sl(phase, 0.03 + fi * 0.022, 0.07 + fi * 0.022) * masterA;
                if (fT < 0.01)
                    return;
                const fy = mTop + fi * fH + fH * 0.5;
                ctx.globalAlpha = fT * 0.22;
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                ctx.font = `400 6px "JetBrains Mono",monospace`;
                ctx.fillStyle = DIM;
                ctx.fillText(`${fi + 1}`, PX + 16, fy);
                ctx.globalAlpha = 1;
                ctx.globalAlpha = fT * 0.78;
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                ctx.font = `400 ${cl(fH * 0.38, 6, 9)}px Inter,sans-serif`;
                ctx.fillStyle = INK;
                ctx.save();
                ctx.beginPath();
                ctx.rect(PX + 20, fy - fH * 0.5, LBLW - 2, fH);
                ctx.clip();
                ctx.fillText(f.name, PX + 20 + LBLW - 2, fy);
                ctx.restore();
                ctx.globalAlpha = 1;
                ctx.globalAlpha = fT * 0.06;
                ctx.fillStyle = TRACK;
                ctx.fillRect(BARX, fy - MB_TOT / 2 - 1, BARTW, MB_TOT + 2);
                ctx.globalAlpha = 1;
                f.vals.forEach((imp, mi) => {
                    const fill = cl(sl(phase, M_START[mi] + fi * 0.004, M_END[mi] + fi * 0.004), 0, 1);
                    const by = fy - MB_TOT / 2 + mi * (MB_H + MB_G), fw = BARTW * (imp / MAX_IMP) * fill;
                    if (fw > 0.4) {
                        ctx.globalAlpha = fT * 0.80;
                        ctx.fillStyle = M_COL[mi];
                        ctx.fillRect(BARX, by, fw, MB_H);
                        ctx.globalAlpha = 1;
                    }
                });
                const consT = sl(phase, 0.86 + fi * 0.006, 0.91 + fi * 0.006) * masterA;
                if (consT > 0.01) {
                    ctx.globalAlpha = consT * 0.52;
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    ctx.font = `500 ${cl(fH * 0.32, 5, 7)}px "JetBrains Mono",monospace`;
                    ctx.fillStyle = MID;
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(BARX + BARTW + 4, fy - fH * 0.5, CONSW, fH);
                    ctx.clip();
                    ctx.fillText((f.cons).toFixed(3), BARX + BARTW + 4, fy);
                    ctx.restore();
                    ctx.globalAlpha = 1;
                }
            });
            const colHdrT = sl(phase, 0.84, 0.90) * masterA;
            if (colHdrT > 0.01) {
                ctx.globalAlpha = colHdrT * 0.28;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.font = `400 6px "JetBrains Mono",monospace`;
                ctx.fillStyle = DIM;
                ctx.fillText('avg', BARX + BARTW + 4, PY);
                ctx.globalAlpha = 1;
            }
            const sepA = sl(phase, 0.01, 0.07) * masterA;
            if (sepA > 0.01) {
                ctx.globalAlpha = sepA * 0.08;
                ctx.strokeStyle = DIM;
                ctx.lineWidth = 0.75;
                ctx.beginPath();
                ctx.moveTo(NET_X - 7, TOP);
                ctx.lineTo(NET_X - 7, BOT);
                ctx.stroke();
                ctx.globalAlpha = 1;
            }
            const netLblA = sl(phase, 0.01, 0.07) * masterA;
            if (netLblA > 0.01) {
                ctx.globalAlpha = netLblA * 0.32;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                ctx.font = `400 7px "JetBrains Mono",monospace`;
                ctx.fillStyle = MID;
                ctx.fillText(isPaused
                    ? 'SHAP interaction network  ·  click to resume'
                    : 'SHAP interaction network  ·  drag · click to pause', NET_X, PY);
                ctx.globalAlpha = 1;
            }
            if (!isDrag && !isPaused) {
                rotX += vRotX;
                rotY += vRotY + 0.0018;
                vRotX *= 0.92;
                vRotY *= 0.92;
            }
            else if (!isDrag) {
                vRotX *= 0.94;
                vRotY *= 0.94;
            }
            const ncx = NET_X + NET_W * 0.50;
            const ncy = TOP + contH * 0.50;
            const S = cl(Math.min(NET_W * 0.42, contH * 0.44), 52, 106);
            const FOC = 3.0;
            const cX = Math.cos(rotX), sX = Math.sin(rotX), cY = Math.cos(rotY), sY = Math.sin(rotY);
            const rot3 = (p) => {
                const [x, y, z] = p;
                const x1 = x * cY + z * sY, z1 = -x * sY + z * cY;
                return [x1, y * cX - z1 * sX, y * sX + z1 * cX];
            };
            const proj = (p) => {
                const [rx, ry, rz] = rot3(p);
                const sc = FOC / (FOC + rz);
                return { sx: ncx + rx * sc * S, sy: ncy + ry * sc * S, rz, sc };
            };
            const PRJ = NP3.map(proj);
            const edgeOrd = [...Array(NET_EDGES.length)].map((_, i) => i)
                .sort((a, b) => (PRJ[NET_EDGES[a][0]].rz + PRJ[NET_EDGES[a][1]].rz)
                - (PRJ[NET_EDGES[b][0]].rz + PRJ[NET_EDGES[b][1]].rz));
            const nodeOrd = [...Array(NET_NODES.length)].map((_, i) => i).filter(i => i !== 8)
                .sort((a, b) => PRJ[b].rz - PRJ[a].rz);
            edgeOrd.forEach(ei => {
                const [ai, bi, str] = NET_EDGES[ei];
                const norm = (str - NET_MIN) / (NET_MAX - NET_MIN);
                const eT = sl(phase, 0.30 + ei * 0.006, 0.40 + ei * 0.006) * masterA;
                if (eT < 0.005)
                    return;
                const { sx: ax, sy: ay, rz: arz } = PRJ[ai], { sx: bx, sy: by, rz: brz } = PRJ[bi];
                const dep = cl(((arz + brz) / 2 + 2) / 4, 0.12, 1.0);
                const chord = Math.hypot(bx - ax, by - ay) + 0.001;
                const gg = ctx.createLinearGradient(ax, ay, bx, by);
                gg.addColorStop(0, PHY_S[NET_NODES[ai].phy]);
                gg.addColorStop(1, PHY_S[NET_NODES[bi].phy]);
                ctx.globalAlpha = (0.04 + norm * 0.06) * eT * dep;
                ctx.strokeStyle = gg;
                ctx.lineWidth = cl(4 + norm * 8, 4, 12);
                ctx.lineCap = 'round';
                ctx.beginPath();
                ctx.moveTo(ax, ay);
                ctx.lineTo(bx, by);
                ctx.stroke();
                const lg = ctx.createLinearGradient(ax, ay, bx, by);
                lg.addColorStop(0, PHY_S[NET_NODES[ai].phy]);
                lg.addColorStop(1, PHY_S[NET_NODES[bi].phy]);
                ctx.globalAlpha = (0.22 + norm * 0.52 + dep * 0.14) * eT;
                ctx.strokeStyle = lg;
                ctx.lineWidth = cl(0.6 + norm * 2.6 + dep * 0.4, 0.6, 3.4);
                ctx.setLineDash([chord, chord]);
                ctx.lineDashOffset = chord * (1 - eT);
                ctx.beginPath();
                ctx.moveTo(ax, ay);
                ctx.lineTo(bx, by);
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.globalAlpha = 1;
            });
            nodeOrd.forEach(i => {
                const n = NET_NODES[i];
                const nT = sl(phase, 0.04 + i * 0.006, 0.14 + i * 0.006) * masterA;
                if (nT < 0.005)
                    return;
                const { sx: x, sy: y, rz, sc } = PRJ[i], deg = NET_DEG[i];
                const dep = cl((rz + 2) / 4, 0.28, 1.0);
                const nr = cl((2.8 + (deg / MAX_DEG) * 4.6) * cl(sc, 0.58, 1.42), 2.2, 9.2);
                const gR = nr + 6 + deg * 1.3;
                const ng = ctx.createRadialGradient(x, y, nr * 0.5, x, y, gR);
                ng.addColorStop(0, PHY_S[n.phy] + '42');
                ng.addColorStop(1, PHY_S[n.phy] + '00');
                ctx.globalAlpha = nT * dep * (0.36 + deg * 0.07);
                ctx.fillStyle = ng;
                ctx.beginPath();
                ctx.arc(x, y, gR, 0, 2 * Math.PI);
                ctx.fill();
                ctx.globalAlpha = nT * dep;
                const rfg = ctx.createRadialGradient(x - nr * 0.38, y - nr * 0.38, 0, x, y, nr);
                rfg.addColorStop(0, '#ffffff');
                rfg.addColorStop(0.40, PHY_F[n.phy]);
                rfg.addColorStop(1, PHY_S[n.phy] + 'cc');
                ctx.beginPath();
                ctx.arc(x, y, nr, 0, 2 * Math.PI);
                ctx.fillStyle = rfg;
                ctx.fill();
                ctx.strokeStyle = PHY_S[n.phy];
                ctx.lineWidth = cl(0.8 + deg * 0.14 * sc, 0.7, 2.0);
                ctx.stroke();
                ctx.globalAlpha = 1;
            });
            ctx.font = '500 7px Inter,sans-serif';
            const pHL = 11, pPX = 6;
            const velMag = Math.abs(vRotX) + Math.abs(vRotY);
            const settled = !isDrag && velMag < 0.0010;
            const sameView = Math.abs(rotX - pillCacheRX) < 0.002 && Math.abs(rotY - pillCacheRY) < 0.002;
            const rawPills = [];
            nodeOrd.forEach(i => {
                const n = NET_NODES[i];
                const lTraw = sl(phase, 0.45 + i * 0.006, 0.56 + i * 0.006) * masterA;
                if (lTraw < 0.005)
                    return;
                const { sx: x, sy: y, rz, sc } = PRJ[i];
                const dep = cl((rz + 2) / 4, 0.28, 1.0);
                const lT = lTraw * dep;
                if (lT < 0.06)
                    return;
                const nr = cl((2.8 + (NET_DEG[i] / MAX_DEG) * 4.6) * cl(sc, 0.58, 1.42), 2.2, 9.2);
                const tw = ctx.measureText(n.label).width, pW = tw + pPX * 2 + 3;
                const ang = Math.atan2(y - PRJ[0].sy, x - PRJ[0].sx);
                const ca = Math.cos(ang), sa = Math.sin(ang), d = nr + 10;
                const cx2 = x + ca * d, cy2 = y + sa * d;
                const px = ca > 0.18 ? cx2 : ca < -0.18 ? cx2 - pW : cx2 - pW / 2;
                const py = sa > 0.18 ? cy2 : sa < -0.18 ? cy2 - pHL : cy2 - pHL / 2;
                rawPills.push({ x: px, y: py, w: pW, h: pHL, ni: i, lT, nr });
            });
            let pills;
            if (settled && sameView && pillCache.length === rawPills.length) {
                pills = pillCache.map(c => { const r = rawPills.find(p => p.ni === c.ni); return { ...c, lT: r?.lT ?? c.lT, nr: r?.nr ?? c.nr }; });
            }
            else {
                const nIter = settled ? 24 : 5;
                for (let it = 0; it < nIter; it++) {
                    for (let a = 0; a < rawPills.length; a++) {
                        for (let b = a + 1; b < rawPills.length; b++) {
                            const A = rawPills[a], B = rawPills[b];
                            const acx = A.x + A.w / 2, acy = A.y + A.h / 2, bcx = B.x + B.w / 2, bcy = B.y + B.h / 2;
                            const ox = Math.max(0, (A.w + B.w) / 2 + 5 - Math.abs(acx - bcx));
                            const oy = Math.max(0, (A.h + B.h) / 2 + 4 - Math.abs(acy - bcy));
                            if (ox > 0 && oy > 0) {
                                if (ox < oy) {
                                    const dd = ox / 2 + 0.5;
                                    if (acx < bcx) {
                                        A.x -= dd;
                                        B.x += dd;
                                    }
                                    else {
                                        A.x += dd;
                                        B.x -= dd;
                                    }
                                }
                                else {
                                    const dd = oy / 2 + 0.5;
                                    if (acy < bcy) {
                                        A.y -= dd;
                                        B.y += dd;
                                    }
                                    else {
                                        A.y += dd;
                                        B.y -= dd;
                                    }
                                }
                            }
                        }
                        rawPills[a].x = cl(rawPills[a].x, NET_X + 3, NET_X + NET_W - rawPills[a].w - 3);
                        rawPills[a].y = cl(rawPills[a].y, TOP + 16, BOT - rawPills[a].h - 11);
                    }
                }
                if (settled) {
                    pillCache = rawPills.map(p => ({ ...p }));
                    pillCacheRX = rotX;
                    pillCacheRY = rotY;
                }
                pills = rawPills;
            }
            ctx.save();
            ctx.beginPath();
            ctx.rect(NET_X + 1, TOP, NET_W - 2, contH);
            ctx.clip();
            const rr2 = (rx, ry, rw, rh, r) => {
                ctx.beginPath();
                ctx.moveTo(rx + r, ry);
                ctx.lineTo(rx + rw - r, ry);
                ctx.arcTo(rx + rw, ry, rx + rw, ry + r, r);
                ctx.lineTo(rx + rw, ry + rh - r);
                ctx.arcTo(rx + rw, ry + rh, rx + rw - r, ry + rh, r);
                ctx.lineTo(rx + r, ry + rh);
                ctx.arcTo(rx, ry + rh, rx, ry + rh - r, r);
                ctx.lineTo(rx, ry + r);
                ctx.arcTo(rx, ry, rx + r, ry, r);
                ctx.closePath();
            };
            pills.forEach(p => {
                const { ni, lT, nr } = p, n = NET_NODES[ni];
                const { sx: nx, sy: ny } = PRJ[ni];
                const pcx = p.x + p.w / 2, pcy = p.y + p.h / 2;
                const la = Math.atan2(pcy - ny, pcx - nx);
                ctx.globalAlpha = lT * 0.40;
                ctx.strokeStyle = PHY_S[n.phy];
                ctx.lineWidth = 0.65;
                ctx.lineCap = 'round';
                ctx.setLineDash([1.4, 2.4]);
                ctx.beginPath();
                ctx.moveTo(nx + Math.cos(la) * nr, ny + Math.sin(la) * nr);
                ctx.lineTo(pcx - Math.cos(la) * (p.w / 2 + 1), pcy - Math.sin(la) * (p.h / 2 + 1));
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.globalAlpha = lT * 0.07;
                ctx.fillStyle = '#1e293b';
                rr2(p.x + 0.5, p.y + 1, p.w, p.h, 3.5);
                ctx.fill();
                ctx.globalAlpha = lT * 0.97;
                ctx.fillStyle = 'rgba(248,250,252,0.97)';
                ctx.strokeStyle = PHY_S[n.phy] + '55';
                ctx.lineWidth = 0.7;
                rr2(p.x, p.y, p.w, p.h, 3.5);
                ctx.fill();
                ctx.stroke();
                ctx.globalAlpha = lT * 0.88;
                ctx.fillStyle = PHY_S[n.phy];
                ctx.beginPath();
                ctx.moveTo(p.x + 3.5, p.y);
                ctx.lineTo(p.x + 3, p.y);
                ctx.arcTo(p.x, p.y, p.x, p.y + 3.5, 3.5);
                ctx.lineTo(p.x, p.y + p.h - 3.5);
                ctx.arcTo(p.x, p.y + p.h, p.x + 3.5, p.y + p.h, 3.5);
                ctx.lineTo(p.x + 3, p.y + p.h);
                ctx.lineTo(p.x + 3, p.y);
                ctx.closePath();
                ctx.fill();
                ctx.globalAlpha = lT;
                ctx.fillStyle = '#1e293b';
                ctx.textAlign = 'left';
                ctx.textBaseline = 'middle';
                ctx.font = '500 7px Inter,sans-serif';
                ctx.fillText(n.label, p.x + 7, p.y + p.h / 2);
                ctx.globalAlpha = 1;
            });
            ctx.restore();
            const hubT = sl(phase, 0.04 + 8 * 0.006, 0.14 + 8 * 0.006) * masterA;
            if (hubT > 0.01) {
                const hubR = cl(3.5 + (NET_DEG[8] / MAX_DEG) * 6.0, 7.0, 10.0);
                const { sx: hx, sy: hy, rz: hubRz } = PRJ[8];
                const hubDep = cl((hubRz + 2) / 4, 0.22, 1.0);
                const pulse = 0.5 + 0.5 * Math.sin(tick * 0.030);
                ctx.globalAlpha = hubT * hubDep * (0.05 + pulse * 0.07);
                ctx.beginPath();
                ctx.arc(hx, hy, hubR + 5 + pulse * 4, 0, 2 * Math.PI);
                ctx.strokeStyle = '#2563eb';
                ctx.lineWidth = 1.4;
                ctx.stroke();
                ctx.globalAlpha = 1;
                const hg = ctx.createRadialGradient(hx, hy, 0, hx, hy, hubR + 22);
                hg.addColorStop(0, `rgba(37,99,235,${0.34 * hubT * hubDep})`);
                hg.addColorStop(0.5, `rgba(37,99,235,${0.06 * hubT * hubDep})`);
                hg.addColorStop(1, 'rgba(37,99,235,0)');
                ctx.fillStyle = hg;
                ctx.beginPath();
                ctx.arc(hx, hy, hubR + 22, 0, 2 * Math.PI);
                ctx.fill();
                ctx.globalAlpha = hubT * hubDep;
                const hcg = ctx.createRadialGradient(hx - hubR * 0.40, hy - hubR * 0.40, 0, hx, hy, hubR);
                hcg.addColorStop(0, '#bfdbfe');
                hcg.addColorStop(0.52, '#3b82f6');
                hcg.addColorStop(1, '#1e3a8a');
                ctx.beginPath();
                ctx.arc(hx, hy, hubR, 0, 2 * Math.PI);
                ctx.fillStyle = hcg;
                ctx.fill();
                ctx.strokeStyle = '#1d4ed8';
                ctx.lineWidth = 1.8;
                ctx.stroke();
                ctx.globalAlpha = 1;
                const hlA = sl(phase, 0.08, 0.17) * masterA * hubDep;
                if (hlA > 0.01) {
                    ctx.save();
                    ctx.font = '600 8.5px Inter,sans-serif';
                    const htw = ctx.measureText('s. Bacteroides vulgatus').width;
                    const hpW = htw + 20, hpH = 14, hpX = hx - hpW / 2, hpY = hy - hubR - 28;
                    ctx.globalAlpha = hlA * 0.10;
                    ctx.fillStyle = '#1e293b';
                    ctx.beginPath();
                    ctx.moveTo(hpX + 4.5, hpY + 2);
                    ctx.lineTo(hpX + hpW - 3.5, hpY + 2);
                    ctx.arcTo(hpX + hpW + 0.5, hpY + 2, hpX + hpW + 0.5, hpY + 6, 4);
                    ctx.lineTo(hpX + hpW + 0.5, hpY + hpH - 2);
                    ctx.arcTo(hpX + hpW + 0.5, hpY + hpH + 2, hpX + hpW - 3.5, hpY + hpH + 2, 4);
                    ctx.lineTo(hpX + 4.5, hpY + hpH + 2);
                    ctx.arcTo(hpX + 0.5, hpY + hpH + 2, hpX + 0.5, hpY + hpH - 2, 4);
                    ctx.lineTo(hpX + 0.5, hpY + 6);
                    ctx.arcTo(hpX + 0.5, hpY + 2, hpX + 4.5, hpY + 2, 4);
                    ctx.closePath();
                    ctx.fill();
                    ctx.globalAlpha = hlA;
                    ctx.fillStyle = 'rgba(239,246,255,0.98)';
                    ctx.strokeStyle = '#3b82f6';
                    ctx.lineWidth = 1.2;
                    ctx.beginPath();
                    ctx.moveTo(hpX + 4, hpY);
                    ctx.lineTo(hpX + hpW - 4, hpY);
                    ctx.arcTo(hpX + hpW, hpY, hpX + hpW, hpY + 4, 4);
                    ctx.lineTo(hpX + hpW, hpY + hpH - 4);
                    ctx.arcTo(hpX + hpW, hpY + hpH, hpX + hpW - 4, hpY + hpH, 4);
                    ctx.lineTo(hpX + 4, hpY + hpH);
                    ctx.arcTo(hpX, hpY + hpH, hpX, hpY + hpH - 4, 4);
                    ctx.lineTo(hpX, hpY + 4);
                    ctx.arcTo(hpX, hpY, hpX + 4, hpY, 4);
                    ctx.closePath();
                    ctx.fill();
                    ctx.stroke();
                    ctx.globalAlpha = hlA * 0.35;
                    ctx.strokeStyle = '#3b82f6';
                    ctx.lineWidth = 0.8;
                    ctx.setLineDash([1.5, 2]);
                    ctx.beginPath();
                    ctx.moveTo(hx, hpY + hpH);
                    ctx.lineTo(hx, hy - hubR - 2);
                    ctx.stroke();
                    ctx.setLineDash([]);
                    ctx.globalAlpha = hlA;
                    ctx.fillStyle = '#1e3a8a';
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    ctx.font = '600 8.5px Inter,sans-serif';
                    ctx.fillText('s. Bacteroides vulgatus', hx, hpY + hpH / 2);
                    ctx.globalAlpha = hlA * 0.42;
                    ctx.fillStyle = MID;
                    ctx.textBaseline = 'top';
                    ctx.font = '400 5.5px "JetBrains Mono",monospace';
                    ctx.fillText(`hub · deg ${NET_DEG[8]}`, hx, hy + hubR + 5);
                    ctx.restore();
                    ctx.globalAlpha = 1;
                }
            }
            const legA2 = sl(phase, 0.40, 0.52) * masterA;
            if (legA2 > 0.01) {
                const usedPhy = new Set(NET_NODES.map(n => n.phy));
                const legendPhy = PHY_N.map((pn, pi) => ({ pn, pi })).filter(({ pi }) => usedPhy.has(pi)).reverse();
                let yl = BOT - 4;
                legendPhy.forEach(({ pn, pi }) => {
                    ctx.globalAlpha = legA2 * 0.80;
                    ctx.beginPath();
                    ctx.arc(NET_X + 6, yl, 3, 0, 2 * Math.PI);
                    ctx.fillStyle = PHY_S[pi];
                    ctx.fill();
                    ctx.globalAlpha = legA2 * 0.55;
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    ctx.font = `500 6px Inter,sans-serif`;
                    ctx.fillStyle = INK;
                    ctx.fillText(pn, NET_X + 13, yl);
                    ctx.globalAlpha = 1;
                    yl -= 11;
                });
            }
            const nlegA = sl(phase, 0.42, 0.54) * masterA;
            if (nlegA > 0.01) {
                const SZ = [{ r: 2.2, lb: '1' }, { r: 4.8, lb: `${Math.round(MAX_DEG / 2)}` }, { r: 7.4, lb: `${MAX_DEG}` }];
                ctx.font = `500 6px Inter,sans-serif`;
                const totalW = SZ.reduce((acc, s, i) => acc + s.r * 2 + (i < SZ.length - 1 ? 18 : 0), 0);
                let nx = NET_X + NET_W / 2 - totalW / 2 + SZ[0].r;
                const ny = BOT - 5;
                ctx.globalAlpha = nlegA * 0.45;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'bottom';
                ctx.font = `400 5.5px Inter,sans-serif`;
                ctx.fillStyle = INK;
                ctx.fillText('node size = no. of connections', NET_X + NET_W / 2, ny - 10);
                ctx.globalAlpha = 1;
                SZ.forEach(({ r, lb }, i) => {
                    const cg = ctx.createRadialGradient(nx - r * 0.35, ny - r * 0.35, 0, nx, ny, r);
                    cg.addColorStop(0, '#ffffff');
                    cg.addColorStop(0.45, '#cbd5e1');
                    cg.addColorStop(1, '#64748b');
                    ctx.globalAlpha = nlegA * 0.85;
                    ctx.beginPath();
                    ctx.arc(nx, ny, r, 0, 2 * Math.PI);
                    ctx.fillStyle = cg;
                    ctx.fill();
                    ctx.strokeStyle = '#94a3b8';
                    ctx.lineWidth = 0.7;
                    ctx.stroke();
                    ctx.globalAlpha = nlegA * 0.55;
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'top';
                    ctx.font = `500 5.5px Inter,sans-serif`;
                    ctx.fillStyle = INK;
                    ctx.fillText(lb, nx, ny + r + 2);
                    ctx.globalAlpha = 1;
                    if (i < SZ.length - 1)
                        nx += r + 18 + SZ[i + 1].r;
                });
            }
            const slegA = sl(phase, 0.42, 0.54) * masterA;
            if (slegA > 0.01) {
                const lx0 = NET_X + NET_W - 58, ly0 = BOT - 4;
                const STP = [{ lb: 'strong', n: 1.0 }, { lb: 'moderate', n: 0.5 }, { lb: 'weak', n: 0.0 }];
                let slY = ly0;
                STP.forEach(({ lb, n }) => {
                    ctx.globalAlpha = slegA * (0.28 + n * 0.60);
                    ctx.strokeStyle = MID;
                    ctx.lineWidth = cl(0.6 + n * 2.8, 0.6, 3.4);
                    ctx.lineCap = 'round';
                    ctx.beginPath();
                    ctx.moveTo(lx0, slY);
                    ctx.lineTo(lx0 + 22, slY);
                    ctx.stroke();
                    ctx.globalAlpha = slegA * 0.55;
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    ctx.font = `500 6px Inter,sans-serif`;
                    ctx.fillStyle = INK;
                    ctx.fillText(lb, lx0 + 26, slY);
                    ctx.globalAlpha = 1;
                    slY -= 11;
                });
            }
            if (isVisible && (!isPaused || isDrag))
                raf = requestAnimationFrame(draw);
        }
        draw();
        return () => {
            cancelAnimationFrame(raf);
            observer.disconnect();
            window.removeEventListener('resize', resize);
            window.removeEventListener('mousemove', onMM);
            window.removeEventListener('mouseup', onMU);
            cv.removeEventListener('click', onCK);
        };
    }, []);
    return <canvas ref={ref} style={{ width: '100%', height: '100%', display: 'block', cursor: 'grab' }}/>;
}
export function LocalExplainCanvas() {
    const ref = useRef(null);
    useEffect(() => {
        const cv = ref.current;
        if (!cv)
            return;
        const dpr = Math.max(3, window.devicePixelRatio || 1);
        let raf, tick = 0;
        const resize = () => { cv.width = cv.offsetWidth * dpr; cv.height = cv.offsetHeight * dpr; };
        resize();
        window.addEventListener('resize', resize);
        const CYCLE = 1600;
        const sm = (x) => 1 / (1 + Math.exp(-7 * (x - 0.5)));
        const cl = (v, a, b) => Math.max(a, Math.min(b, v));
        const sl = (p, a, b) => sm(cl((p - a) / (b - a), 0, 1));
        const INK = '#2C3E50';
        const C_POS = '#2563eb';
        const C_NEG = '#94a3b8';
        const MID = '#64748b';
        const BG = '#f8fafc';
        const LFEATS = [
            { name: 'g. Prevotella', val: 0.089 },
            { name: 'g. Lachnospira', val: 0.054 },
            { name: 'g. Alistipes', val: 0.038 },
            { name: 'g. Dorea', val: 0.021 },
            { name: 'g. Bacteroides', val: -0.072 },
            { name: 'g. Faecalibacterium', val: -0.058 },
            { name: 'g. Blautia', val: -0.031 },
            { name: 'g. Ruminococcus', val: -0.019 },
        ];
        const L_MAX_P = 0.089, L_MAX_N = 0.072, L_SPAN = 0.089 + 0.072;
        function draw() {
            tick++;
            const ctx = cv.getContext('2d');
            const W = cv.offsetWidth, H = cv.offsetHeight;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, W, H);
            ctx.fillStyle = BG;
            ctx.fillRect(0, 0, W, H);
            const rawP = (tick % CYCLE) / CYCLE;
            const phase = Math.min(rawP < 0.55 ? rawP * (0.965 / 0.55) : 0.965, 0.965);
            const masterA = 1 - sm(cl((rawP - 0.930) / 0.040, 0, 1));
            const PX = 16, PY = 14;
            const cW = W - PX * 2;
            const c_lime = PX;
            const TOP = PY + 18, BOT = H - PY - 16, contH = BOT - TOP;
            const NF = 8;
            const fH = cl(contH / NF, 10, 36);
            const mTop = TOP + cl((contH - NF * fH) * 0.25, 0, 20);
            const LBLW = cl(cW * 0.36, 75, 130);
            const VALW = 28;
            const BAR_ZONE = cW - LBLW - 8 - 4 - VALW;
            const zF = L_MAX_N / L_SPAN;
            const L_BARX = c_lime + LBLW + 8;
            const L_ZERO = L_BARX + BAR_ZONE * zF;
            const L_LEFT = BAR_ZONE * zF;
            const L_RIGHT = BAR_ZONE * (1 - zF);
            const lBarH = cl(fH * 0.28, 2, 6);
            const axT = sl(phase, 0.38, 0.45) * masterA;
            ctx.globalAlpha = axT * 0.24;
            ctx.beginPath();
            ctx.moveTo(L_ZERO, mTop - 2);
            ctx.lineTo(L_ZERO, mTop + NF * fH + 2);
            ctx.strokeStyle = MID;
            ctx.lineWidth = 0.7;
            ctx.setLineDash([2, 2]);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.globalAlpha = 1;
            LFEATS.forEach((f, fi) => {
                const lT = sl(phase, 0.40 + fi * 0.030, 0.46 + fi * 0.030) * masterA;
                if (lT < 0.01)
                    return;
                const ly = mTop + fi * fH + fH * 0.5;
                const fill = cl(sl(phase, 0.42 + fi * 0.030, 0.50 + fi * 0.030), 0, 1);
                const isPos = f.val >= 0;
                ctx.globalAlpha = lT * 0.72;
                ctx.textAlign = 'right';
                ctx.textBaseline = 'middle';
                ctx.font = `${isPos ? '500' : '400'} ${cl(fH * 0.40, 6, 9)}px Inter,sans-serif`;
                ctx.fillStyle = INK;
                ctx.save();
                ctx.beginPath();
                ctx.rect(c_lime, ly - fH * 0.5, LBLW, fH);
                ctx.clip();
                ctx.fillText(f.name, c_lime + LBLW, ly);
                ctx.restore();
                ctx.globalAlpha = 1;
                const bw = (isPos ? f.val / L_MAX_P * L_RIGHT : Math.abs(f.val) / L_MAX_N * L_LEFT) * fill;
                if (bw > 0.3) {
                    if (isPos) {
                        ctx.globalAlpha = lT * 0.84;
                        ctx.fillStyle = C_POS;
                        ctx.fillRect(L_ZERO, ly - lBarH / 2, bw, lBarH);
                    }
                    else {
                        ctx.globalAlpha = lT * 0.40;
                        ctx.fillStyle = C_NEG;
                        ctx.fillRect(L_ZERO - bw, ly - lBarH / 2, bw, lBarH);
                        ctx.globalAlpha = lT * 0.50;
                        ctx.strokeStyle = C_NEG;
                        ctx.lineWidth = 0.6;
                        ctx.strokeRect(L_ZERO - bw, ly - lBarH / 2, bw, lBarH);
                    }
                    ctx.globalAlpha = 1;
                }
                if (fill > 0.70) {
                    const pctStr = (Math.abs(f.val) / L_SPAN * 100).toFixed(1) + '%';
                    const vx = c_lime + LBLW + 8 + BAR_ZONE + 4;
                    ctx.globalAlpha = lT * 0.50;
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    ctx.font = `500 ${cl(fH * 0.32, 5, 7)}px "JetBrains Mono",monospace`;
                    ctx.fillStyle = isPos ? C_POS : MID;
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(vx, ly - fH * 0.5, VALW, fH);
                    ctx.clip();
                    ctx.fillText(pctStr, vx, ly);
                    ctx.restore();
                    ctx.globalAlpha = 1;
                }
            });
            raf = requestAnimationFrame(draw);
        }
        draw();
        return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize); };
    }, []);
    return <canvas ref={ref} style={{ width: '100%', height: '100%', display: 'block' }}/>;
}
function AbstractFlow() {
    const cells = [0.08, 0.62, 0.18, 0.86, 0.35, 0.12, 0.72, 0.22, 0.51, 0.93, 0.28, 0.14, 0.44, 0.77, 0.11, 0.58, 0.31, 0.81, 0.17, 0.68, 0.24, 0.49, 0.89, 0.19, 0.37, 0.66, 0.13, 0.74, 0.41, 0.26, 0.56, 0.84, 0.16, 0.47, 0.71, 0.21, 0.34, 0.79, 0.09, 0.63, 0.29, 0.52, 0.87, 0.15, 0.39, 0.69, 0.23, 0.57];
    const network = [
        [472, 76, 13], [523, 52, 9], [555, 99, 11], [510, 121, 8],
        [583, 58, 7], [606, 110, 9], [551, 146, 7], [624, 76, 6]
    ];
    const edges = [[0,1],[0,2],[0,3],[1,2],[1,4],[2,3],[2,5],[2,6],[3,6],[4,5],[4,7],[5,6],[5,7]];
    return <svg viewBox="0 0 680 210" role="img" aria-label="Microbiome abundance data flowing through a machine learning model to a network of learned patterns" style={{ width: '100%', height: '100%', display: 'block', overflow: 'visible' }}>
        <g transform="translate(18 40)">
            {cells.map((value, index) => {
                const row = Math.floor(index / 8);
                const col = index % 8;
                const opacity = 0.10 + value * 0.82;
                return <rect key={index} x={col * 16} y={row * 16} width="12" height="12" rx="2" fill={`rgba(37,99,235,${opacity})`} />;
            })}
            <text x="56" y="122" textAnchor="middle" fontSize="11" fill="#64748b">abundance data</text>
        </g>
        <path className="lp-abstract-path" d="M162 102 C205 102 218 102 253 102" fill="none" stroke="#cbd5e1" strokeWidth="1.5" strokeDasharray="5 7" />
        <path d="M245 96 L254 102 L245 108" fill="none" stroke="#94a3b8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <g transform="translate(277 47)">
            {[0,1,2,3].map(i => <circle key={`a${i}`} cx="0" cy={18 + i * 25} r="6" fill="#dbeafe" stroke="#60a5fa" strokeWidth="1" />)}
            {[0,1,2].map(i => <circle key={`b${i}`} cx="49" cy={30 + i * 28} r="7" fill="#bfdbfe" stroke="#3b82f6" strokeWidth="1" />)}
            {[0,1].map(i => <circle key={`c${i}`} cx="100" cy={44 + i * 39} r="8" fill="#93c5fd" stroke="#2563eb" strokeWidth="1" />)}
            {[0,1,2,3].flatMap(a => [0,1,2].map(b => <line key={`ab${a}-${b}`} x1="6" y1={18 + a * 25} x2="42" y2={30 + b * 28} stroke="#dbeafe" strokeWidth="1" />))}
            {[0,1,2].flatMap(a => [0,1].map(b => <line key={`bc${a}-${b}`} x1="56" y1={30 + a * 28} x2="92" y2={44 + b * 39} stroke="#bfdbfe" strokeWidth="1" />))}
            <text x="50" y="125" textAnchor="middle" fontSize="11" fill="#64748b">machine learning</text>
        </g>
        <path className="lp-abstract-path lp-abstract-path-delay" d="M401 102 C428 102 437 102 459 102" fill="none" stroke="#cbd5e1" strokeWidth="1.5" strokeDasharray="5 7" />
        <path d="M451 96 L460 102 L451 108" fill="none" stroke="#94a3b8" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <g>
            {edges.map(([a,b], index) => <line key={index} x1={network[a][0]} y1={network[a][1]} x2={network[b][0]} y2={network[b][1]} stroke="#cbd5e1" strokeWidth={index % 3 === 0 ? 1.8 : 1.1} opacity="0.86" />)}
            {network.map(([x,y,r], index) => <circle key={index} cx={x} cy={y} r={r} fill={index % 3 === 0 ? '#2563eb' : index % 3 === 1 ? '#60a5fa' : '#bfdbfe'} opacity={index % 3 === 0 ? 0.88 : 0.72} />)}
            <text x="550" y="178" textAnchor="middle" fontSize="11" fill="#64748b">patterns and interactions</text>
        </g>
    </svg>;
}

function DataContextCanvas() {
    const ref = useRef(null);
    useEffect(() => {
        const cv = ref.current;
        if (!cv) return;
        const ctx = cv.getContext('2d');
        const dpr = Math.max(2, window.devicePixelRatio || 1);
        let raf;
        let tick = 0;
        const resize = () => {
            cv.width = cv.offsetWidth * dpr;
            cv.height = cv.offsetHeight * dpr;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        };
        resize();
        window.addEventListener('resize', resize);
        const sm = t => t * t * (3 - 2 * t);
        const cl = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
        const sl = (p, s, e) => sm(cl((p - s) / (e - s + 1e-9), 0, 1));
        const lrp = (a, b, t) => a + (b - a) * t;
        const rr = (x, y, w, h, r) => rRect(ctx, x, y, w, h, r);
        const cellRGB = v => {
            const m = cl(v, 0, 1);
            return `rgb(${Math.round(lrp(241, 2, m))},${Math.round(lrp(245, 155, m))},${Math.round(lrp(249, 190, m))})`;
        };
        const GROUP_A = '#2563eb';
        const GROUP_B = '#64748b';
        const N_A = 18;
        const N_B = 16;
        const N_SAMP = N_A + N_B;
        const N_TAXA = 160;
        let seed = 3317 >>> 0;
        const lcg = () => { seed = Math.imul(1664525, seed) + 1013904223 >>> 0; return seed / 4294967296; };
        const samples = Array.from({ length: N_SAMP }, (_, i) => {
            const inA = i < N_A;
            return {
                inA,
                sx: lcg() * 0.42 + 0.04,
                sy: lcg() * 0.38 + (inA ? 0.04 : 0.56),
                taxa: Array.from({ length: N_TAXA }, () => {
                    const v = lcg();
                    if (v < 0.69) return 0;
                    return Math.pow(lcg(), 1.8);
                })
            };
        });
        const CYCLE = 1500;
        const draw = () => {
            const cW = cv.offsetWidth;
            const cH = cv.offsetHeight;
            ctx.clearRect(0, 0, cW, cH);
            const rawP = (tick % CYCLE) / CYCLE;
            const masterA = 1 - sm(cl((rawP - 0.93) / 0.07, 0, 1));
            const pFadeIn = sl(rawP, 0.00, 0.18);
            const pConverge = sl(rawP, 0.24, 0.52);
            const pLabel = sl(rawP, 0.18, 0.38);
            const pUnfold = sl(rawP, 0.54, 0.84);
            const PX = 72;
            const PY_TOP = 38;
            const PY_BOT = 24;
            const PXR = 20;
            const GAP = 12;
            const availH = cH - PY_TOP - PY_BOT - GAP;
            const ROW_H = availH / N_SAMP;
            const DOT_R = cl(ROW_H * 0.30, 2.4, 4.4);
            const DOT_COL = PX + 10;
            const CELL = Math.max(2, Math.floor(ROW_H * 0.72));
            const hx0 = DOT_COL + 18;
            const tgtY = i => PY_TOP + i * ROW_H + ROW_H / 2 + (i >= N_A ? GAP : 0);
            const srcX = s => PX + s.sx * (cW - PX - PXR) * 0.52;
            const srcY = s => PY_TOP + s.sy * (cH - PY_TOP - PY_BOT);
            if (masterA > 0.01 && pLabel > 0.01) {
                const aY = lrp(srcY(samples[Math.floor(N_A / 2)]), (tgtY(0) + tgtY(N_A - 1)) / 2, pConverge);
                const bY = lrp(srcY(samples[N_A + Math.floor(N_B / 2)]), (tgtY(N_A) + tgtY(N_SAMP - 1)) / 2, pConverge);
                [[aY, 'group A', GROUP_A], [bY, 'group B', GROUP_B]].forEach(([y, label, color]) => {
                    ctx.save();
                    ctx.translate(PX - 8, y);
                    ctx.rotate(-Math.PI / 2);
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'bottom';
                    ctx.font = '600 8px Inter,sans-serif';
                    ctx.globalAlpha = masterA * pLabel * 0.65;
                    ctx.fillStyle = color;
                    ctx.fillText(label, 0, 0);
                    ctx.restore();
                });
            }
            for (let i = 0; i < N_SAMP; i++) {
                const s = samples[i];
                const color = s.inA ? GROUP_A : GROUP_B;
                const ty = tgtY(i);
                const cx = lrp(srcX(s), DOT_COL, pConverge);
                const cy = lrp(srcY(s), ty, pConverge);
                const dotA = masterA * pFadeIn * Math.max(0, 1 - pUnfold * 0.85);
                if (dotA > 0.01) {
                    ctx.globalAlpha = dotA;
                    ctx.beginPath();
                    ctx.arc(cx, cy, DOT_R, 0, 2 * Math.PI);
                    ctx.fillStyle = color;
                    ctx.fill();
                }
                if (pUnfold > 0.01) {
                    const rowDelay = i / N_SAMP * 0.22;
                    const rowReveal = sl(rawP, 0.54 + rowDelay, 0.70 + rowDelay);
                    if (rowReveal < 0.01) continue;
                    ctx.globalAlpha = masterA * sm(cl((rowReveal - 0.05) / 0.12, 0, 1)) * 0.88;
                    ctx.beginPath();
                    ctx.arc(DOT_COL, ty, DOT_R * 0.60, 0, 2 * Math.PI);
                    ctx.fillStyle = color;
                    ctx.fill();
                    const nVisible = Math.round(rowReveal * N_TAXA);
                    for (let j = 0; j < nVisible; j++) {
                        const x = hx0 + j * CELL;
                        if (x > cW - PXR) break;
                        const v = s.taxa[j];
                        const cellA = masterA * sm(cl((rowReveal * N_TAXA - j) / 1.4, 0, 1));
                        if (cellA < 0.02) continue;
                        ctx.globalAlpha = cellA * 0.92;
                        rr(x + 0.3, ty - CELL / 2, CELL - 0.6, CELL, 1);
                        ctx.fillStyle = v < 0.04 ? '#f1f5f9' : cellRGB(v);
                        ctx.fill();
                    }
                }
                ctx.globalAlpha = 1;
            }
            ctx.globalAlpha = masterA * pUnfold * 0.7;
            ctx.font = '500 9px Inter,sans-serif';
            ctx.fillStyle = '#64748b';
            ctx.textAlign = 'left';
            ctx.fillText('samples', 12, 18);
            ctx.textAlign = 'right';
            ctx.fillText('microbial abundance features', cW - 18, 18);
            ctx.globalAlpha = 1;
            tick++;
            raf = requestAnimationFrame(draw);
        };
        raf = requestAnimationFrame(draw);
        return () => {
            cancelAnimationFrame(raf);
            window.removeEventListener('resize', resize);
        };
    }, []);
    return <canvas ref={ref} style={{ width: '100%', height: '100%', display: 'block' }} />;
}

const FlowArrow = ({ text }) => <div className="lp-flow-arrow"><div className="lp-flow-line"/><svg width="14" height="8" viewBox="0 0 14 8" fill="none"><path d="M1 1l6 6 6-6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round"/></svg><span>{text}</span></div>;

export default function Home() {
    return <div className="lp">
        <nav className="lp-nav">
            <Link href="/" className="lp-brand">mllabiome</Link>
            <div className="lp-nav-links">
                <Link href="/docs/current" className="lp-nav-link">Documentation</Link>
                <Link href="https://github.com/CMG-GUTS/mllabiome" className="lp-nav-link lp-nav-extra">GitHub</Link>
                <Link href="/docs/current/quickstart" className="lp-nav-primary">Get started</Link>
            </div>
        </nav>

        <main>
            <section className="lp-hero lp-section">
                <div className="lp-hero-copy">
                    <div className="lp-version">mllabiome <span>0.1.0rc123</span></div>
                    <h1>Machine learning for microbiome research</h1>
                    <p className="lp-hero-lead">Build, compare, combine and explain predictive models from microbiome abundance data.</p>
                    <p className="lp-hero-detail">A research library for finding reproducible patterns in microbiome data and testing whether modelling choices generalize beyond the samples used to build them.</p>
                    <div className="lp-actions">
                        <Link href="/docs/current/quickstart" className="lp-button lp-button-primary">Quickstart</Link>
                        <Link href="/docs/current" className="lp-button">Read the documentation</Link>
                    </div>
                </div>
                <div className="lp-abstract"><AbstractFlow /></div>
            </section>

            <section className="lp-question lp-section">
                <div>
                    <span className="lp-kicker">Compare methods fairly</span>
                    <h2>Does a new algorithm really improve on what already exists?</h2>
                    <p>Place a new estimator beside established approaches and evaluate them under the same data, preprocessing rules and held-out samples. Model selection stays inside the training data, while final comparisons are made on observations that did not guide that selection.</p>
                </div>
                <Link href="/docs/current/evaluation" className="lp-text-link">How comparison works →</Link>
            </section>

            <section className="lp-data lp-section">
                <div className="lp-section-heading">
                    <span className="lp-kicker">From cohort to analysis table</span>
                    <h2>Start with microbiome abundance profiles and study metadata</h2>
                    <p>Samples from a study cohort become rows in an abundance matrix. mllabiome starts from these processed profiles; sequencing and taxonomic profiling happen upstream.</p>
                </div>
                <div className="lp-data-canvas"><DataContextCanvas /></div>
            </section>

            <section className="lp-process lp-section">
                <div className="lp-section-heading lp-process-intro">
                    <span className="lp-kicker">How it works</span>
                    <h2>From candidate models to interpretable results</h2>
                    <p>The detailed workflow stays below the introduction so the first view explains the purpose before the machinery.</p>
                </div>

                <div className="lp-stage-heading">
                    <div className="lp-stage-number">1</div>
                    <div><h3>Compare ways of representing and modelling the data</h3><p>Different microbiome representations and learning algorithms are evaluated under the same validation design.</p><span>MPDR × MPMA-B</span></div>
                </div>
                <div className="lp-animation lp-animation-large"><HeroCanvas /></div>

                <FlowArrow text="promising base models can be combined" />

                <div className="lp-stage-heading">
                    <div className="lp-stage-number">2</div>
                    <div><h3>Combine complementary models</h3><p>Eligible base models can enter ensemble selection using evidence from the training side of the evaluation.</p><span>MPMA-E</span></div>
                </div>
                <div className="lp-animation"><EnsembleSweepCanvas /></div>

                <FlowArrow text="selected models can then be inspected" />

                <div className="lp-stage-heading">
                    <div className="lp-stage-number">3</div>
                    <div><h3>Understand which features drive predictions</h3><p>Global explanations summarize patterns across held-out samples; local explanations show contributions for individual predictions.</p><span>Explainability</span></div>
                </div>
                <div className="lp-explain-grid">
                    <div><div className="lp-mini-title">Across samples</div><div className="lp-animation"><GlobalExplainCanvas /></div></div>
                    <div><div className="lp-mini-title">For one sample</div><div className="lp-animation"><LocalExplainCanvas /></div></div>
                </div>
            </section>

            <section className="lp-workflow lp-section">
                <div className="lp-section-heading">
                    <span className="lp-kicker">Complete workflow</span>
                    <h2>Explore, evaluate, explain and report</h2>
                </div>
                <div className="lp-workflow-row">
                    {['explore', 'evaluate', 'ensemble', 'inference', 'explain', 'robustness', 'report'].map((stage, index) => <React.Fragment key={stage}><span>{stage}</span>{index < 6 && <b>→</b>}</React.Fragment>)}
                </div>
                <p className="lp-research-note">Research use only. Not a medical device. Study-specific models and biological interpretations require independent validation.</p>
            </section>
        </main>

        <footer className="lp-footer"><span>mllabiome · Apache-2.0</span><span>HEREDITARY · EU grant agreement No 101137074</span></footer>
    </div>;
}
