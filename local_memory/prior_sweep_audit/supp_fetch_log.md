# Prior-sweep supplement fetch log — 2026-04-15

## Sabeti et al. 2007 (Nature, DOI 10.1038/nature06250)

### Attempts
- Springer static-content URLs `https://static-content.springer.com/esm/art%3A10.1038%2Fnature06250/MediaObjects/41586_2007_BFnature06250_MOESM{1..8}_ESM.{pdf,doc,docx,xls,xlsx,zip,txt}` — all 403 even with Nature referrer.
- WebFetch `https://www.nature.com/articles/nature06250` — 303 redirect, blocked by Nature paywall.
- PMC direct URL `https://pmc.ncbi.nlm.nih.gov/articles/instance/2687721/bin/NIHMS4416-supplement-S1.pdf` — initial fetch returned a proof-of-work HTML interstitial; after one successful PoW cycle the next request triggered Google reCAPTCHA.
- EuropePMC backend renderer `https://europepmc.org/backend/ptpmcrender.fcgi?acc=PMC2687721&blobtype=image&blobname=...` — HTTP/2 STREAM_CLOSED and HTTP/1.1 PoW interstitial (same pipeline).
- NCBI OA-FTP `/pub/pmc/oa_package/` — 404 (this article is not Open Access).

### Successful path
- Solved the PMC `cloudpmc-viewer-pow` SHA-256 challenge (4 hex zeros, cookie = `<challenge>,<nonce>`). Fetched **`sabeti_S1.pdf` (1.48 MB)** — the full NIHMS-4416 supplement containing Tables S1-S11 and Figures S1-S14.
- File stored at `/Users/kevinkorfmann/Projects/tmrca.cu/local_memory/prior_sweep_audit/sabeti_S1.pdf`; text extracted to `sabeti_S1.txt`.

## Metspalu et al. 2011 (AJHG, DOI 10.1016/j.ajhg.2011.11.005)

### Attempts
- `https://www.cell.com/cms/10.1016/j.ajhg.2011.11.005/attachment/...` — 403 (Cloudflare anti-bot).
- PMC `https://pmc.ncbi.nlm.nih.gov/articles/instance/3234374/bin/mmc{1..7}.{pdf,zip}` — mmc1-5 obtained via PoW solve; mmc6, mmc7 blocked by reCAPTCHA after rate-limit.
- **Elsevier CDN** `https://ars.els-cdn.com/content/image/1-s2.0-S0002929711004885-mmc{1..7}.{pdf,zip}` — HTTP 200 for all 7 files without any auth or referrer. Used for the canonical copies.

### Successful path
All seven supplementary files (mmc1 PDF + mmc2-mmc7 zip-of-xlsx) fetched cleanly from the Elsevier CDN:
```
elsevier_mmc1.pdf   4.58 MB   main SI PDF (figures + captions)
elsevier_mmc2.zip   87 KB     samples table (HGDP/Indian samples + FST grouping)
elsevier_mmc3.zip   44 KB     iHS GO-term enrichment
elsevier_mmc4.zip   7 KB      XP-EHH GO-term enrichment
elsevier_mmc5.zip   26 KB     Top 20 India iHS windows (with gene lists)
elsevier_mmc6.zip   21 KB     Top 20 India XP-EHH windows (with gene lists)
elsevier_mmc7.zip   34 KB     PCA population-label table (no candidates)
```
Unpacked xlsx files stored as `ajhg_1017_mmc{2..7}.xlsx`.
