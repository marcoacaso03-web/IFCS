# IFCS 2026 Data Challenge — Report

**Profiling and Predicting Financial Distress in Italian SMEs**
Repo: `marcoacaso03-web/IFCS` · Dati: `train.csv` (13.956 SMEs, FY2023)

---

## 1. Dataset & preprocessing

- **Righe:** 13.956 imprese (1 cella mancante totale, trascurabile).
- **Target:** `Financial distress` = TRUE nel **10,8 %** dei casi (squilibrio di classe ~1:8).
- **Variabili finanziarie usate (10):** Sales Revenue, Employees, Net income, Operating Income, Maximum deductible amount, Total financial expenses, Tax shield, Operating cash flow, Current taxes, Alert Index.
- **Escluse dalla modellazione:** `Company ID` (identificativo), `Province`/`sector` (usate solo per l'interpretazione geografica/settoriale) e il target.

**Pulizia — `Alert Index` (mixed type).** La colonna è numerica per la maggior parte, ma **519 righe** contengono la stringa `"EXCELLENT"` (miglior classe di rischio). Mappate a **0** (rischio minimo), coerente con un indicatore di early-warning.

**Trasformazione.** Le variabili monetarie/contabili sono fortemente asimmetriche (sales fino a ~50 M k€, molte imprese piccole). Applico `log1p` alle 9 variabili pesate e **standardizzo** (media 0, dev.std 1) prima di clustering e classificazione, così poche grandi imprese non dominano le distanze euclidean e i coefficienti logistici sono confrontabili.

**Multicollinearità (prima della modellazione).** La matrice contabile contiene variabili fortemente correlate. Il **VIF** rivela collinearità severa:

| Variabile | VIF |
|---|---|
| Maximum deductible amount | **203,8** |
| Operating Income | **182,0** |
| Tax shield | 10,2 |
| Net income | 8,7 |
| Total financial expenses | 8,0 |
| Sales Revenue | 2,3 |
| Operating cash flow | 4,6 |
| Current taxes | 3,9 |
| Employees | 1,4 |
| Alert Index | 1,0 |

`Maximum deductible amount` e `Operating Income` hanno correlazione 0,99 (quasi collinearità perfetta); `Tax shield` è meccanicamente derivato da `Total financial expenses` (r = 0,80). Con VIF > 100 i coefficienti logistici diventano instabili e le "importanze" non sono interpretabili singolarmente. **Decisione:** per la classificazione si **rimuovono le variabili derivate/ridondanti** (`Maximum deductible amount`, `Tax shield`), mantenendo l'8-set pulito. Il clustering le trattiene (K-Means è robusto alla collinearità e le variabili restano informative per il profiling).

**Variable selection — feature usate nel clustering vs classificazione.**
- *Clustering (10 feat):* tutte le variabili finanziarie, incluse le derivate (K-Means non soffre di collinearità e servono per il profiling economico).
- *Classificazione (4 feat, finali):* `Operating Income`, `Net income`, `Total financial expenses`, `Sales Revenue per Employee` (Employees trasformata in produttività).

**Come è stata fatta la selection (non solo collinearità, ma anche significatività):**
1. **Multicollinearità (VIF):** rimosse `Maximum deductible amount` (VIF=204, derivata da Operating Income) e `Tax shield` (derivata da Total financial expenses, r=0,80).
2. **Variabili composte/derivate identificate sui dati:** `Alert Index` risulta = `Net income / Total financial expenses` (r=0,953, indice di copertura del debito) → rimossa perché ricombinazione di due predittori già presenti.
3. **Collinearità con il reddito:** `Current taxes` è proporzionale a `Operating Income` (r=0,87; rapporto mediano 0,27 ≈ aliquota effettiva IRES+IRAP) → rimossa per doppia conta.
4. **Non-significatività (Wald):** `Operating cash flow` non significativa (p=0,55) una volta nel modello Op/Net income → rimossa.
5. **Ridondanza dimensionale:** `Sales Revenue` e `Employees` misurano entrambi la dimensione (r=0,52). Anziché scartare `Employees`, è stata **trasformata in `Sales Revenue per Employee`** (produttività del lavoro, r=−0,22 con Employees → informazione ortogonale) che diventa altamente significativa (Wald p=2,3e-04); a quel punto `Sales Revenue` risulta ridondante (p=0,60) e viene rimossa.
6. **Confronto AUC:** il set da 4 ha CV ROC-AUC = **0,853**, in linea (anzi lievemente superiore) con i set più ampi → la rimozione delle variabili ridondanti non ha costo predittivo.

Le 4 variabili finali sono **tutte non-derivate (nel set), economicamente distinte e statisticamente significative** (Wald, modello class-weighted):

| Variabile | ruolo economico | coef | z | p-value | sig |
|---|---|---|---|---|---|
| Operating Income | redditività operativa | −0,790 | −7,58 | 3,3e-14 | `***` |
| Net income | utile netto | −0,442 | −3,32 | 9,2e-04 | `***` |
| Total financial expenses | oneri / leva finanziaria | +0,343 | +4,84 | 1,3e-06 | `***` |
| Sales Revenue per Employee | produttività del lavoro | −0,253 | −3,68 | 2,3e-04 | `***` |

(I segni negativi sui conti indicano che maggiore redditività/produttività → minor distress; `Total financial expenses` positivo = maggior onere del debito → più distress. `Company ID`, `Province`, `sector` non entrano come predittori.)

**Data cleaning — outlier contestuali.** Prima di modellare si rimuovono le righe errate, distinguendo gli outlier *coerenti* da quelli *incongruenti*:
- **Sempre rimosse:** `Sales Revenue ≤ 0` (impossibile) → 3 righe.
- **Outlier incongruente (rimosso):** una riga con ≥1 variabile oltre il fence **3×IQR calcolato sui dati log1p** (asimmetria corretta) **E tutte le variabili correlate dentro il proprio fence** → il valore estremo è ingiustificato dal resto del profilo (errore di rilevazione). → 85 righe.
- **Outlier coerente (mantenuto):** se l'estremo è accompagnato da movimenti congruenti sulle variabili correlate (es. `Net income` molto negativo insieme a `Operating Income` e `Operating cash flow` negativi) la riga è un'impresa realmente in difficoltà ed è **tenuta** (rilevante per il Task B).

Risultato: **13.956 → 13.871 righe** (85 rimosse, 0,6 %). Il distress rate resta 10,8 % e l'AUC di classificazione non cambia (0,852), a conferma che le righe rimosse erano rumore e non segnale.

---

## 2. Task A — Clustering (profilazione non supervisionata)

**Metodo.** K-Means (implementato via `scipy.cluster.vq`, inizializzazione deterministica) su feature standardizzate e log-trasformate.

**Scelta di k.** Silhouette calcolata per k = 2…8 (campionata su 2.500 punti per velocità):

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| silhouette | 0.308 | 0.281 | 0.265 | **0.228** | 0.217 | 0.213 | 0.202 |

La silhouette è massima a **k = 2** (i dati sono un continuum), ma per uno scavo di *profiling* servono segmenti economicamente distinguibili. Si sceglie quindi **k = 5** (trade-off documentato: separazione vs granularità interpretabile) — abbastanza per distinguere micro/piccole/medie/grandi e firme di rischio, senza frammentare eccessivamente.

**I 5 cluster (medie del raw, non trasformate):**

| Cluster | n | % distress | Sales k€ | Emp | Net inc k€ | Op inc k€ | Fin exp k€ | Op CF k€ | Alert Idx |
|---|---|---|---|---|---|---|---|---|---|
| **0** | 3.811 | 7,0 % | 4.314 | 22 | 122 | 230 | 67 | 266 | 8,3 |
| **1** | 3.360 | 12,5 % | 2.060 | 18 | 43 | 63 | 13 | 95 | 383 |
| **2** | 2.810 | 3,0 % | 13.578 | 49 | 815 | 1.232 | 165 | 1.202 | 32,7 |
| **3** | 2.547 | 0,6 % | 5.837 | 23 | 547 | 740 | 4 | 673 | 15.021 |
| **4** | 1.428 | 50,1 % | 4.557 | 33 | **−488** | **−475** | 61 | **−263** | −810 |

**Interpretazione economica**
- **Cluster 2 — "Large healthy"**: fatturato e redditività nettamente superiori (13,6 M k€, margine operativo >1 M k€), distress solo 3 %. Ancore solide del campione.
- **Cluster 3 — "Profitable lean"**: dimensione media ma redditività molto alta (*net income* 547 k€ su 23 dipendenti), distress bassissimo (0,6 %). L'`Alert Index` estremo (15.021) segnala eccellenza.
- **Cluster 0 — "Core SME"**: piccola-media impresa canonica, distress moderato (7 %).
- **Cluster 1 — "Small tight-margin"**: le più piccole e con margini sottili (net income 43 k€), distress 12,5 % — fascia più esposta tra quelle in attivo.
- **Cluster 4 — "Distress core"**: l'unico con **perdite operative e di cassa negative** (Op income −475 k€, Op CF −263 k€) e *net income* −488 k€. Distress al **50,1 %**: è il segmento a rischio sistemico, da monitorare con priorità.

**Distribuzione geografica (macro-aree).** Ogni cluster è dominato dal **Nord** (45–61 % dei casi), coerente con il tessuto industriale italiano; il **Sud** e le **Isole** sono sottorappresentati in tutti i cluster. Il distress non è uniforme: si concentra nel cluster 4, presente ma minoritario nelle aree meridionali/insulari relative. Non emerge un cluster "geograficamente puro", ma una chiara *skew* Nord-centrica della popolazione SME del dataset.

---

## 3. Task B — Classificazione (previsione del distress)

**Modello.** Regressione Logistica (L2, *class-weighted* per gestire lo sbilanciro 1:8) — baseline robusta ed economicamente interpretabile per l'early-warning. Addestrata sulle **5 variabili significative e non-derivate** (vedi selection sopra) così i coefficienti sono stabili e le importanze interpretabili.

**Validazione.** 5-fold stratified CV (ROC-AUC):

| Metrica | Valore |
|---|---|
| **CV ROC-AUC (media)** | **0,852** |
| CV ROC-AUC (std) | 0,008 |
| Train ROC-AUC | 0,856 |

*Nota:* lo stesso AUC (0,852) si otteneva anche con le 10 variabili — le due rimosse erano quindi puramente ridondanti, e ora i coefficienti non sono più distorti dalla collinearità.

**Driver principali** (|coefficiente| standardizzato, set pulito):
1. `Operating Income` (0,66) — redditività operativa è il segnale dominante;
2. `Net income` (0,33);
3. `Total financial expenses` (0,32) — onere del debito;
4. `Operating cash flow` (0,19).

Il modello cattura quindi il distress guardando a redditività operativa, utili e peso degli oneri finanziari — coerente con l'interpretazione dei cluster (il cluster 4 ha proprio questi indicatori negativi).

**Consegna.** `predictions.csv` con colonne `Company ID`, `pred_class` (TRUE/FALSE) — formato richiesto.

---

## 4. Nota su `test_features.csv`

Il file `test_features.csv` **non è presente** nel repo (citato nel task ma non fornito). Le predizioni qui (`artifacts/predictions.csv`) sono quindi **dimostrative** (calcolate *sui dati di train* per verificare formato e pipeline). È presente `score_model.py`: una volta ricevuto `test_features.csv`,

```bash
python3 score_model.py test_features.csv predictions.csv
```

ri-addestra sul `train.csv` completo e produce le predizioni reali sul test, con lo stesso preprocessing e modello.

---

## 5. File prodotti

| File | Contenuto |
|---|---|
| `analysis.py` | Pipeline completa (clustering + classificazione + figure), solo numpy/scipy. |
| `make_slides.py` | Genera la slide deck `.pptx` (stdlib, no python-pptx). |
| `score_model.py` | Scorer riusabile per il test set ufficiale. |
| `artifacts/clusters.csv` | Company ID, cluster, region, macro. |
| `artifacts/predictions.csv` | Company ID, pred_class (dimostrative). |
| `artifacts/metrics.json` | Tutti i numeri. |
| `artifacts/fig_*.png` | 5 figure (silhouette, profili, geo, distress, importanza). |
| `artifacts/IFCS_2026_presentation.pptx` | Slide deck 5 min (8 slide). |

**Ambiente.** Eseguito con Python 3.14 di sistema (numpy/pandas/scipy/matplotlib da `pkg`). Nessuna dipendenza da `scikit-learn`/`python-pptx`, che non compilano su questo Termux.
