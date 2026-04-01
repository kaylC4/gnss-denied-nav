# gnss-denied-nav

Pipeline modulare per navigazione UAV in ambienti GNSS-denied tramite matching di immagini drone con tile satellitari.

[![CI](https://github.com/YOUR_USERNAME/gnss-denied-nav/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/gnss-denied-nav/actions)

---

## Architettura

Il sistema è composto da 7 moduli plug-in collegati da contratti dati immutabili.
Ogni modulo è sostituibile modificando solo il file di configurazione.

```
SensorFrame
    │
    ▼
PoseEstimator  ──→  CameraPose
    │
    ▼
TileProvider   ──→  TileMosaic
    │
    ▼
PatchSampler   ──→  PatchSet
    │
    ▼
ViewTransformer──→  TransformedQuery   (drone frame → ortometrico)
    │
    ▼
FeatureEncoder ──→  EmbeddingBatch     (patch + query)
    │
    ▼
RetrievalEngine──→  MatchResult        (top-1 ANN)
    │
    ▼
NavigationFilter─→  NavState           (lat, lon, covarianza)
```

## Installazione

### 1. Dipendenze di sistema (macOS)

Richiede [Homebrew](https://brew.sh). Installa Python 3.12 e le librerie di sistema necessarie:

```bash
brew install python@3.12 gdal python-tk@3.12
```

> **Nota:** `gdal` è richiesto da `rasterio` e ha molte dipendenze — l'installazione può richiedere qualche minuto e ~1 GB di spazio libero.

### 2. Clona il repository

```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/gnss-denied-nav
cd gnss-denied-nav
```

### 3. Ambiente virtuale

```bash
/usr/local/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,viz]"
```

### 4. Variabili d'ambiente

```bash
cp .env.example .env
# Compila .env con le chiavi API reali
```

Per attivare l'ambiente in sessioni future:

```bash
source .venv/bin/activate   # all'inizio
deactivate                   # alla fine
```

---

## Dataset — MUN-FRL

Scarica le sequenze dal sito ufficiale: https://mun-frl-vil-dataset.readthedocs.io/en/latest/

Sequenza consigliata per iniziare: **DJI-M600 Quarry-1** (300 m, paesaggio aperto).

### Estrazione del bag (flatting)

Converte un ROS 1 bag (`.bag`) in formato flat (Parquet + immagini PNG). Non richiede ROS installato.

```bash
extract-bag --bag data/lighthouse.bag --out data/lighthouse_flat/
```

L'output prodotto in `data/lighthouse_flat/` sarà:

```
lighthouse_flat/
├── imu.parquet         # [timestamp_ns, ax, ay, az, gx, gy, gz]
├── gnss.parquet        # [timestamp_ns, lat, lon, alt_wgs84_m, alt_agl_m, is_gt]
├── odometry.parquet    # [timestamp_ns, roll_deg, pitch_deg, yaw_deg]
├── frames.parquet      # [timestamp_ns, filename]
└── images/
    └── <timestamp_ns>.png
```

> **Note:** `alt_agl_m` è `NaN` finché non viene eseguito il post-processing con un DEM. `odometry.parquet` contiene roll/pitch/yaw fusi dal topic `/Odometry` (VINS o LiDAR odometry) — nessuna deriva.

Per usare un topic odometry diverso:

```bash
extract-bag --bag data/lighthouse.bag --out data/lighthouse_flat/ \
    --odometry-topic /vins_estimator/odometry
```

### Visualizzazione del dataset (GUI)

Apre la GUI di ispezione del dataset estratto:

```bash
verify-dataset data/lighthouse_flat/
```

## Utilizzo

```bash
# Esegui la pipeline su una sequenza estratta
python -m gnss_denied_nav.pipeline \
    --config config/mun_frl_m600.yaml \
    --data data/quarry1/

# Benchmark retrieval engine (confronto faiss_flat vs faiss_hnsw vs mp_stochastic)
python -m gnss_denied_nav.tools.benchmark \
    --config config/mun_frl_m600.yaml \
    --data data/quarry1/
```

## Aggiungere un nuovo plug-in

```python
# 1. Implementa l'interfaccia
from gnss_denied_nav.interfaces.base import RetrievalEngine

class MyCustomEngine(RetrievalEngine):
    def build_index(self, embeddings, centers): ...
    def query(self, embedding, timestamp_ns=0): ...
    def save_index(self, path): ...
    def load_index(self, path): ...
    name = "my_engine"

# 2. Registra nella factory (nessun altro file da modificare)
from gnss_denied_nav.interfaces.factory import ModuleFactory
ModuleFactory.register(
    "retrieval_engine", "my_engine",
    "my_package.my_engine", "MyCustomEngine"
)

# 3. Seleziona nel config
# pipeline:
#   retrieval_engine:
#     backend: my_engine
```

## Struttura repository

```
gnss-denied-nav/
├── src/gnss_denied_nav/
│   ├── interfaces/
│   │   ├── contracts.py     # strutture dati condivise (frozen dataclass)
│   │   ├── base.py          # classi astratte di ogni modulo
│   │   └── factory.py       # istanziazione da config, registro plug-in
│   ├── modules/
│   │   ├── pose/            # PoseEstimator implementations
│   │   ├── tiles/           # TileProvider implementations
│   │   ├── sampling/        # PatchSampler implementations
│   │   ├── transform/       # ViewTransformer implementations
│   │   ├── encoder/         # FeatureEncoder implementations
│   │   └── retrieval/       # RetrievalEngine implementations
│   └── filters/             # NavigationFilter implementations
├── config/
│   └── mun_frl_m600.yaml    # configurazione DJI-M600 MUN-FRL
├── tools/
│   ├── extract_bag/         # ROS bag → flat files (no ROS required)
│   └── benchmark/           # confronto plug-in su dataset di riferimento
├── tests/
│   ├── unit/
│   └── integration/
├── .env.example             # template chiavi API — copiare in .env
├── pyproject.toml
└── .github/workflows/ci.yml
```

## Requisiti implementati

| ID | Descrizione | Stato |
|----|-------------|-------|
| RF-01 | Configurazione sensori da file YAML | ✅ stub |
| RF-02 | Stima posa camera da INS + RadAlt | 🔲 TODO |
| RF-03 | Search area da ultimo GPS fix + buffer | 🔲 TODO |
| RF-04 | Sliding window + allineamento GSD | 🔲 TODO |
| RF-05 | Trasformazione prospettica inversa | 🔲 TODO |
| RF-06 | Estrazione embedding ONNX / DINOv2 | 🔲 TODO |
| RF-07 | Retrieval ANN (FAISS flat/HNSW/MP) | 🔲 TODO |
| RF-08 | Georeferenziazione top-1 match | 🔲 TODO |
| RF-09 | Output NavState con covarianza | 🔲 TODO |
| RF-10 | EKF loosely-coupled INS + vision | 🔲 TODO |
| RF-11 | Test Mahalanobis outlier rejection | 🔲 TODO |
| RF-12 | Covarianza misura adattiva | 🔲 TODO |
| RF-13 | Benchmark harness multi-backend | 🔲 TODO |
| RA-01 | Architettura plug-in via factory | ✅ done |
| RA-02 | Data contract frozen dataclass | ✅ done |
| RA-03 | Benchmark harness per confronto | 🔲 TODO |

## Citazione dataset

```bibtex
@article{thalagala2024munfrl,
  title   = {MUN-FRL: A Visual-Inertial-LiDAR Dataset for Aerial Autonomous Navigation and Mapping},
  author  = {Thalagala, Ravindu G and De Silva, Oscar and Jayasiri, Awantha and others},
  journal = {The International Journal of Robotics Research},
  year    = {2024},
  doi     = {10.1177/02783649241238358}
}
```

## Licenza

MIT — vedi [LICENSE](LICENSE)
