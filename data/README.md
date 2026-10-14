# Data

`sectors_synthetic.csv` — **synthetic surrogate**, not the real study data.

- Schema identical to the real dataset: `facility_type`, `year_established`, `latitude`, `longitude`
- 150 seeded records clustered around Meskel Flower, Chechenia, and Bole (Addis Ababa)
- Generated with `scripts/generate_figures.py` (`synthetic_data(n=150, seed=42)`)

Use it to reproduce the pipeline end-to-end:

```bash
python scripts/generate_figures.py --data data/sectors_synthetic.csv
```

The real Kirkos/Bole sector data cannot be released (participant confidentiality).
