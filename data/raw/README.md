# Dataset provenance

The pipeline downloads CIFAR-10 from the official Toronto URL:

<https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz>

The archive is intentionally not committed. The code records the requested source
URL, actual source used, selected source IDs, and random seed in
`data/processed/dataset_metadata.txt`, `data/processed/benchmark.csv`, and
`reports/results.json`.

If the archive is incomplete, the pipeline does not pretend it downloaded a full
CIFAR-10 copy: it falls back to scikit-learn's local digits image set, upscales it
to 32×32 RGB, and labels that provenance in the generated reports.

The manipulated images are **synthetic research examples** generated locally from
the downloaded authentic source images. They are not an external forgery dataset.
For the next milestone, add CASIA v2.0 and CoMoFoD small following the grouped-split
and license guidance in the root README.
