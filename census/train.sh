#/bin/sh

PYTHONPATH=$PWD

tfx train $1 \
  --module dnn.main \
  --output /tmp/tensorfx/census \
  --data-train /tmp/tensorfx/census/data/train.csv \
  --data-eval /tmp/tensorfx/census/data/eval.csv \
  --data-schema /tmp/tensorfx/census/data/schema.yaml \
  --data-metadata /tmp/tensorfx/census/data/metadata.json \
  --data-features dnn/features.yaml \
  --log-level-tensorflow ERROR \
  --log-level INFO \
  --batch-size 5 \
  --max-steps 1000 \
  --checkpoint-interval-secs 1 \
  --hidden-layers:1 200 \
  --hidden-layers:2 100 \
  --hidden-layers:3 20 \
