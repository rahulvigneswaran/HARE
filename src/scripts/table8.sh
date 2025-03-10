#!/bin/bash

# Define parameter values
DATASETS=(adult compas give_me_some_credit)
GT_PERTURBATIONS=(far)
MLMODELS=(ann)
RECOURSE_METHODS=(wachter growing_spheres face)
OUR_RECOURSE_METHODS=(None binary_search)
SEEDS=(1 2 3 4 5)
K_VALUES=(30)
MULTI_ITER_VALUES=(5)
NOISE_PROB_VALUES=(0.0 0.01 0.05 0.1 0.2)

# Loop over all parameter combinations
for DATASET in "${DATASETS[@]}"; do
  for GT_PERTURBATION in "${GT_PERTURBATIONS[@]}"; do
    for MLMODEL in "${MLMODELS[@]}"; do
      for OUR_RECOURSE_METHOD in "${OUR_RECOURSE_METHODS[@]}"; do
        for RECOURSE_METHOD in "${RECOURSE_METHODS[@]}"; do
          for SEED in "${SEEDS[@]}"; do
            for K in "${K_VALUES[@]}"; do
              for MULTI_ITER in "${MULTI_ITER_VALUES[@]}"; do
                for NOISE_PROB in "${NOISE_PROB_VALUES[@]}"; do
                  CMD="python main.py --dataset $DATASET --gt_perturbation $GT_PERTURBATION \
                  --mlmodel $MLMODEL --our_recourse_method $OUR_RECOURSE_METHOD \
                  --recourse_method $RECOURSE_METHOD --seed $SEED --K $K \
                  --multi_iter $MULTI_ITER --noise_prob $NOISE_PROB"
                  echo "Running: $CMD"
                  eval $CMD
                done
              done
            done
          done
        done
      done
    done
  done
done
