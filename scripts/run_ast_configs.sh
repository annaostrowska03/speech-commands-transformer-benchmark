#!/usr/bin/env bash
set -e

python src/train_ast.py --config configs/ast_full_baseline.yaml
python src/train_ast.py --config configs/ast_full_frozen_backbone.yaml
python src/train_ast.py --config configs/ast_full_dropout_p03.yaml
python src/train_ast.py --config configs/ast_full_balancing_loss.yaml
python src/train_ast.py --config configs/ast_full_balancing_undersample.yaml
python src/train_ast.py --config configs/ast_full_balancing_loss_undersample.yaml
