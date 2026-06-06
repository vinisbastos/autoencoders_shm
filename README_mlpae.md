# Pipeline MLP-AE — LUMO SHM

Versão limpa e reprodutível do pipeline MLP-AE usado na dissertação.

## Escopo

- Avaliação dos cenários CP1, CP2, CP3 e CP4 com MLP-AE de referência.
- Execução das variantes MLP-AE-SL1024-DZ1200 e MLP-AE-SL1024-DZ12228.
- Exportação de RCOV, MD² e R².

## Instalação

```bash
pip install -r requirements_mlpae.txt
```

## Exemplos de execução

Avaliar CP1--CP4:

```bash
python pipeline_mlpae_lumo_clean.py --scenario DAM4_1 --mode reference_cp --download
```

Executar variantes finais:

```bash
python pipeline_mlpae_lumo_clean.py --scenario DAM4_1 --mode variants --preprocess CP4 --download
```

Executar uma configuração específica:

```bash
python pipeline_mlpae_lumo_clean.py --scenario DAM4_1 --mode single --preprocess CP4 --sequence-length 1024 --overlap 896 --bottleneck-size 1200 --download
```

## Saídas

Os resultados são gravados em:

```text
outputs_mlp_ae/<cenario>/<experimento>/
```

Arquivos exportados:

- `*_config.json`
- `*_training_history.csv`
- `*_global_metrics.csv`
- `*_md2_by_level.csv`
- `*_rcov_by_level.csv`
- `*_model_state_dict.pt`
- `*_training_curve.png`

## Observação

Os termos “referência íntegra” e “alteração estrutural” descrevem a condição física da estrutura durante a aquisição dos sinais.
