# Pipelines de Autoencoders — LUMO SHM

Este repositório contém versões limpas e reprodutíveis dos pipelines utilizados na dissertação para detecção e localização de alterações estruturais na estrutura LUMO, a partir de sinais de aceleração e modelos de autoencoder treinados exclusivamente com dados da condição estrutural íntegra de referência.

Atualmente, o repositório contempla duas arquiteturas:

- **MLP-AE**: autoencoder de camadas densas, com entrada achatada.
- **CAE-1D**: autoencoder convolucional unidimensional, com entrada multicanal no formato `(N, 18, T)`.

As métricas exportadas pelos pipelines incluem:

- **RCOV** por nível de medição;
- **distância de Mahalanobis quadrática**, `MD²`, global e por nível;
- **coeficiente de determinação global**, `R²`.

---

## 1. Estrutura do repositório

```text
.
├── pipeline_mlpae_lumo_clean.py
├── pipeline_cae1d_lumo_clean.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    ├── DAM6_3_Train.mat
    ├── DAM6_3_Test_1.mat
    ├── ...
    └── DAM6_3_Test_6.mat
```

A pasta `data/` é utilizada localmente para armazenar os arquivos `.mat` da base LUMO. Essa pasta não deve ser versionada no GitHub.

---

## 2. Instalação local

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente virtual no Windows PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

O arquivo `requirements.txt` deve conter:

```text
numpy
pandas
matplotlib
scipy
scikit-learn
torch
h5py
gdown
```

---

## 3. Dados

Os scripts podem baixar automaticamente os arquivos `.mat` da base LUMO utilizando `gdown`, por meio da opção:

```bash
--download
```

Caso os arquivos sejam baixados ou organizados manualmente, eles devem ser salvos na pasta `data/` com o seguinte padrão de nomes:

```text
data/<cenario>_Train.mat
data/<cenario>_Test_1.mat
data/<cenario>_Test_2.mat
data/<cenario>_Test_3.mat
data/<cenario>_Test_4.mat
data/<cenario>_Test_5.mat
data/<cenario>_Test_6.mat
```

Exemplo para o cenário `DAM6_3`:

```text
data/DAM6_3_Train.mat
data/DAM6_3_Test_1.mat
data/DAM6_3_Test_2.mat
data/DAM6_3_Test_3.mat
data/DAM6_3_Test_4.mat
data/DAM6_3_Test_5.mat
data/DAM6_3_Test_6.mat
```

Os cenários disponíveis são:

```text
DAM6_3
DAM4_3
DAM3_3
DAM4_1
```

---

## 4. Pipeline MLP-AE

O arquivo `pipeline_mlpae_lumo_clean.py` implementa o autoencoder de camadas densas utilizado na dissertação.

Nesse modelo, cada janela temporal multissensorial é achatada antes de ser fornecida à rede. Assim, uma janela originalmente organizada como:

```text
18 sensores × SL amostras
```

é convertida em um vetor de dimensão:

```text
18 × SL
```

Exemplos:

```text
SL = 100   → entrada com 1800 atributos
SL = 1024  → entrada com 18432 atributos
```

### 4.1 Especificidades do MLP-AE

O pipeline MLP-AE contempla duas etapas principais:

1. **Pipeline de referência** para avaliação dos cenários de pré-processamento.
2. **Execução das variantes finais** com janelas longas.

### 4.2 MLP-AE de referência

O MLP-AE de referência é utilizado para avaliar os cenários de pré-processamento CP1, CP2, CP3 e CP4.

Configuração padrão:

```text
Arquitetura: MLP-AE
SL = 100
OV = 0
DZ = 1200
learning_rate = 1e-3
batch_size = 128
max_epochs = 5000
patience = 10
scheduler_patience = 5
scheduler_factor = 0.5
```

Cenários de pré-processamento:

```text
CP1: sinal bruto
CP2: filtro passa-banda
CP3: RobustScaler
CP4: filtro passa-banda + RobustScaler
```

### 4.3 Variantes MLP-AE

As variantes finais do MLP-AE utilizam janelas longas e sobreposição:

```text
SL = 1024
OV = 896
```

Variantes avaliadas:

```text
MLP-AE-SL1024-DZ1200
MLP-AE-SL1024-DZ12228
```

### 4.4 Exemplos de iniciação — MLP-AE

Teste rápido do MLP-AE de referência:

```bash
python pipeline_mlpae_lumo_clean.py --scenario DAM6_3 --mode single --preprocess CP4 --sequence-length 100 --overlap 0 --bottleneck-size 1200 --max-epochs 2 --log-every 1 --download
```

Avaliar CP1–CP4 com o MLP-AE de referência:

```bash
python pipeline_mlpae_lumo_clean.py --scenario DAM6_3 --mode reference_cp --download
```

Essa execução roda automaticamente:

```text
CP1
CP2
CP3
CP4
```

com:

```text
SL = 100
OV = 0
DZ = 1200
```

Executar uma configuração específica do MLP-AE:

```bash
python pipeline_mlpae_lumo_clean.py --scenario DAM6_3 --mode single --preprocess CP4 --sequence-length 100 --overlap 0 --bottleneck-size 1200 --download
```

Executar as variantes finais do MLP-AE:

```bash
python pipeline_mlpae_lumo_clean.py --scenario DAM6_3 --mode variants --preprocess CP4 --download
```

Essa execução avalia automaticamente:

```text
MLP-AE-SL1024-DZ1200
MLP-AE-SL1024-DZ12228
```

com:

```text
SL = 1024
OV = 896
```

### 4.5 Saídas do MLP-AE

Os resultados do MLP-AE são gravados em:

```text
outputs_mlp_ae/<cenario>/<experimento>/
```

Exemplo:

```text
outputs_mlp_ae/DAM6_3/single_SL100_OV0_DZ1200_CP4/
```

---

## 5. Pipeline CAE-1D

O arquivo `pipeline_cae1d_lumo_clean.py` implementa o autoencoder convolucional unidimensional utilizado na dissertação.

Diferentemente do MLP-AE, o CAE-1D preserva a estrutura multicanal dos sinais. Assim, as janelas são fornecidas ao modelo no formato:

```text
N × 18 × SL
```

em que:

```text
N  = número de janelas
18 = número de canais de aceleração
SL = comprimento da sequência temporal
```

### 5.1 Especificidades do CAE-1D

O CAE-1D utiliza convoluções unidimensionais ao longo do eixo temporal, considerando os 18 sinais de aceleração como canais de entrada.

Essa representação evita o achatamento dos dados e preserva explicitamente a relação entre sensores e tempo.

Configuração padrão:

```text
Arquitetura: CAE-1D
Entrada: N × 18 × 1024
SL = 1024
OV = 896
Pré-processamento = CP4
```

### 5.2 Pré-processamento do CAE-1D

O CAE-1D utiliza cenário fixo de pré-processamento CP4:

```text
filtro passa-banda + RobustScaler 3D por sensor
```

O RobustScaler 3D é ajustado usando apenas o conjunto de treino da condição estrutural íntegra de referência.

### 5.3 Arquitetura CAE-1D

A arquitetura implementada é um autoencoder convolucional 1D profundo, sem conexões residuais ou skip connections.

Blocos principais:

```text
Conv1d
GroupNorm
ReLU
ConvTranspose1d
```

Configuração padrão:

```text
c1 = 64
c2 = 128
c3 = 256
latent_c = 64
kernel_size = 5
use_gap = False
```

Treinamento padrão:

```text
optimizer = AdamW
learning_rate = 3e-3
weight_decay = 0.0
batch_size = 256
max_epochs = 3500
patience = 40
scheduler_patience = 10
scheduler_factor = 0.5
gradient_clipping = 1.0
```

### 5.4 Exemplos de iniciação — CAE-1D

Teste rápido do CAE-1D:

```bash
python pipeline_cae1d_lumo_clean.py --scenario DAM6_3 --max-epochs 2 --log-every 1 --batch-size 64 --download
```

Execução completa do CAE-1D:

```bash
python pipeline_cae1d_lumo_clean.py --scenario DAM6_3 --max-epochs 3500 --log-every 10 --download
```

Execução completa com batch reduzido:

```bash
python pipeline_cae1d_lumo_clean.py --scenario DAM6_3 --max-epochs 3500 --log-every 10 --batch-size 64 --download
```

Execução para outro cenário:

```bash
python pipeline_cae1d_lumo_clean.py --scenario DAM4_1 --max-epochs 3500 --log-every 10 --batch-size 64 --download
```

Execução com limite superior do filtro em 120 Hz:

```bash
python pipeline_cae1d_lumo_clean.py --scenario DAM6_3 --high-hz 120 --download
```

### 5.5 Execução no Google Colab — CAE-1D

No Google Colab, recomenda-se executar o arquivo `.py` como script.

Teste rápido no Colab:

```python
!python pipeline_cae1d_lumo_clean.py --scenario DAM6_3 --max-epochs 2 --log-every 1 --batch-size 64 --download
```

Execução completa no Colab:

```python
!python pipeline_cae1d_lumo_clean.py --scenario DAM6_3 --max-epochs 3500 --log-every 10 --batch-size 256 --download
```

Execução completa no Colab com batch reduzido:

```python
!python pipeline_cae1d_lumo_clean.py --scenario DAM6_3 --max-epochs 3500 --log-every 10 --batch-size 64 --download
```

### 5.6 Saídas do CAE-1D

Os resultados do CAE-1D são gravados em:

```text
outputs_cae1d/<cenario>/<experimento>/
```

Exemplo:

```text
outputs_cae1d/DAM6_3/CAE1D_SL1024_OV896_CP4_2-80Hz/
```

---

## 6. Arquivos exportados

Tanto o pipeline MLP-AE quanto o pipeline CAE-1D exportam arquivos com o mesmo padrão:

```text
*_config.json
*_training_history.csv
*_global_metrics.csv
*_md2_by_level.csv
*_rcov_by_level.csv
*_model_state_dict.pt
*_training_curve.png
```

Descrição dos arquivos:

- `*_config.json`: configuração utilizada no experimento;
- `*_training_history.csv`: histórico de treino e validação;
- `*_global_metrics.csv`: métricas globais, incluindo `R²` e `MD²` global médio;
- `*_md2_by_level.csv`: distância de Mahalanobis quadrática média por nível de medição;
- `*_rcov_by_level.csv`: RCOV e diferença absoluta de RCOV em relação à referência íntegra de treino;
- `*_model_state_dict.pt`: pesos treinados do modelo;
- `*_training_curve.png`: curva de perda de treino e validação.

---

## 7. Logs de treinamento

Os pipelines exibem mensagens de progresso no terminal.

Exemplo:

```text
[Epoch 0010/3500] train_mse=... | val_mse=... | best_val=... | wait=... | elapsed=...
```

A frequência de exibição pode ser controlada por:

```bash
--log-every
```

Exemplos:

```bash
--log-every 1
```

mostra o progresso a cada época.

```bash
--log-every 10
```

mostra o progresso a cada 10 épocas.

---

## 8. Observação terminológica

Os termos “referência íntegra”, “condição íntegra” e “alteração estrutural” descrevem exclusivamente a condição física da estrutura durante a aquisição dos sinais.

Esses termos não indicam qualquer comprometimento, corrupção ou inconsistência na qualidade dos arquivos de dados.

---

## 9. Arquivos não versionados

Os seguintes diretórios e arquivos não devem ser versionados no GitHub:

```text
.venv/
data/
outputs_mlp_ae/
outputs_cae1d/
*.mat
*.pt
__pycache__/
```

Recomenda-se manter essas entradas no arquivo `.gitignore`.

---

## 10. Fluxo básico com Git

Após alterações no código ou documentação:

```bash
git status
git add .
git commit -m "Update pipelines and documentation"
git push
```

Antes de usar `git add .`, recomenda-se verificar se arquivos grandes, dados experimentais ou outputs não estão sendo adicionados ao versionamento.