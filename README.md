# 🍎 Classificador de Frutas

## 📝 Descrição do Projeto

Este projeto consiste em um classificador de imagens de frutas. O processo documentado no notebook `fruit_dataset.ipynb` abrange desde a criação da base de dados, com fotos tiradas de 10 tipos de frutas, até o pré-processamento dessas imagens para serem utilizadas em modelos de aprendizado de máquina.

As fotos foram tiradas com fundo branco e preto em diferentes posições para aumentar a variabilidade do dataset.

## 👥 Autores

- Eduardo Bif Pitol  
- Eduardo Bombonatto Lorenzetti  
- Gabriela Strieder Ramos  
- Jose Vitor Montanger Ribeiro da Silva  

## 📊 Informações do Dataset

- **Classes**: 10  
- **Nomes das Classes**:  
  `acerola`, `lemon`, `cherry_tomato`, `khaki`, `banana`, `lime`, `lemon_clove`, `avocado`, `bergamot`, `pear`  
- **Resolução Original**: 3024 x 4032 pixels  
- **Resolução Redimensionada**: 336 x 448 pixels  
- **Câmera Utilizada**: iPhone 12 e iPhone 13 (Modo Retrato)  
- **Condição de Iluminação**: Luz Natural  

## 📁 Estrutura de Diretórios Sugerida

Para que o notebook funcione corretamente, a seguinte estrutura de pastas é recomendada:

trabalho pdi/<br>
├── fruit_dataset.ipynb<br>
├── cnn.ipynb<br>
├── data_frames_vector.ipynb<br>
├── svm_classi.ipynb<br>
├── metadata.json<br>
├── ground_truth/<br>
├── ground_truth_clases/<br>
├── original_dataset/<br>
├── augmented_test/<br>
├── augmented_dataset/<br>
├── fruits_heic/<br>
├── fruits_png/<br>
├── fruits_classes/<br>
│ ├── acerola/<br>
│ ├── banana/<br>
│ ├── ... (outras classes)<br>
└── utils/<br>
│ ├── show_images.py<br>
│ ├── data_normalization.py<br>
│ ├── show_metadata.py<br>
│ ├── augmented_dataset.py<br>
└── docs/<br>