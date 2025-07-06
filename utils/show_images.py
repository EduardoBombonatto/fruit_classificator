import os
import numpy as np
import skimage.io
import skimage.color
import skimage.util
import matplotlib.pyplot as plt
from pathlib import Path  # Usar pathlib é mais moderno e robusto


def criar_montagem_imagens(
        diretorio_imagens: str,
        quantidade_imagens: int = 16,
        grid_shape: tuple = (4, 4),
        titulo: str = "Montagem de Imagens"
):
    """
    Carrega imagens de um único diretório e exibe uma montagem.
    Suporta imagens coloridas (RGB), com transparência (RGBA) e tons de cinza.

    Args:
        diretorio_imagens (str): O caminho para a pasta que contém as imagens.
        quantidade_imagens (int): O número máximo de imagens a serem incluídas na montagem.
        grid_shape (tuple): A forma da grade para a montagem (linhas, colunas).
        titulo (str): O título a ser exibido acima da montagem.
    """
    # Converte o caminho para um objeto Path para facilitar a manipulação
    caminho_dir = Path(diretorio_imagens)

    # 1. Validação do Diretório
    if not caminho_dir.is_dir():
        print(f"❌ Erro: Diretório não encontrado em '{caminho_dir}'.")
        return

    # 2. Leitura e Seleção das Imagens
    # Pega todos os arquivos de imagem, ordena e seleciona a quantidade desejada
    arquivos_imagem = sorted(
        [f for f in caminho_dir.glob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    )[:quantidade_imagens]

    if not arquivos_imagem:
        print(f"⚠️ Aviso: Nenhuma imagem encontrada no diretório '{caminho_dir}'.")
        return

    # 3. Carregamento e Padronização das Imagens (lógica reaproveitada)
    imagens_carregadas = []
    for arquivo in arquivos_imagem:
        try:
            imagens_carregadas.append(skimage.io.imread(arquivo))
        except Exception as e:
            print(f"Erro ao ler a imagem {arquivo.name}: {e}")

    # Padroniza todas as imagens para o formato RGB, tratando grayscale e RGBA
    imagens_padronizadas = []
    for img in imagens_carregadas:
        if img.ndim == 2:  # Imagem em tons de cinza
            imagens_padronizadas.append(skimage.color.gray2rgb(img))
        elif img.ndim == 3 and img.shape[2] == 4:  # Imagem com canal alfa (RGBA)
            imagens_padronizadas.append(skimage.color.rgba2rgb(img))
        else:
            imagens_padronizadas.append(img)

    # Garante que as imagens estão no formato de 8 bits (0-255) para a montagem
    imagens_finais = [skimage.util.img_as_ubyte(img) for img in imagens_padronizadas]

    # 4. Criação da Montagem
    if not imagens_finais:
        print("Nenhuma imagem pôde ser processada para a montagem.")
        return

    montagem = skimage.util.montage(
        imagens_finais,
        grid_shape=grid_shape,
        channel_axis=-1  # Indica que o canal de cor é a última dimensão
    )

    # 5. Exibição da Montagem
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.imshow(montagem)
    ax.set_title(titulo, fontsize=24)
    ax.set_axis_off()

    plt.tight_layout()
    plt.show()