import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Classificador de Frutas - Fase III",
    page_icon="🍎",
    layout="centered"
)

# --- CARREGAR O MODELO (COM CACHE PARA SER RÁPIDO) ---
@st.cache_resource
def carregar_modelo():
    # Carrega o modelo gerado pelo gwo_meta3.py
    model = load_model('melhor_modelo.h5')
    return model

# Tentar carregar o modelo
try:
    model = carregar_modelo()
    st.sidebar.success("Modelo carregado com sucesso!")
except Exception as e:
    st.error(f"Erro ao carregar o modelo 'melhor_modelo.h5'. Verifica se o ficheiro existe na pasta. Erro: {e}")
    st.stop()

# --- CLASSES (Têm de ser iguais ao treino) ---
classes = ['abacate', 'acai', 'kumquat', 'papaya', 'pitaya']

# --- FUNÇÃO DE PREPROCESSAMENTO ---
def processar_imagem(image):
    # 1. Garantir que é RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # 2. Redimensionar para 224x224 (Tamanho da MobileNetV2)
    image = image.resize((224, 224))
    
    # 3. Converter para Array Numpy
    img_array = np.array(image)
    
    # 4. Preprocessamento específico da MobileNetV2 
    # (Igual ao usado no treino: preprocess_input coloca valores entre -1 e 1)
    img_array = preprocess_input(img_array)
    
    # 5. Adicionar dimensão do batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- INTERFACE GRÁFICA ---
st.title("Classificação de Frutas usando Redes Neuronais")
st.markdown("Projeto de Inteligência Computacional - Fase III")
st.write("Esta aplicação utiliza **Transfer Learning (MobileNetV2)** otimizada com **GWO**.")

# Escolha do método de entrada
opcao = st.radio("Escolha a fonte da imagem:", ("📸 Usar Câmara", "📂 Carregar Imagens"))

imagens_usuario = []

if opcao == "📂 Carregar Imagens":
    files = st.file_uploader(
        "Carregue imagens (jpg, png, webp)...",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True
    )
    if files:
        imagens_usuario = [Image.open(file) for file in files]

elif opcao == "📸 Usar Câmara":
    camera_file = st.camera_input("Tire uma fotografia")
    if camera_file:
        imagens_usuario = [Image.open(camera_file)]


# --- CLASSIFICAÇÃO ---
if imagens_usuario:
    
    st.markdown("### 📷 Imagens selecionadas:")
    cols = st.columns(3)

    # Mostrar miniaturas
    for idx, img in enumerate(imagens_usuario):
        with cols[idx % 3]:
            st.image(img, caption=f"Imagem {idx+1}", use_container_width=True)
    
    # Botão para classificar todas
    if st.button("🔍 Classificar Todas as Imagens"):
        
        with st.spinner("A analisar todas as frutas..."):
            
            limite = 60  # threshold
            
            for idx, imagem_usuario in enumerate(imagens_usuario):
                st.markdown("---")
                st.subheader(f"📌 Resultado da Imagem {idx+1}")
                
                # Preprocessar
                img_pronta = processar_imagem(imagem_usuario)

                # Prever
                preds = model.predict(img_pronta)
                
                classe_vencedora = classes[np.argmax(preds)]
                confianca = 100 * np.max(preds)

                # Lógica fruta desconhecida
                if confianca < limite:
                    st.error("### Fruta Desconhecida")
                    st.write(f"A confiança foi baixa: **{confianca:.2f}%**")
                else:
                    st.success(f"### Resultado: **{classe_vencedora.upper()}**")
                    st.write(f"Confiança: **{confianca:.2f}%**")
                    st.progress(int(confianca))

                # Probabilidades detalhadas
                with st.expander("Ver detalhes das probabilidades"):
                    for i, class_name in enumerate(classes):
                        prob = preds[0][i] * 100

                        st.write(f"{class_name}: {prob:.2f}%")

