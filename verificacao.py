import streamlit as st
import cv2
import numpy as np
import os
import mediapipe as mp

# Inicializar Mediapipe e o reconhecedor
mp_face_detection = mp.solutions.face_detection
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Função para carregar imagens de treino
def get_images_and_labels(path):
    imagens, labels = [], []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith("jpg") or file.endswith("png"):
                img_path = os.path.join(root, file)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                label = int(os.path.basename(root))
                imagens.append(image)
                labels.append(label)
    return np.array(imagens), np.array(labels)

# Função para cadastrar novos usuários
def register_user(user_id, image):
    user_path = os.path.join('imagens_treino', str(user_id))
    os.makedirs(user_path, exist_ok=True)
    image_path = os.path.join(user_path, f'{user_id}.jpg')
    cv2.imwrite(image_path, image)

# Treinamento
def train_model():
    caminho_treino = 'imagens_treino/'
    imagens_treino, labels = get_images_and_labels(caminho_treino)
    if len(imagens_treino) > 0:
        recognizer.train(imagens_treino, labels)
        recognizer.save('modelo_treinado.yml')

# Carregar o modelo treinado
def load_model():
    recognizer.read('modelo_treinado.yml')

# Função para capturar vídeo
def capture_video():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    return cap

def main():
    st.title("Verificação de Identidade - Detecção Facial")
    
    # Cadastro de usuário
    st.subheader("Cadastrar Usuário")
    user_id = st.text_input("ID do Usuário (Número):")
    uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "png"])
    
    if st.button("Cadastrar"):
        if user_id and uploaded_file:
            # Ler a imagem do usuário
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Registrar o usuário
            register_user(user_id, image_gray)
            st.success(f"Usuário {user_id} cadastrado com sucesso!")
            train_model()  # Re-treinar o modelo após o cadastro
        else:
            st.error("Por favor, insira um ID de usuário e faça o upload de uma imagem.")

    # Captura de vídeo
    if st.button("Iniciar Captura"):
        stframe = st.empty()
        cap = capture_video()
        load_model()  # Carregar o modelo treinado

        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Erro ao capturar a imagem da webcam.")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

                        face = cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
                        if face.size != 0:
                            label, confianca = recognizer.predict(face)
                            if confianca < 100:
                                cv2.putText(frame, f'ID: {label}, Conf: {confianca:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.putText(frame, "Reconhecido", (x, y + height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            else:
                                cv2.putText(frame, "Desconhecido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Exibir o vídeo na interface do Streamlit
                stframe.image(frame, channels='BGR')

        cap.release()

if __name__ == '__main__':
    main()
