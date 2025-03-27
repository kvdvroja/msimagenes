import json
import cv2
import numpy as np
import urllib.request
from PIL import Image
from math import floor
import mediapipe as mp
import os
import configparser
import random
import string
from ultralytics import YOLO
import threading
import time

class Facial:
    def __init__(self, json_input):
        self.json_local = json_input
        self.config = self.load_config()
        self.model = self.cargar_modelo_yolo()
        self.model_accessories = self.cargar_modelo_accesorios()
        self.model_corbatta = None
        self.model_traje = None
        self.message = ""
        self.mp_face_mesh = mp.solutions.face_mesh

    def load_config(self):
        """Carga las variables de configuración desde un archivo .cfg."""
        config = configparser.ConfigParser()
        config.read('config/config.cfg')
        return config
    
    def cargar_modelo_yolo(self):
        """Carga el modelo YOLOv8 entrenado desde un archivo local."""
        model_path = './best.pt'  # Cambia esto a la ruta donde está guardado tu archivo best.pt
        model = YOLO(model_path)  # Cargar el modelo localmente
        return model
    
    def cargar_modelo_accesorios(self):
        """Carga el modelo YOLOv8 entrenado para detectar accesorios."""
        model_path_accessories = './accesories.pt'  # Ruta del archivo entrenado para accesorios
        model = YOLO(model_path_accessories)
        return model
    
    def cargar_modelo_anteojos(self):
        """Carga el modelo YOLOv8 entrenado para detectar anteojos."""
        model_path_glasses = './glasses.pt'  # Ruta del archivo entrenado para anteojos
        self.model_glasses = YOLO(model_path_glasses)
    
    def cargar_modelo_traje_corbatta(self):
        """Carga el modelo YOLOv8 entrenado para trajes y corbatas."""
        model_path_ties = './ties.pt'
        model_path_suits = './suits.pt'
        self.model_corbatta = YOLO(model_path_ties)
        self.model_traje = YOLO(model_path_suits)

    def generar_nombre_archivo(self, usuario_id):
        """Genera un nombre aleatorio para el archivo de imagen."""
        number_of_strings = 9
        length_of_string = 15
        name_aleatorio = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length_of_string))
        return f"{usuario_id}-{name_aleatorio}.jpg"

    def guardar_imagen(self, image, nombre_archivo):
        """Guarda la imagen con el nombre y ruta especificada en el archivo de configuración."""
        ruta_directorio = os.path.join(self.config['GENERAL']['upload_static'], self.config['GENERAL']['upload_static_dir'])
        ruta_completa = os.path.join(ruta_directorio, nombre_archivo)

        # Crear los directorios si no existen
        os.makedirs(ruta_directorio, exist_ok=True)
        
        cv2.imwrite(ruta_completa, image)
        print(f"Imagen bordeada guardada en: {ruta_completa}")
        
        url_final = os.path.join(self.config['GENERAL']['url_static'], self.config['GENERAL']['upload_static_dir'], nombre_archivo)
        return url_final

    def eliminar_imagen(self, ruta_imagen, delay):
        """Elimina una imagen después de un tiempo de espera (delay)."""
        def eliminar_con_delay():
            time.sleep(delay)
            try:
                os.remove(ruta_imagen)
                print(f"Imagen temporal eliminada: {ruta_imagen}")
            except OSError as e:
                print(f"Error al eliminar la imagen: {e}")

        # Crear un hilo que ejecute la eliminación con retraso
        threading.Thread(target=eliminar_con_delay).start()
        
    def validar_fondo_blanco(self, image):
        data = json.loads(self.json_local)
        parametros = json.loads(data["v_parametro_json"].replace("'", '"'))

        blanco_inferior = np.array([250, 250, 250], dtype=np.uint8)
        blanco_superior = np.array([255, 255, 255], dtype=np.uint8)

        height, width, _ = image.shape

        border_width = int(width * 0.1)
        border_height = int(height * 0.1)

        top_border = image[0:border_height, :]
        bottom_border = image[height - border_height:height, :] 
        left_border = image[:, 0:border_width] 
        right_border = image[:, width - border_width:width] 

        total_white_pixels = 0
        total_pixels = 0
        all_borders = [top_border, bottom_border, left_border, right_border]

        for border in all_borders:
            white_mask = cv2.inRange(border, blanco_inferior, blanco_superior)

            white_pixels = cv2.countNonZero(white_mask)
            total_white_pixels += white_pixels
            total_pixels += border.shape[0] * border.shape[1]

        # Calcular el porcentaje de píxeles blancos en los bordes
        porcentaje_similitud = (total_white_pixels / total_pixels) * 100

        # Obtener el umbral de fondo blanco desde los parámetros
        v_fondo_blanco_similitud = int(parametros["fondo_blanco_similitud"])

        # Si el porcentaje de píxeles blancos es mayor que el umbral, consideramos que el fondo es blanco
        if porcentaje_similitud > v_fondo_blanco_similitud:
            print("Fondo tiene el color blanco requerido.")
            return True  
        else:
            print("Fondo NO tiene el color blanco requerido.")
            return False
        
    def validar_cabeza_recta(self, image):
        """
        Verifica si la cabeza está inclinada hacia un lado o está recta.
        """
        with self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Landmarks específicos de los ojos (consultar los índices en Mediapipe)
                    ojo_izq = face_landmarks.landmark[33]  # Landmarks del ojo izquierdo
                    ojo_der = face_landmarks.landmark[263]  # Landmarks del ojo derecho

                    # Convertir las coordenadas relativas en píxeles
                    h, w, _ = image.shape
                    x_ojo_izq, y_ojo_izq = int(ojo_izq.x * w), int(ojo_izq.y * h)
                    x_ojo_der, y_ojo_der = int(ojo_der.x * w), int(ojo_der.y * h)

                    # Calcular la diferencia de altura entre los ojos
                    diferencia_altura = abs(y_ojo_izq - y_ojo_der)

                    # Si la diferencia de altura entre los ojos es mayor que un umbral, la cabeza está inclinada
                    umbral_inclinacion = 8  # Puedes ajustar este valor según sea necesario

                    if diferencia_altura > umbral_inclinacion:
                        print(f"La cabeza está inclinada (diferencia entre ojos: {diferencia_altura}px)")
                        return False  # La cabeza no está recta
                    else:
                        print(f"La cabeza está recta (diferencia entre ojos: {diferencia_altura}px)")
                        return True  # La cabeza está recta

        # Si no se detecta ningún rostro, se considera inválido
        print("No se detectó ningún rostro en la imagen.")
        return False
    
    def detectar_anteojos(self, image):
        """
        Detecta anteojos en la imagen utilizando el modelo YOLOv8.
        """
        # Cargar el modelo de anteojos si no ha sido cargado aún
        if not hasattr(self, 'model_glasses'):
            self.cargar_modelo_anteojos()

        # Realizar la predicción usando el modelo de anteojos
        resultados_anteojos = self.model_glasses.predict(image)
        anteojos_detectados = False

        for deteccion in resultados_anteojos[0].boxes:
            x1, y1, x2, y2 = map(int, deteccion.xyxy[0].cpu().numpy())
            clase = int(deteccion.cls[0].item())
            confianza = deteccion.conf[0].item()

            # Asumiendo que la clase 0 corresponde a anteojos y el umbral de confianza es 0.5
            if clase == 0 and confianza >= 0.5:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                self.message += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Anteojos detectados</td></tr>"
                anteojos_detectados = True

        if not anteojos_detectados:
            self.message += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #ff2d41;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> No se detectaron anteojos</td></tr>"

        return anteojos_detectados
    
    def detectar_traje_corbatta(self, image, parametros):
        """
        Detecta corbatas en toda la imagen y trajes solo en la parte inferior de la imagen.
        Solo considera válido un traje si también se detecta una corbata.
        Además, limita la detección a un solo traje por imagen.
        """
        if 'traje' in parametros and parametros['traje'] == "SI":
            # Cargar los modelos entrenados si aún no se han cargado
            if not self.model_corbatta or not self.model_traje:
                self.cargar_modelo_traje_corbatta()

            # Detectar corbatas en toda la imagen
            resultados_corbatta = self.model_corbatta.predict(image)
            corbata_detectada = False

            for deteccion in resultados_corbatta[0].boxes:
                x1, y1, x2, y2 = map(int, deteccion.xyxy[0].cpu().numpy())
                clase = int(deteccion.cls[0].item())
                confianza = deteccion.conf[0].item()

                if clase == 0 and confianza >= 0.5:  # Asumiendo clase 0 para corbatas
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    self.message += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Corbata detectada</td></tr>"
                    corbata_detectada = True

            # Detectar trajes solo en la mitad inferior de la imagen
            height, width, _ = image.shape
            mitad_inferior = image[int(height / 2):, :]  # Cortar la imagen en la mitad inferior

            resultados_traje = self.model_traje.predict(mitad_inferior)

            # Solo considerar la detección de un solo traje (el de mayor área o confianza)
            mejor_deteccion = None
            mayor_confianza = 0
            traje_detectado = False

            for deteccion in resultados_traje[0].boxes:
                x1, y1, x2, y2 = map(int, deteccion.xyxy[0].cpu().numpy())
                clase = int(deteccion.cls[0].item())
                confianza = deteccion.conf[0].item()

                if clase == 0 and confianza >= 0.5:  # Asumiendo clase 0 para trajes
                    if confianza > mayor_confianza:
                        mejor_deteccion = (x1, y1, x2, y2)
                        mayor_confianza = confianza
                        traje_detectado = True

            # Si se detectó un traje y una corbata, marcar el traje como válido
            if traje_detectado and corbata_detectada:
                x1, y1, x2, y2 = mejor_deteccion
                # Ajustar las coordenadas y1 e y2 para considerar que cortamos la mitad inferior de la imagen
                y1 += int(height / 2)
                y2 += int(height / 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                self.message += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Traje detectado (con corbata)</td></tr>"
                return True
            else:
                print("No se detectó un traje válido con corbata.")
                if traje_detectado and not corbata_detectada:
                    self.message += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #ff2d41;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Traje detectado pero sin corbata</td></tr>"
                return False
            

    def detectar_accesorios(self, image, msg):
        """
        Detecta accesorios como anteojos, auriculares, collares, etc., utilizando el modelo YOLOv8.
        Ignora los sombreros (hat) para que no afecten la validación.
        """
        resultados_accesorios = self.model_accessories.predict(image)
        accesorios_detectados = []

        # Mapeo de clases de accessories.pt
        clases_nombres = ['Earphones', 'diamond', 'glasses', 'gold', 'hat', 'silver']

        for deteccion in resultados_accesorios[0].boxes:
            x1, y1, x2, y2 = map(int, deteccion.xyxy[0].cpu().numpy())
            clase = int(deteccion.cls[0].item())
            confianza = deteccion.conf[0].item()

            # Si la confianza es mayor o igual a 0.5, consideramos que es una detección válida
            if confianza >= 0.5:
                accesorio_detectado = clases_nombres[clase]

                # Ignorar los sombreros (hat) y continuar
                if accesorio_detectado == 'hat':
                    continue  # Omitir la detección de sombreros y seguir con otros accesorios

                # Dibujar el rectángulo si no es un sombrero
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                accesorios_detectados.append(accesorio_detectado)

        # Generar mensajes basados en los accesorios detectados
        if 'glasses' in accesorios_detectados:
            msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'>Se han detectado anteojos.</td></tr>"
        if 'Earphones' in accesorios_detectados:
            msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'>Se han detectado auriculares.</td></tr>"
        if 'diamond' in accesorios_detectados:
            msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'>Se ha detectado un diamante.</td></tr>"
        if 'gold' in accesorios_detectados:
            msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'>Se ha detectado oro.</td></tr>"
        if 'silver' in accesorios_detectados:
            msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'>Se ha detectado plata.</td></tr>"

        # Si no se detectan accesorios válidos
        if not accesorios_detectados:
            self.message += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #ff2d41;font-size: 20px;'></i></td><td class='ytradre_tbl_td'>No se detectaron accesorios.</td></tr>"

        return accesorios_detectados, msg
        
    def detectar_collares(self, image, msg):
        resultados_collar = self.model.predict(image) 
        collares_detectados = False

        # Detección usando la lógica antigua
        for deteccion in resultados_collar[0].boxes: 
            xyxy = deteccion.xyxy[0].cpu().numpy()  
            x1, y1, x2, y2 = map(int, xyxy) 
            confianza = deteccion.conf[0].item()  

            umbral_confianza = 0.5 

            if confianza >= umbral_confianza:
                collares_detectados = True
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                break  # Si ya se detecta un collar, no es necesario seguir buscando

        # Llama a detectar_accesorios y pasa msg como argumento
        accesorios_detectados, msg = self.detectar_accesorios(image, msg)
        
        # Combinación de la detección de collares entre la lógica antigua y accessories.pt
        if collares_detectados or 'gold' in accesorios_detectados or 'silver' in accesorios_detectados:
            # Solo agregar un mensaje, independientemente de cuántas lógicas lo detecten
            msg += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Se han detectado collares.</td></tr>"
            collares_detectados = True
        else:
            # Si ninguna lógica detecta collares
            self.message += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #ff2d41;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> No se han detectado collares.</td></tr>"

        return collares_detectados, msg



    def Validar(self):
        try:
            rpta_success = "2"
            respuesta_ojo_izq = False
            respuesta_ojo_dere = False
            respuesta = True
            respuesta_fondo_blanco = False
            msg = "<tr><td colspan='2'><h3 style='text-align: center;'>RESULTADOS DE LA VALIDACIÓN BIOMÉTRICA</h3></td></tr>"

            # Parsear el JSON proporcionado
            data = json.loads(self.json_local)
            url_sta = data["url"]
            usuario_id = data["hd_usuario_id"]

            # Obtener parámetros dinámicos desde 'v_parametro_json'
            parametros = json.loads(data["v_parametro_json"].replace("'", '"'))

            # Parámetros dinámicos
            v_resolucion_x_de = int(parametros["resolucion_x"][0]["de"])
            v_resolucion_x_a = int(parametros["resolucion_x"][0]["a"])

            v_resolucion_y_de = int(parametros["resolucion_y"][0]["de"])
            v_resolucion_y_a = int(parametros["resolucion_y"][0]["a"])

            v_dpi_x_de = int(parametros["dpi_x"][0]["de"])
            v_dpi_x_a = int(parametros["dpi_x"][0]["a"])

            v_dpi_y_de = int(parametros["dpi_y"][0]["de"])
            v_dpi_y_a = int(parametros["dpi_y"][0]["a"])

            v_peso_kb_de = int(parametros["peso_kb"][0]["de"])
            v_peso_kb_a = int(parametros["peso_kb"][0]["a"])

            # Descargar la imagen directamente en memoria
            resp = urllib.request.urlopen(url_sta)
            image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # Verificar si la imagen fue cargada correctamente
            if image is None:
                raise Exception("Error al cargar la imagen desde la URL.")

            # Validación de tamaño y resolución de la imagen
            size = len(image_array)  # Tamaño en bytes
            wid = image.shape[1]  # Ancho
            hgt = image.shape[0]  # Alto

            # Validación de peso en KB
            peso_en_kb = int(floor(size / 1024))
            if not (v_peso_kb_de <= peso_en_kb <= v_peso_kb_a):
                respuesta = False
                msg += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #ff2d41;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Peso fuera del rango permitido <b style='color:#ff0018'>({peso_en_kb} KB)</b>&nbsp;<b style='color:blue'>({v_peso_kb_de} KB</b>&nbsp;<b style='color:blue'>-</b>&nbsp;<b style='color:blue'>{v_peso_kb_a} KB)</b></td></tr>"
            else:
                msg += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Peso dentro del rango permitido ({peso_en_kb} KB)&nbsp;<b style='color:blue'>({v_peso_kb_de} KB</b>&nbsp;<b style='color:blue'>-</b>&nbsp;<b style='color:blue'>{v_peso_kb_a} KB)</b></td></tr>"

            # Validación de resolución
            if not (v_resolucion_x_de <= wid <= v_resolucion_x_a):
                respuesta = False
                msg += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #ff2d41;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Resolución ancho fuera del rango permitido <b style='color:#ff0018'>({wid})</b>&nbsp;<b style='color:blue'>({v_resolucion_x_a})</b></td></tr>"
            else:
                msg += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Resolución ancho dentro del rango permitido ({wid})&nbsp;<b style='color:blue'>({v_resolucion_x_a})</b></td></tr>"

            if not (v_resolucion_y_de <= hgt <= v_resolucion_y_a):
                respuesta = False
                msg += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #ff2d41;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Resolución alto fuera del rango permitido <b style='color:#ff0018'>({hgt})</b>&nbsp;<b style='color:blue'>({v_resolucion_y_a})</b></td></tr>"
            else:
                msg += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Resolución alto dentro del rango permitido ({hgt})&nbsp;<b style='color:blue'>({v_resolucion_y_a})</b></td></tr>"

            # Validación de DPI
            dpi_x, dpi_y = self.obtener_dpi(url_sta)
            if not (v_dpi_x_de <= dpi_x <= v_dpi_x_a):
                respuesta = False
                msg += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #ff2d41;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> DPI ancho fuera del rango permitido <b style='color:#ff0018'>({dpi_x})</b>&nbsp;<b style='color:blue'>({v_dpi_x_a})</b></td></tr>"
            else:
                msg += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> DPI ancho dentro del rango permitido ({dpi_x})&nbsp;<b style='color:blue'>({v_dpi_x_a})</b></td></tr>"

            if not (v_dpi_y_de <= dpi_y <= v_dpi_y_a):
                respuesta = False
                msg += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #ff2d41;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> DPI alto fuera del rango permitido <b style='color:#ff0018'>({dpi_y})</b>&nbsp;<b style='color:blue'>({v_dpi_y_a})</b></td></tr>"
            else:
                msg += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> DPI alto dentro del rango permitido ({dpi_y})&nbsp;<b style='color:blue'>({v_dpi_y_a})</b></td></tr>"

            # Detección de rostros y ojos, remarcar cualquier cosa fuera del rango
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Variable para controlar si se bordearon los ojos
            ojos_bordeados = False
            rostro_bordeado = False

            for (x, y, w, h) in faces:
                # Dibuja el rectángulo alrededor del rostro solo una vez
                if not rostro_bordeado:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    rostro_bordeado = True
                
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = image[y:y+h, x:x+w]

            # Detectar los ojos usando Mediapipe
            with self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Landmarks específicos de los ojos (consultar los índices en Mediapipe)
                        ojo_izq = face_landmarks.landmark[133]  # Ejemplo para el ojo izquierdo
                        ojo_der = face_landmarks.landmark[362]  # Ejemplo para el ojo derecho

                        # Convertir las coordenadas relativas en píxeles
                        h, w, _ = image.shape
                        x_ojo_izq, y_ojo_izq = int(ojo_izq.x * w), int(ojo_izq.y * h)
                        x_ojo_der, y_ojo_der = int(ojo_der.x * w), int(ojo_der.y * h)

                        # Tamaño dinámico para los cuadros, basado en la proporción del rostro detectado
                        tamaño_cuadro = int(w * 0.05)  # 5% del ancho de la imagen (puedes ajustar el porcentaje según sea necesario)

                        intensidad_izq = np.mean(image[y_ojo_izq-tamaño_cuadro:y_ojo_izq+tamaño_cuadro, x_ojo_izq-tamaño_cuadro:x_ojo_izq+tamaño_cuadro])
                        intensidad_der = np.mean(image[y_ojo_der-tamaño_cuadro:y_ojo_der+tamaño_cuadro, x_ojo_der-tamaño_cuadro:x_ojo_der+tamaño_cuadro])
                        umbral_intensidad = 50
                        
                        # Desplazamiento horizontal para separar los cuadros entre sí
                        desplazamiento_x = int(w * 0.05)  # 2% del ancho de la imagen (ajustar según sea necesario)

                        # Dibujar los cuadros en los ojos con el desplazamiento
                        if intensidad_izq > umbral_intensidad and not respuesta_ojo_izq:
                            respuesta_ojo_izq = True
                            msg += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Ojo izquierdo detectado en ({x_ojo_izq}, {y_ojo_izq})</td></tr>"
                            cv2.rectangle(image, (x_ojo_izq - tamaño_cuadro - desplazamiento_x, y_ojo_izq - tamaño_cuadro), 
                                  (x_ojo_izq + tamaño_cuadro - desplazamiento_x, y_ojo_izq + tamaño_cuadro), (0, 0, 255), 2)

                        if intensidad_der > umbral_intensidad and not respuesta_ojo_dere:
                            respuesta_ojo_dere = True
                            msg += f"<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Ojo derecho detectado en ({x_ojo_der}, {y_ojo_der})</td></tr>"
                            cv2.rectangle(image, (x_ojo_der - tamaño_cuadro + desplazamiento_x, y_ojo_der - tamaño_cuadro), 
                                  (x_ojo_der + tamaño_cuadro + desplazamiento_x, y_ojo_der + tamaño_cuadro), (0, 0, 255), 2)

                    # Verificación final para los ojos
                if not respuesta_ojo_izq:
                    msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: red;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Ojo izquierdo no detectado</td></tr>"
                    respuesta = False

                if not respuesta_ojo_dere:
                    msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: red;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Ojo derecho no detectado</td></tr>"
                    respuesta = False
                    
            if self.validar_fondo_blanco(image):
                msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Fondo tiene el color blanco requerido </td></tr>"
                respuesta_fondo_blanco = True
            else:
                msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #ff2d41;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> Fondo <b style='color:red;'>NO</b> tiene el color blanco requerido </td></tr>"
                respuesta = False

            # Detección de collares
            collares_detectados, msg = self.detectar_collares(image, msg)
            if collares_detectados:
                self.message += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'>Se han detectado collares.</td></tr>"
            else:
                #respuesta = True
                self.message += "COLLAR DETECTADO"
            
            if 'traje' in parametros and parametros['traje'] == "SI":
                if self.detectar_traje_corbatta(image, parametros):
                    msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'>Se ha detectado un traje.</td></tr>"
                else:
                    msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #ff2d41;font-size: 20px;'></i></td><td class='ytradre_tbl_td'><b style='color:red;'>NO</b> se detecta traje.</td></tr>"
                    respuesta = False
            else:
                print('TRAJE == NO')
                #msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #ff2d41;font-size: 20px;'></i></td><td class='ytradre_tbl_td'>Detección de traje no habilitada.</td></tr>"


            # Si se detectaron ojos o rostros, generar nombre del archivo y guardar la imagen
            if ojos_bordeados or rostro_bordeado:
                nombre_archivo = self.generar_nombre_archivo(usuario_id)
                url_final = self.guardar_imagen(image, nombre_archivo)

                # Eliminar la imagen temporal después de un retraso de 10 segundos
                ruta_imagen = os.path.join(self.config['GENERAL']['upload_static'], self.config['GENERAL']['upload_static_dir'], nombre_archivo)
                self.eliminar_imagen(ruta_imagen, 1200)  # Eliminar la imagen después de 20 minutos
                
            # if self.detectar_anteojos(image):
            #     msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'>Se han detectado anteojos.</td></tr>"
            # else:
            #     #msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #ff2d41;font-size: 20px;'></i></td><td class='ytradre_tbl_td'><b style='color:red;'>NO</b> se detectaron anteojos.</td></tr>"
            #     respuesta = False
                
            if self.validar_cabeza_recta(image):
                #msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #28a745;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> La cabeza está recta </td></tr>"
                respuesta_cabeza_recta = True
            else:
                msg += "<tr><td><i class='fa fa-circle' aria-hidden='true' style='color: #ff2d41;font-size: 20px;'></i></td><td class='ytradre_tbl_td'> La cabeza está&nbsp;<b style='color:red;'>inclinada</b></td></tr>"
                respuesta = False

            # Finalizar
            self.success = "1" if respuesta else "2"
            self.message = msg
            #self.url_micro = url_sta
            self.url_micro = self.config['GENERAL']['url_static']+nombre_archivo

        except Exception as e:
            msg += f"<tr><td>Error: {str(e)}</td></tr>"
            self.success = "2"
            self.message = msg
            self.url_micro = url_sta

    def obtener_dpi(self, image_url):
        """Descarga la imagen y obtiene el DPI con PIL."""
        resp = urllib.request.urlopen(image_url)
        image_file = Image.open(resp)
        dpi = image_file.info.get('dpi', (72, 72))  # Valor por defecto si no tiene DPI
        return dpi
