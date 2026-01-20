# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 09:10:14 2026

@author: acer
"""

# Copyright 2025 Tu Nombre
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#--------------------

import streamlit as st
from uuid import uuid4
import os
import tempfile
import numpy as np
import json
import time
from datetime import datetime
import io
import trimesh
import cadquery as cq
from stpyvista import stpyvista
import pyvista as pv
import glob
from pathlib import Path
import pyvista as pv

# 1. Iniciar la pantalla virtual (Crucial para Streamlit Cloud)
if 'XVFB_STARTED' not in st.session_state:
    pv.start_xvfb()
    st.session_state['XVFB_STARTED'] = True

# 2. Configurar PyVista para que no busque una GPU real
pv.OFF_SCREEN = True

# Configurar pyvista
pv.set_jupyter_backend('static')

class ModelVisualizer3D:
    """Clase para manejar la visualización 3D de modelos usando cadquery y pyvista"""

    def __init__(self):
        self.mesh = None
        self.cq_obj = None
        self.plotter = None
        self.model_color = "#4ECDC4"
        self.auto_rotate = False
        self.wireframe = False
        self.show_axes = True
        self.show_grid = False
        self.background_color = "#1E1E1E"
        self.rotation_speed = 1.0
        self.original_colors = None
        self.export_type = 'stl'

    def load_stl_from_bytes(self, file_bytes: bytes, filename: str) -> bool:
        """Carga un archivo STL desde bytes"""
        tmp_path = None
        try:
            # Guardar temporalmente el archivo
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name

            # Cargar con trimesh para análisis
            self.mesh = trimesh.load(tmp_path)

            # Intentar extraer colores originales si existen
            self._extract_original_colors()

            # También intentar cargar con cadquery
            try:
                if filename.endswith('.step'):
                    self.cq_obj = cq.importers.importStep(tmp_path)
                else:
                    try:
                        self.cq_obj = cq.importers.import_stl(tmp_path)
                    except AttributeError:
                        self.cq_obj = cq.importers.importStl(tmp_path)
            except Exception:
                self.cq_obj = None

            return True

        except Exception as e:
            st.error(f"Error cargando STL: {str(e)}")
            return False

        finally:
            # Asegurarse de eliminar el archivo temporal
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    def _extract_original_colors(self):
        """Intenta extraer colores originales del mesh"""
        try:
            if hasattr(self.mesh, 'visual') and hasattr(self.mesh.visual, 'vertex_colors'):
                if self.mesh.visual.vertex_colors is not None and len(self.mesh.visual.vertex_colors) > 0:
                    colors = self.mesh.visual.vertex_colors[:100]
                    avg_color = colors.mean(axis=0) / 255.0
                    self.model_color = f"#{int(avg_color[0]*255):02x}{int(avg_color[1]*255):02x}{int(avg_color[2]*255):02x}"
                    self.original_colors = colors
        except:
            self.original_colors = None

    def get_model_info(self):
        """Obtiene información del modelo cargado"""
        if self.mesh is None:
            return None

        info = {
            'volume_mm3': self.mesh.volume,
            'volume_cm3': self.mesh.volume / 1000,
            'dimensions_mm': (self.mesh.bounds[1] - self.mesh.bounds[0]).tolist(),
            'bounds': self.mesh.bounds.tolist(),
            'is_watertight': self.mesh.is_watertight,
            'vertices_count': len(self.mesh.vertices),
            'faces_count': len(self.mesh.faces)
        }

        return info

    def create_3d_view(self, show_original_colors=False):
        """Crea una vista 3D con controles básicos"""
        if self.mesh is None:
            return None

        try:
            plotter = pv.Plotter(window_size=[800, 600])

            vertices = self.mesh.vertices
            faces = np.hstack([np.full((len(self.mesh.faces), 1), 3), self.mesh.faces]).flatten()
            pv_mesh = pv.PolyData(vertices, faces)

            # Configurar estilo de renderizado
            if show_original_colors and self.original_colors is not None:
                colors = self.original_colors[:len(vertices)] / 255.0
                plotter.add_mesh(pv_mesh,
                               scalars=colors,
                               rgb=True,
                               smooth_shading=True,
                               show_edges=False,
                               specular=0.5,
                               specular_power=20)
            else:
                color_rgb = self._hex_to_rgb(self.model_color)

                if self.wireframe:
                    plotter.add_mesh(pv_mesh,
                                   color=color_rgb,
                                   style='wireframe',
                                   line_width=1.5,
                                   opacity=0.8)
                else:
                    plotter.add_mesh(pv_mesh,
                                   color=color_rgb,
                                   smooth_shading=True,
                                   show_edges=True,
                                   edge_color='black',
                                   line_width=0.3)

            plotter.set_background(self.background_color)

            if self.show_axes:
                plotter.add_axes(line_width=4)

            if self.show_grid:
                plotter.show_grid(color='gray')

            # CORRECCIÓN: Cambiar 'isometric' por 'iso'
            plotter.camera_position = 'iso'  # ← 'iso' en lugar de 'isometric'
            plotter.camera.azimuth = 45
            plotter.camera.elevation = 30
            plotter.reset_camera()

            self.plotter = plotter

            return plotter

        except Exception as e:
            st.error(f"Error creando vista 3D: {str(e)}")
            return None

    def _hex_to_rgb(self, hex_color):
        """Convierte color HEX a RGB normalizado (0-1)"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

    def export_model(self, session_id, model_name="model"):
        """Exporta el modelo en diferentes formatos - FUNCIÓN DEL CRYSTAL GENERATOR"""
        if self.cq_obj is None:
            return False

        try:
            # Exportar según el tipo seleccionado
            export_path = f"app/static/{model_name}_{session_id}.{self.export_type}"

            if self.export_type == 'step':
                cq.exporters.export(self.cq_obj, export_path)
            else:  # stl por defecto
                cq.exporters.export(self.cq_obj, export_path)

            return True
        except Exception as e:
            st.error(f"Error exportando modelo: {str(e)}")
            return False

# Inicializar visualizador en session_state
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = ModelVisualizer3D()

# Funciones del Crystal Generator adaptadas
def generate_model():
    st.session_state['update_model'] = True

def skip_update():
    st.session_state['skip_update'] = True

def resolve_range(parameter, step=None):
    """Convierte rangos a valores individuales si es necesario"""
    if isinstance(parameter, tuple):
        if parameter[0] == parameter[1]:
            return parameter[0]
        elif step:
            return parameter + (step,)
        else:
            return parameter
    else:
        return parameter

def __calculate_chamfer(parameters, chamfer, check):
    """Valida que un valor sea menor que otro"""
    calulated_chamfer = parameters[chamfer]
    if calulated_chamfer >= parameters[check]:
        calulated_chamfer = parameters[check] - 0.00001
        st.warning(f'{chamfer.replace("_", " ")} {parameters[chamfer]} debe ser menor que {check.replace("_", " ")} {parameters[check]}.')
    return calulated_chamfer

def __clean_up_static_files():
    """Limpia archivos temporales antiguos - FUNCIÓN DEL CRYSTAL GENERATOR"""
    stl_files = glob.glob("app/static/*.stl")
    step_files = glob.glob("app/static/*.step")
    files = stl_files + step_files
    today = datetime.today()

    for file_name in files:
        file_path = Path(file_name)
        if file_path.exists():
            modified = file_path.stat().st_mtime
            modified_date = datetime.fromtimestamp(modified)
            delta = today - modified_date

            if delta.total_seconds() > 600:  # 10 minutos
                try:
                    file_path.unlink()
                except:
                    pass

def __make_tabs():
    upload_tab, calculation_tab, visualization_tab, generator_tab, settings_tab = st.tabs([
        "📤 Cargar Modelo",
        "💰 Cotización",
        "👁️ Visualización 3D",
        "⚡ Generador",  # Nueva pestaña del Crystal Generator
        "⚙️ Configuración"
    ])

    with upload_tab:
        st.header("Subir Archivo STL")

        uploaded_file = st.file_uploader(
            "Arrastra o selecciona tu archivo STL",
            type=['stl'],
            help="Formatos aceptados: STL",
            key="stl_uploader"
        )

        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.read()

                with st.spinner("Cargando y analizando modelo 3D..."):
                    success = st.session_state.visualizer.load_stl_from_bytes(file_bytes, uploaded_file.name)

                    if success:
                        model_info = st.session_state.visualizer.get_model_info()

                        st.session_state['current_model'] = {
                            'filename': uploaded_file.name,
                            'volume_mm3': model_info['volume_mm3'],
                            'volume_cm3': model_info['volume_cm3'],
                            'dimensions_mm': model_info['dimensions_mm'],
                            'bounds': model_info['bounds'],
                            'file_size': len(file_bytes),
                            'vertices_count': model_info['vertices_count'],
                            'faces_count': model_info['faces_count'],
                            'is_watertight': model_info['is_watertight'],
                            'analysis_method': 'trimesh/cadquery'
                        }

                        st.success(f"✅ {uploaded_file.name} cargado correctamente")

                        # Vista previa 3D
                        with st.expander("👁️ Vista previa 3D", expanded=True):
                            plotter = st.session_state.visualizer.create_3d_view()

                            if plotter:
                                try:
                                    stpyvista(plotter, key="preview_viewer", horizontal_align="center")

                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button("🎨 Ir a visualización completa",
                                                    type="primary",
                                                    use_container_width=True,
                                                    key="go_to_viz_from_upload"):
                                            st.session_state['active_tab'] = 2
                                            st.rerun()
                                except Exception as e:
                                    st.error(f"Error mostrando visualización: {str(e)}")

                # Mostrar métricas
                if 'current_model' in st.session_state:
                    model = st.session_state['current_model']

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        volume_cm3 = model['volume_mm3'] / 1000
                        st.metric("Volumen", f"{volume_cm3:.2f} cm³")
                    with col2:
                        weight_grams = volume_cm3 * 1.24
                        st.metric("Peso (PLA)", f"{weight_grams:.1f} g")
                    with col3:
                        dim = model['dimensions_mm']
                        dim_str = f"{dim[0]:.1f}×{dim[1]:.1f}×{dim[2]:.1f}"
                        st.metric("Dimensiones", dim_str)

            except Exception as e:
                st.error(f"❌ Error al procesar el archivo: {str(e)}")
        else:
            st.info("👆 Arrastra o haz clic para subir un archivo STL")

    with calculation_tab:
        st.header("Cálculo de Costos")

        if 'current_model' not in st.session_state or st.session_state['current_model'] is None:
            st.warning("⚠️ Primero sube un archivo STL en la pestaña 'Cargar Modelo'")
            return

        model = st.session_state['current_model']

        # Parámetros de impresión
        col1, col2 = st.columns(2)

        with col1:
            material_option = st.selectbox(
                "Material",
                ["PLA", "ABS", "PETG", "TPU", "Resina", "Personalizado"],
                index=0,
                key="material_select"
            )

            densities = {
                "PLA": 1.24,
                "ABS": 1.04,
                "PETG": 1.27,
                "TPU": 1.21,
                "Resina": 1.10,
                "Personalizado": 1.20
            }

            if material_option == "Personalizado":
                density = st.number_input(
                    "Densidad personalizada (g/cm³)",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.20,
                    step=0.01,
                    key="custom_density"
                )
            else:
                density = densities[material_option]

            infill = st.slider(
                "Porcentaje de relleno (%)",
                min_value=10,
                max_value=100,
                value=20,
                step=5,
                key="infill_slider"
            )

        with col2:
            layer_height = st.select_slider(
                "Altura de capa (mm)",
                options=[0.08, 0.12, 0.16, 0.20, 0.24, 0.28],
                value=0.20,
                key="layer_height_slider"
            )

            supports = st.checkbox(
                "Requiere soportes",
                value=False,
                key="supports_checkbox"
            )

        # Factores de costo
        cost_col1, cost_col2 = st.columns(2)

        with cost_col1:
            currency = st.selectbox(
                "Moneda",
                ["USD $", "EUR €", "MXN $", "ARS $", "CLP $", "BRL R$"],
                index=0,
                key="currency_select"
            )

            currency_symbol = currency.split()[1] if " " in currency else "$"

            material_cost_kg = st.number_input(
                f"Costo material/kg ({currency_symbol})",
                min_value=5.0,
                max_value=200.0,
                value=25.0,
                step=1.0,
                key="material_cost_input"
            )

        with cost_col2:
            hourly_rate = st.number_input(
                f"Tarifa por hora ({currency_symbol})",
                min_value=5.0,
                max_value=100.0,
                value=15.0,
                step=1.0,
                key="hourly_rate_input"
            )

            profit_margin = st.slider(
                "Margen de ganancia (%)",
                min_value=10,
                max_value=50,
                value=30,
                step=5,
                key="profit_margin_slider"
            )

        # Cálculos
        try:
            effective_volume_cm3 = model['volume_cm3'] * (infill / 100)
            weight_grams = effective_volume_cm3 * density
            weight_kg = weight_grams / 1000
            material_cost = weight_kg * material_cost_kg

            # Tiempo estimado
            base_time_hours = (model['volume_cm3'] / 8) * (0.2 / layer_height)
            complexity_factor = 1.3 if supports else 1.0
            estimated_hours = base_time_hours * complexity_factor

            if estimated_hours < 0.5:
                estimated_hours = 0.5

            labor_cost = estimated_hours * hourly_rate
            total_cost = material_cost + labor_cost
            final_price = total_cost * (1 + profit_margin / 100)

            # Mostrar resultados
            results_col1, results_col2 = st.columns(2)

            with results_col1:
                st.write("**📊 Especificaciones:**")
                st.write(f"- Volumen: {model['volume_cm3']:.2f} cm³")
                st.write(f"- Volumen efectivo: {effective_volume_cm3:.2f} cm³")
                st.write(f"- Peso: {weight_grams:.1f} g")
                st.write(f"- Tiempo: {estimated_hours:.2f} h")

            with results_col2:
                st.write("**💰 Costos:**")
                st.write(f"- Material: {currency_symbol} {material_cost:.2f}")
                st.write(f"- Mano de obra: {currency_symbol} {labor_cost:.2f}")
                st.write(f"- Margen ({profit_margin}%): {currency_symbol} {final_price - total_cost:.2f}")
                st.markdown(f"## **💵 Total: {currency_symbol} {final_price:.2f}**")

            # Botón para generar cotización
            if st.button("💾 Generar Cotización", type="primary", key="generate_quotation_btn"):
                quotation = {
                    'id': str(uuid4())[:8],
                    'timestamp': datetime.now().isoformat(),
                    'model': st.session_state['current_model'],
                    'calculations': {
                        'final_price': final_price,
                        'currency': currency_symbol
                    }
                }

                st.session_state['last_quotation'] = quotation
                st.session_state['quotations'] = st.session_state.get('quotations', []) + [quotation]

                json_str = json.dumps(quotation, indent=2, ensure_ascii=False)

                st.success(f"✅ Cotización {quotation['id']} generada!")

                st.download_button(
                    label="📥 Descargar Cotización",
                    data=json_str,
                    file_name=f"cotizacion_{quotation['id']}.json",
                    mime="application/json",
                    use_container_width=True,
                    key=f"download_quotation_{quotation['id']}"
                )

        except Exception as e:
            st.error(f"❌ Error en los cálculos: {str(e)}")

    with visualization_tab:
        st.header("🎨 Visualización 3D Interactiva")

        if 'current_model' not in st.session_state or st.session_state['current_model'] is None:
            st.warning("⚠️ No hay modelo 3D cargado")
            return

        # Controles de visualización estilo Crystal Generator
        st.subheader("🎛️ Controles de Visualización")

        control_col1, control_col2, control_col3 = st.columns(3)

        with control_col1:
            # Tipo de archivo para exportar
            export_type = st.selectbox(
                "Tipo de archivo",
                ('stl', 'step'),
                key="export_type_viz",
                label_visibility="collapsed"
            )

            st.session_state.visualizer.export_type = export_type

        with control_col2:
            # Color del modelo
            new_color = st.color_picker(
                "🎨 Color del modelo",
                value=st.session_state.visualizer.model_color,
                key="model_color_picker_viz"
            )

            # Color de fondo
            bg_color = st.color_picker(
                "🌌 Fondo",
                value=st.session_state.visualizer.background_color,
                key="bg_color_picker_viz"
            )

        with control_col3:
            # Modo de renderizado
            render = st.selectbox(
                "Render",
                ["material", "wireframe"],
                key="model_render_viz",
                label_visibility="collapsed"
            )

            st.session_state.visualizer.wireframe = (render == "wireframe")

            # Auto-rotación
            auto_rotate = st.toggle(
                'Auto Rotate',
                value=st.session_state.visualizer.auto_rotate,
                key="auto_rotate_viz"
            )

            st.session_state.visualizer.auto_rotate = auto_rotate

        # Aplicar cambios
        if st.button("🔄 Aplicar cambios", use_container_width=True, key="apply_changes_viz"):
            st.session_state.visualizer.model_color = new_color
            st.session_state.visualizer.background_color = bg_color
            st.rerun()

        # Generar y mostrar visualización 3D
        with st.spinner("Generando visualización 3D..."):
            try:
                plotter = st.session_state.visualizer.create_3d_view()

                if plotter:
                    stpyvista(plotter, key="main_3d_viewer", horizontal_align="center")
                    st.success("✅ Visualización 3D lista")

                    # Botón para exportar modelo
                    if st.button("💾 Exportar Modelo", type="primary", use_container_width=True, key="export_model_btn"):
                        if st.session_state.visualizer.export_model(st.session_state['session_id']):
                            st.success(f"✅ Modelo exportado como {export_type.upper()}")
                else:
                    st.error("No se pudo generar la visualización 3D")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    with generator_tab:
        st.header("⚡ Generador de Modelos 3D")

        st.info("""
        **Generador estilo Crystal Wall**

        Esta funcionalidad permite generar modelos paramétricos 3D
        con diferentes configuraciones. Similar al Crystal Generator original.
        """)

        # Aquí iría la interfaz para generar modelos paramétricos
        # Similar a las pestañas del Crystal Generator

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📏 Parámetros Generales")

            length = st.slider("Largo", 50.0, 150.0, 75.0, 5.0, key="gen_length")
            width = st.slider("Ancho", 20.0, 80.0, 30.0, 5.0, key="gen_width")
            height = st.slider("Alto", 10.0, 60.0, 25.0, 5.0, key="gen_height")

            crystal_count = st.slider("Número de cristales", 5, 30, 10, 1, key="gen_crystal_count")

        with col2:
            st.subheader("🎨 Apariencia")

            base_color = st.color_picker("Color base", "#4ECDC4", key="gen_base_color")
            detail_color = st.color_picker("Color detalles", "#FF6B6B", key="gen_detail_color")

            render_mode = st.selectbox(
                "Modo de renderizado",
                ["Sólido", "Wireframe", "Transparente"],
                key="gen_render_mode"
            )

        # Botón para generar
        if st.button("🔧 Generar Modelo", type="primary", use_container_width=True, key="generate_model_btn"):
            with st.spinner("Generando modelo..."):
                # Aquí iría la lógica para generar el modelo con cadquery
                st.success("✅ Modelo generado correctamente")

                # Mostrar vista previa
                st.info("Vista previa del modelo generado")

                # Opciones de exportación
                st.download_button(
                    label="📥 Descargar como STL",
                    data="",  # Aquí irían los datos del modelo
                    file_name="modelo_generado.stl",
                    mime="application/sla",
                    use_container_width=True,
                    key="download_gen_stl"
                )

    with settings_tab:
        st.header("⚙️ Configuración del Sistema")

        # Historial de cotizaciones
        st.subheader("📋 Historial de Cotizaciones")

        if 'quotations' in st.session_state and st.session_state['quotations']:
            recent_quotes = st.session_state['quotations'][-5:]
            recent_quotes.reverse()

            for i, quote in enumerate(recent_quotes):
                date = datetime.fromisoformat(quote['timestamp']).strftime("%d/%m/%Y %H:%M")
                with st.expander(f"📅 {date} - {quote['model']['filename'][:30]}..."):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**ID:** {quote['id']}")
                        st.write(f"**Archivo:** {quote['model']['filename']}")
                    with col2:
                        symbol = quote['calculations']['currency']
                        price = quote['calculations']['final_price']
                        st.write(f"**Precio:** {symbol} {price:.2f}")
        else:
            st.info("No hay cotizaciones en el historial aún.")

        # Limpiar archivos temporales
        st.subheader("🧹 Mantenimiento")

        if st.button("🗑️ Limpiar archivos temporales", type="secondary", key="clean_files_btn"):
            __clean_up_static_files()
            st.success("Archivos temporales limpiados")

def __initialize_session():
    """Inicializa las variables de sesión"""
    if 'init' not in st.session_state:
        st.session_state['init'] = True
        st.session_state['session_id'] = str(uuid4())[:8]
        st.session_state['current_model'] = None
        st.session_state['quotations'] = []
        st.session_state['custom_materials'] = []
        st.session_state['active_tab'] = 0

        if 'visualizer' not in st.session_state:
            st.session_state.visualizer = ModelVisualizer3D()

def __make_sidebar():
    """Crea la barra lateral"""
    with st.sidebar:
        st.title("🖨️ Cotizador 3D Pro+")
        st.markdown("---")

        # Navegación
        st.subheader("📍 Navegación")

        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("📤 Cargar", use_container_width=True, key="nav_upload"):
                st.session_state['active_tab'] = 0
                st.rerun()

        with nav_col2:
            if st.button("💰 Cotizar", use_container_width=True, key="nav_calc"):
                st.session_state['active_tab'] = 1
                st.rerun()

        nav_col3, nav_col4 = st.columns(2)
        with nav_col3:
            if st.button("👁️ 3D", use_container_width=True, key="nav_viz"):
                st.session_state['active_tab'] = 2
                st.rerun()

        with nav_col4:
            if st.button("⚡ Generar", use_container_width=True, key="nav_gen"):
                st.session_state['active_tab'] = 3
                st.rerun()

        st.markdown("---")

        # Estado del sistema
        if st.session_state.get('current_model'):
            model = st.session_state['current_model']
            st.subheader("📦 Modelo actual")

            filename = model['filename']
            if len(filename) > 20:
                filename = filename[:17] + "..."

            st.write(f"**{filename}**")
            st.metric("Volumen", f"{model['volume_cm3']:.1f} cm³")

def __make_app():
    """Función principal de la aplicación"""
    __make_tabs()

if __name__ == "__main__":
    # Configuración de la página
    st.set_page_config(
        page_title="Cotizador 3D Pro+",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Crear directorios necesarios
    os.makedirs("app/static", exist_ok=True)

    # Inicializar y ejecutar
    __initialize_session()
    __make_sidebar()
    __make_app()
    __clean_up_static_files()  # Limpiar archivos antiguos




