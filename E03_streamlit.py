import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Análisis de Variables y Predicción", layout="wide")

# Título y descripción de la aplicación
st.title("🔍 Análisis de Variables y Predicción de Esperanza de Vida")
st.markdown("Esta aplicación permite analizar y predecir la esperanza de vida basándose en distintas variables socioeconómicas y demográficas.")

# Cargar datos directamente desde el archivo state_x77.csv
datos = pd.read_csv('state_x77.csv')
st.success("Datos cargados con éxito!")
st.write("Vista previa de los datos:", datos.head(3))

# Selección de características y variable objetivo
X = datos[['habitantes', 'ingresos', 'analfabetismo', 'asesinatos', 'universitarios', 'heladas', 'area', 'densidad_pobl']]
y = datos['esp_vida']

# División en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Matriz de Correlación
with st.expander("📊 Matriz de Correlación y Mapa de Calor"):
    st.subheader("Matriz de Correlación")
    corr_matrix = datos.corr(method='pearson')

    # Formatear y limpiar la matriz de correlación
    tril = np.tril(np.ones(corr_matrix.shape)).astype(bool)
    corr_matrix[tril] = np.nan
    corr_matrix_tidy = corr_matrix.stack().reset_index(name='r')
    corr_matrix_tidy = corr_matrix_tidy.rename(columns={'level_0': 'variable_1', 'level_1': 'variable_2'})
    corr_matrix_tidy = corr_matrix_tidy.dropna().sort_values('r', key=abs, ascending=False).reset_index(drop=True)

    # Mostrar matriz tidy
    st.write("### Matriz de Correlación Ordenada")
    st.dataframe(corr_matrix_tidy)

    # Mostrar el mapa de calor con valores
    st.write("### Mapa de Calor de la Correlación")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(datos.corr(), annot=True, cmap='coolwarm', ax=ax, cbar_kws={'shrink': 0.8}, fmt=".2f")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    st.pyplot(fig)

# Selección de variables con Forward Selection
with st.expander("📈 Selección de Variables con Forward Selection"):
    def forward_selection(X, y, criterio='aic', add_constant=True):
        if add_constant:
            X = sm.add_constant(X, prepend=True).rename(columns={'const': 'intercept'})
        restantes = X.columns.to_list()
        seleccion = []
        mejor_metrica = -np.inf if criterio == 'rsquared_adj' else np.inf
        ultima_metrica = mejor_metrica

        while restantes:
            metricas = []
            for candidata in restantes:
                seleccion_temp = seleccion + [candidata]
                modelo = sm.OLS(endog=y, exog=X[seleccion_temp])
                modelo_res = modelo.fit()
                metrica = getattr(modelo_res, criterio)
                metricas.append(metrica)
            mejor_metrica = max(metricas) if criterio == 'rsquared_adj' else min(metricas)
            if (criterio == 'rsquared_adj' and mejor_metrica > ultima_metrica) or (criterio != 'rsquared_adj' and mejor_metrica < ultima_metrica):
                mejor_variable = restantes[np.argmax(metricas) if criterio == 'rsquared_adj' else np.argmin(metricas)]
                seleccion.append(mejor_variable)
                restantes.remove(mejor_variable)
                ultima_metrica = mejor_metrica
            else:
                break

        return sorted(seleccion)

    criterio = st.selectbox("Selecciona el criterio de evaluación:", ["aic", "rsquared_adj"])
    predictores = forward_selection(X_train, y_train, criterio=criterio, add_constant=False)
    st.write("### Variables Seleccionadas:", predictores)

    # Entrenamiento del modelo con las variables seleccionadas
    X_train = sm.add_constant(X_train, prepend=True).rename(columns={'const': 'intercept'})
    modelo_final = sm.OLS(endog=y_train, exog=X_train[predictores])
    modelo_final_res = modelo_final.fit()

    # Mostrar el resumen del modelo en varias líneas
    st.write("### Resumen del Modelo:")
    modelo_resumen = modelo_final_res.summary().as_text()  # Convertir el resumen a texto con saltos de línea
    st.text(modelo_resumen)  # Mostrar el resumen completo en varias líneas

# Selección de predictores con Sklearn
with st.expander("🤖 Selección de Predictores con Sklearn SequentialFeatureSelector"):
    modelo_sklearn = LinearRegression()
    sfs = SequentialFeatureSelector(modelo_sklearn, n_features_to_select='auto', direction='forward', scoring='r2', cv=5)
    sfs.fit(X_train, y_train)
    st.write("### Predictores Seleccionados:", sfs.get_feature_names_out().tolist())

# Predicción con una nueva muestra
with st.expander("📋 Predicción con Nueva Muestra"):
    st.write("Ingresa los valores para los predictores seleccionados:")
    nueva_muestra_input = {col: 0.0 for col in predictores if col != 'intercept'}
    nueva_muestra_input['intercept'] = 1  # Asegurarse de añadir el intercepto

    # Crear entradas en la interfaz para cada predictor
    for predictor in predictores:
        if predictor != 'intercept':
            nueva_muestra_input[predictor] = st.number_input(f"{predictor}", value=0.0)

    # Crear DataFrame con los valores ingresados
    nueva_muestra = pd.DataFrame([nueva_muestra_input])

    # Realizar la predicción
    prediccion = modelo_final_res.predict(nueva_muestra[predictores])
    st.write(f"### Predicción de Esperanza de Vida: {prediccion[0]:.2f} años")

    # Predicción con intervalo de confianza
    predicciones_intervalo = modelo_final_res.get_prediction(nueva_muestra[predictores]).summary_frame(alpha=0.05)
    st.write("### Intervalo de Confianza de la Predicción:")
    st.dataframe(predicciones_intervalo)