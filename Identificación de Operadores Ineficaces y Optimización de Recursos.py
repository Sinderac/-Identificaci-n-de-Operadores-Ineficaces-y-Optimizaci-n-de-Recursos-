
# # Sprint 14: Proyecto final
# Link presentación: https://drive.google.com/file/d/1O-3hxd6jsBmfVx29RMpy1cCuybjIoquj/view?usp=sharing  
# Link tableu: https://public.tableau.com/views/Sprint14Proyectofinal/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link  


# ## Definición del Problema
# CallMeMaybe está desarrollando una función para ayudar a los supervisores a identificar a los operadores menos eficaces, basándose en indicadores como llamadas perdidas, tiempos de espera y bajo volumen de llamadas salientes. Este análisis explorará los datos de los conjuntos telecom_dataset_us.csv y telecom_clients_us.csv para identificar a estos operadores y probar hipótesis sobre su desempeño. El objetivo es proporcionar información útil que permita optimizar la atención al cliente.

# ### Objetivo: 
# Identificar operadores ineficaces basándose en las métricas definidas (llamadas perdidas, tiempo de espera, llamadas salientes).
# ### Entregables:
# Un informe con la identificación de operadores ineficaces y recomendaciones.

# ## Análisis Exploratorio de Datos (AED)

# In[1]:


#Cargamos librerías
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import math
import datetime as dt
from scipy import stats
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ### Carga de datos
# Importar los datasets telecom_dataset_us.csv y telecom_clients_us.csv.

# In[2]:


dataset_df = pd.read_csv("/datasets/telecom_dataset_us.csv")
clients_df = pd.read_csv("/datasets/telecom_clients_us.csv")


# ### Limpieza de datos
# Verificar y manejar valores nulos y duplicados.  
# Corregir los tipos de datos en caso de ser necesario.  

# #### telecom_dataset_us

# In[3]:


dataset_df.info()


# In[4]:


#Veamos como se ven los datos
dataset_df.head(5)


# In[5]:


#Operator_id cuenta con valores ausentes, debemos revisar el porcentaje que representa antes de excluirlos simplemente
IS_NA_sum = dataset_df["operator_id"].isna().sum()
total_count = dataset_df.shape[0]
IS_NA_perc = round(IS_NA_sum * 100 / total_count,2)
IS_NA_perc


# Representan un 15.16% de los datos, lo cual es una cantidad considerable. Podemos dejar los valores ausentes como "unknown" para no afectar el cálculo de los umbrales necesarios para categorizar a los operadores como eficientes o ineficientes. Aunque estos valores no afectarían a un operador en particular, al incluirlos nuestros umbrales contarían con una mayor cantidad de datos, lo que nos permitiría obtener un valor más preciso.

# In[6]:


#Verificamos que el operator_id no tenga valores decimales antes de pasarlo a tipo objeto
valores_con_decimales = dataset_df[(dataset_df["operator_id"] % 1 != 0)&(~dataset_df["operator_id"].isna())]
valores_con_decimales["operator_id"].count()


# No hay valores decimales en el operator_id, asi que podemos pasarlo a entero y luego tipo objeto sin preocuparnos de eliminar algún decimal.

# In[7]:


#Rellenamos los valores ausentes con 0, para al momento de pasarlo a entero no nos de error
dataset_df["operator_id"] = dataset_df["operator_id"].fillna(0)
#Pasamos a tipo entero para al momento de pasarlo a tipo objeto sea sin valores decimales
dataset_df["operator_id"] = dataset_df["operator_id"].astype("int")
#Pasamos a tipo objeto
dataset_df["operator_id"] = dataset_df["operator_id"].astype("object")
#Reemplazamos los ceros con unknown
dataset_df["operator_id"] = dataset_df["operator_id"].replace(0, "unknown")

#Cambiaremos los user_id a tipo objeto
dataset_df["user_id"] = dataset_df["user_id"].astype("object")

#Cambiaremos la columna data a tipo date
dataset_df["date"] = pd.to_datetime(dataset_df["date"])

#Cambiaremos la columna direction a tipo categoria
dataset_df["direction"] = dataset_df["direction"].astype("category")


# In[8]:


# Filtrar los valores que son iguales a '00:00:00+03:00'
same_time = dataset_df[dataset_df["date"].dt.strftime("%H:%M:%S%z") == "00:00:00+0300"]
same_time.shape[0]


# Vemos que en la columna date lo que son los datos de las horas, minutos, segundos y y timezone son iguales, asi que podemos excluir esa parte para que la fecha se vea más limpia.

# In[9]:


#Guardamos en date unicamente el año, mes y día
dataset_df["date"] = dataset_df["date"].dt.date

#Regresamos a tipo date la columna date
dataset_df["date"] = pd.to_datetime(dataset_df["date"])


# In[10]:


#Contamos con valores nulos en la columna internal, veremos que porcentaje representa de los datos
IS_NA_sum = dataset_df["internal"].isna().sum()
total_count = dataset_df.shape[0]
IS_NA_perc = round(IS_NA_sum * 100 / total_count,2)
IS_NA_perc


# Solo representa un 0.22%, no afectará en simplemente eliminar estos datos del dataframe.

# In[11]:


#Eliminamos los valores ausentes del dataframe
dataset_df = dataset_df.dropna()


# In[12]:


#Ya sin valores ausentes podemos pasar internal a tipo bool
dataset_df["internal"] = dataset_df["internal"].astype(bool)


# In[13]:


#Buscamos valores duplicados
dataset_df.duplicated().sum()


# Contamos con 4,893 valores duplicados. Como es muy poco probable que hayan llamado al mismo usuario, el mismo dia, que hayan durado el mismo tiempo en la llamada, etc. Eliminaremos estos valores duplicados.

# In[14]:


#Eliminamos valores duplicados
dataset_df = dataset_df.drop_duplicates().reset_index(drop=True)


# In[15]:


#veamos la diferencia de los datos
dataset_df.info()


# Ya se mira mejor el dataframe, continuemos con el siguiente.

# #### telecom_clients_us

# In[16]:


clients_df.info()


# In[17]:


#Hay que ver los primeros valores
clients_df.head(5)


# In[18]:


#Cambiaremos user_id a tipo objeto
clients_df["user_id"] = clients_df["user_id"].astype("object")

#Cambiaremos tariff_plan a category
clients_df["tariff_plan"] = clients_df["tariff_plan"].astype("category")

#Cambiaremos date_start a tipo date
clients_df["date_start"] = pd.to_datetime(clients_df["date_start"])


# In[19]:


#buscaremos valores duplicados
clients_df.duplicated().sum()


# Ya con los tipos de datos corregidos, sin valores duplicados y sin valores ausentes(viendo en info), ya esta listo el dataframe.


# ### Exploración inicial
# #### Describir los datos estadisticas descriptivas.  

# In[20]:


#hacemos un describe a la columna date para ver el rango de los valores
dataset_df["date"].describe()


# In[21]:


#Agruparemos los datos por operador y de manera mensual para obtener sus estadisticas mensuales
# Extraemos el mes, no hay necesidad de separarlo por año ya que todos se encuentran en el año 2019
dataset_df["month"] = dataset_df["date"].dt.month

#Agrupamos por operator y por mes
monthly_avg_operators_df = dataset_df.groupby(["operator_id", "month"]).mean().reset_index()

meses = {
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre"
}
# Creamos una nueva columna con los nombres de los meses
monthly_avg_operators_df["month"] = monthly_avg_operators_df["month"].map(meses)


# In[22]:


#Revisamos los datos estadisticos
monthly_avg_operators_df.describe().round(2)


# #### Visualizar distribuciones por medio de gráficos.  

# In[23]:


#Hacemos un bucle for para obtener la frecuencia de los valores de las columnas númericas
for col in monthly_avg_operators_df.select_dtypes(include="number").columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=monthly_avg_operators_df, x=col, bins="auto", kde=True)
    plt.title(f"Histograma del Promedio mensual de '{col}'")
    plt.xlabel(f"Promedio de {col}")
    plt.ylabel("Frecuencia")
    plt.show()
    #Hacemos un boxplot para 
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=monthly_avg_operators_df, x="month", y=col)
    plt.title(f"Diagrama de Cajas del Promedio mensual de '{col}'")
    plt.xlabel("Mes")
    plt.ylabel(f"Promedio de {col}")
    plt.show()


# Con la función detectamos que hay valores atipicos considerables en nuestro dataframe. La diferencia de la media y desviación estándar en las 3 columnas es considerable, viendo la diferencia entre el 75% al 100% y los resultados de las graficas. Tenemos valores atipicos que requieren de ser filtrados antes de proceder con el análisis.

# In[24]:


# Utilizaremos la función zscore para calcular los valores z de las variables numéricas del DataFrame
z_score = stats.zscore(monthly_avg_operators_df.select_dtypes("number"))


# In[25]:


#Filtraremos los datos en el que el zcore fue menor a 2.5
filtered_entries = (z_score <3).all(axis=1)
filtered_monthly_avg_operators_df = monthly_avg_operators_df[filtered_entries]


# In[26]:


#Hacemos un bucle for para obtener graficar la frecuencia de los valores de las columnas númericas
for col in filtered_monthly_avg_operators_df.select_dtypes(include="number").columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=filtered_monthly_avg_operators_df, x=col, bins="auto", kde=True)
    plt.title(f"Histograma del Promedio mensual de '{col}' ")
    plt.xlabel(f"Promedio de {col}")
    plt.ylabel("Frecuencia")
    plt.show()
    #Hacemos un boxplot para 
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=filtered_monthly_avg_operators_df, x="month", y=col)
    plt.title(f"Diagrama de Cajas del Promedio mensual de '{col}'")
    plt.xlabel("Mes")
    plt.ylabel(f"Promedio de {col}")
    plt.show()


# Aún tenemos valores atípicos, pero ahora podemos observar con mayor claridad los cuartiles. Al analizar el histograma y el scatter plot, notamos que la mayoría de los valores son relativamente pequeños, aunque algunos se encuentran fuera del rango esperado. Sin embargo, la frecuencia de estos valores atípicos es muy baja.

# #### Analizar correlaciones entre variables.  

# In[27]:


# Seleccionar columnas numéricas
num_cols = filtered_monthly_avg_operators_df.select_dtypes(include="number").columns

# Calcular y mostrar el coeficiente de correlación de Spearman para cada par de columnas
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):  # Asegúrate de no comparar la misma columna
        col1 = num_cols[i]
        col2 = num_cols[j]
        coef, p_value = spearmanr(filtered_monthly_avg_operators_df[col1], filtered_monthly_avg_operators_df[col2])
        print(f"Coeficiente de correlación entre {col1} y {col2}: {coef}, Valor p: {p_value} \n")


# In[28]:


# Listas para almacenar los coeficientes y los nombres de las variables
coefs = []
var_names = []

# Calcular el coeficiente de correlación de Spearman para cada par de columnas
for i in range(len(num_cols)):
    for j in range(i + 1, len(num_cols)):
        col1 = num_cols[i]
        col2 = num_cols[j]
        
        coef, p_value = spearmanr(filtered_monthly_avg_operators_df[col1], filtered_monthly_avg_operators_df[col2])
        coefs.append(coef)
        var_names.append(f"{col1} vs {col2}")

# Crear la gráfica
plt.figure(figsize=(12, 6))
x = range(len(var_names))

# Graficar coeficientes
plt.bar(x, coefs, width=0.4, color="blue", alpha=0.7)

# Añadir etiquetas y título
plt.xticks(x, var_names, rotation=45, ha="right")
plt.axhline(0, color='grey', lw=0.8)
plt.ylabel("Coeficiente de Correlación de Spearman")
plt.title("Coeficientes de Correlación de Spearman entre variables numéricas")
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# Entre is_missed_call y calls_count: 0.4855  
# Esto indica una correlación moderada positiva, lo que sugiere que a medida que aumenta el número de llamadas, también tiende a aumentar el número de llamadas perdidas. 
# 
# Entre calls_count y call_duration: 0.8174
# Una correlación alta positiva indica que un mayor número de llamadas se asocia con una mayor duración total de las llamadas.
# 
# Entre call_duration y total_call_duration: 0.9907  
# Esta es una correlación muy alta, lo que implica que la duración de las llamadas individuales y la duración total están muy fuertemente relacionadas.


# ### Identificación de métricas

# In[29]:


#Excluiremos operadores que solo tengan llamdas internas
filtered_operators_df = filtered_monthly_avg_operators_df[filtered_monthly_avg_operators_df["internal"]!=1]

#Eliminaremos la columna internal ya que no es necesaria
filtered_operators_df = filtered_operators_df.drop(columns =["internal"]).reset_index(drop=True)


# #### Porcentaje de llamadas  pérdidas por operador

# In[30]:


# Calculamos el porcentaje promedio de llamadas perdidas por operador
avg_missed_calls = round((filtered_operators_df["is_missed_call"].mean())*100,2)


# #### Tiempo medio de espera.  

# In[31]:


#Calculamos el tiempo de espera en las llamadas 
filtered_operators_df["waiting_time"] = filtered_operators_df["total_call_duration"] - filtered_operators_df["call_duration"]

#Calculamos el tiempo promedio de espera en las llamadas
avg_wait_time = filtered_operators_df["waiting_time"].mean().round(2)


# #### Promedio de llamadas por operador.  

# In[32]:


# Calculamos el promedio de llamadas 
avg_calls = filtered_operators_df["calls_count"].mean().round(2)


# In[33]:


# Calculamos la desviación estándar del tiempo de espera
std_wait_time = filtered_operators_df["waiting_time"].std().round(2)

# Calculamos la desviación estándar del porcentaje de llamadas perdidas por operador
std_missed_calls = round(filtered_operators_df["is_missed_call"].std()*100, 2)

# Calculamos la desviación estándar del promedio de llamadas realizadas por operador
std_calls_count = filtered_operators_df["calls_count"].std().round(2)

print(f" El tiempo promedio de espera es de: {avg_wait_time}")
print(f" El porcentaje de llamadas perdidas promedio es de: {avg_missed_calls}")
print(f" El  promedio de llamadas es de: {avg_calls}")
print()
print(f" La desviación estándar del tiempo de espera es de: {std_wait_time}")
print(f" La desviación estándar del porcentaje las llamadas perdidas es de: {std_missed_calls}")
print(f" La desviación estándar de las llamadas es de: {std_calls_count}")


# Observando las desviaciones estándar vemos que son mayores que los valores promedio, lo cual nos dice que hay una varianza entre los datos. 
# Para no afectar en los calculos de los umbrales, usaremos el metodo de los cuartiles ya que si lo calculamos con las desviaciones terminariamos siendo muy flexibles y dejando una gran parte de los operadores como eficientes.

# ## Identificar operadores ineficaces

# Consideramos ineficiente a un operador que se encuentre por debajo del umbral promedio del 70% del cuartil en dos o más de los siguientes indicadores:
# 
# Tiempo Promedio de Espera: Si tiene un mayor tiempo promedio de espera o un mayor tiempo promedio de inactividad en las llamadas.
# 
# Porcentaje de Llamadas Perdidas: Si presenta un mayor porcentaje promedio de llamadas perdidas.
# 
# Cantidad de Llamadas: Si tiene una menor cantidad promedio de llamadas, ya sean recibidas o realizadas.

# ### Establecer umbrales

# In[34]:


# Definimos los umbrales
threshold_wait_time = filtered_operators_df["waiting_time"].quantile(0.7)
threshold_missed_calls = filtered_operators_df["is_missed_call"].quantile(0.7)
threshold_calls = filtered_operators_df["calls_count"].quantile(0.7)

print(f" El valor de la umbral de el tiempo de espera es de: {threshold_wait_time:.2f}")
print(f" El valor de la umbral del porcentaje de llamadas perdidas es de: {threshold_missed_calls:.2f}")
print(f" El valor de la umbral de llamadas es de: {threshold_calls:.2f}")



# ### Clasificar operadores

# In[35]:


def clasificar_operador(row):
    count = 0
    if row["is_missed_call"] > threshold_missed_calls:
        count += 1
    if row["waiting_time"] > threshold_wait_time:
        count += 1
    if row["calls_count"] < threshold_calls:
        count += 1
    return "inefficient" if count >= 2 else "efficient"

# Agregamos una nueva columna con la clasificación
filtered_operators_df["classification"] = filtered_operators_df.apply(clasificar_operador, axis=1)

# Contamos la cantidad de meses por operador
months_count = filtered_operators_df.groupby("operator_id")["month"].nunique().reset_index()
months_count.columns = ["operator_id", "months_count"]

# Convertimos la clasificación a un valor numérico
months_classification = filtered_operators_df.groupby(["operator_id", "month"])["classification"].first().reset_index()
months_classification["clasificacion_num"] = months_classification["classification"].map({"efficient": 1, "inefficient": 0})

# Calculamos el promedio mensual de eficiencia por operador
avg_classification = months_classification.groupby("operator_id")["clasificacion_num"].mean().reset_index()
avg_classification.columns = ["operator_id", "avg_efficiency"]

# Unimos la cantidad de meses con la clasificación promedio
avg_classification = avg_classification.merge(months_count, on="operator_id")

# Clasificamos a los operadores según su cantidad de meses y umbral
def classify_based_on_months(row):
    if row["months_count"] <= 2:
        return 1 if row["avg_efficiency"] >= 0.5 else 0
    elif row["months_count"] >= 3:
        return 1 if row["avg_efficiency"] >= 0.65 else 0

# Aplicamos la función de clasificación
avg_classification["final_classification"] = avg_classification.apply(classify_based_on_months, axis=1)

# Mapeamos los valores 1 a "efficient" y 0 a "inefficient"
avg_classification["final_classification"] = avg_classification["final_classification"].map({1: "efficient", 0: "inefficient"})

# Unimos la clasificación con las métricas
merged_df = filtered_operators_df.merge(avg_classification[["operator_id", "final_classification", "months_count", "avg_efficiency"]], on="operator_id")

merged_df


# In[36]:


# Aplicamos la misma lógica de clasificación para crear una columna 'class'
merged_df["class"] = merged_df.apply(classify_based_on_months, axis=1)

# Definir las características (X) y la variable objetivo (y)
X = merged_df[["is_missed_call", "calls_count", "call_duration", "total_call_duration", "waiting_time", "avg_efficiency"]]
y = merged_df["class"]

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Crear y entrenar un modelo
model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Predecir y evaluar el modelo
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Mostrar resultados
print("Precisión del modelo:", accuracy)


# Obtuvimos una precisión practicamente del 99%, lo cual casi lo vuelve una clasificación perfecta.

# In[37]:


#Obtenemos frames con operadores eficientes e ineficientes
efficient = merged_df[merged_df["final_classification"] == "efficient"]
inefficient = merged_df[merged_df["final_classification"] == "inefficient"]



   
# Me di cuenta que hay operadores que trabajaron una diferente cantidad de meses. Ajuste los umbrales dependiendo de la cantidad de meses que trabajaron.
#     
# Si trabajaron 1 o 2 meses, su umbral es de 0.5; esto para que si fueron eficiente en uno de esos meses, sea considerado como eficiente.  
# Mientras si trabajaron 3 o 4 meses, su umbral es de 0.65; para que si fueron eficientes en 2 meses en el caso de tener 3 meses y 3 meses en caso de contar con 4 meses, sean considerados como eficientes.


# ## Prueba de Hipótesis Estadísticas
# Hipótesis a considerar:
# 
# H0: No hay diferencia en el número de llamadas perdidas entre operadores eficaces e ineficaces.  
# H1: Hay una diferencia significativa en el número de llamadas perdidas entre operadores eficaces e ineficaces.

# ### Escoger y realizar la prueba:
# Calcular el estadístico de la prueba y el valor p.  
# Ya que vamos a comprar medias de 2 grupos, escogi la prueba T con varizanas diferentes.

# In[38]:


# Variables a comparar
variables = ["waiting_time", "is_missed_call", "calls_count"]

# Almacenar resultados
results = []

for var in variables:
    # Separar los datos
    efficient_ = efficient[var]
    inefficient_ = inefficient[var]
    
    # Realizar la prueba t
    t_statistic, p_value = stats.ttest_ind(efficient_, inefficient_, equal_var=False)
    
    # Guardar resultados
    results.append({
        "variable": var,
        "t_value": t_statistic,
        'p_value': p_value
    })

# Crear un DataFrame con los resultados
results_df = pd.DataFrame(results)

results_df



# ### Interpretar los resultados:
# Comparar el valor p con el nivel de significancia (e.g., α = 0.05) para aceptar o rechazar la hipótesis nula.

# In[39]:


# Definimos el nivel de significancia
alpha = 0.05

# Recorremos los resultados y comparar los valores p con el nivel de significancia
for index, row in results_df.iterrows():
    var = row["variable"]
    p_value = row["p_value"]
    
    if p_value <= alpha:
        print(f"Rechazamos la hipótesis nula: hay diferencia significativa en la columna '{var}'.")
    else:
        print(f"No podemos rechazar la hipótesis nula: no hay diferencia significativa en la columna '{var}'.")


# In[40]:


efficent_operator= efficient["operator_id"].unique()
inefficent_operator= inefficient["operator_id"].unique()


# In[41]:


print(f"El total de operadores eficientes fue de : {efficent_operator.shape[0]}")
print(f"El total de operadores ineficientes fue de : {inefficent_operator.shape[0]}")


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a><br>
# 
# <b>Éxito</b> - Interpretaste correctamente los resultados de las pruebas de hipótesis, rechazando la hipótesis nula en las variables clave con valores p significativamente bajos. Tus conclusiones son claras y concisas.
# </div>
# 
# 

# In[42]:


# Defimos las columnas a analizar
vars = [
    "calls_count",
    "call_duration",
    "total_call_duration",
    "waiting_time",
    "avg_efficiency"
]

# Calculamos y mostramos el coeficiente de correlación de Spearman para cada par de columnas
for i in range(len(vars)):
    for j in range(i + 1, len(vars)): 
        col1 = vars[i]
        col2 = vars[j]
        coef, p_value = spearmanr(efficient[col1], efficient[col2])
        if abs(coef) > 0.7: #Imprimimos solo correlaciones alta
            print(f"Coeficiente de correlación alta entre {col1} y {col2}: {coef:.2f}, Valor p: {p_value:2e}\n")
        elif abs(coef)>= 0.4: #Imprimimos solo correlaciones moderadas
            print(f"Coeficiente de correlación moderada entre {col1} y {col2}: {coef:.2f}, Valor p: {p_value:2e}\n")
        else: #Imprimimos correlaciones bajas
            print(f"Coeficiente de correlación baja entre {col1} y {col2}: {coef:.2f}, Valor p: {p_value:2e}\n")


# Fuertes correlaciones: Las correlaciones más fuertes se observan entre calls_count, call_duration, y total_call_duration, lo que indica que a medida que aumentan las llamadas, tanto la duración de las llamadas como el total de duración de las llamadas también tienden a aumentar.

# In[43]:


# Calculamos y mostramos el coeficiente de correlación de Spearman para cada par de columnas
for i in range(len(vars)):
    for j in range(i + 1, len(vars)): 
        col1 = vars[i]
        col2 = vars[j]
        coef, p_value = spearmanr(inefficient[col1], inefficient[col2])
        if abs(coef) > 0.7: #Imprimimos solo correlaciones alta
            print(f"Coeficiente de correlación alta entre {col1} y {col2}: {coef:.2f}, Valor p: {p_value:2e}\n")
        elif abs(coef)>= 0.4: #Imprimimos solo correlaciones moderadas
            print(f"Coeficiente de correlación moderada entre {col1} y {col2}: {coef:.2f}, Valor p: {p_value:2e}\n")
        else: #Imprimimos correlaciones bajas
            print(f"Coeficiente de correlación baja entre {col1} y {col2}: {coef:.2f}, Valor p: {p_value:2e}\n")


# Fuertes correlaciones: Los coeficientes entre calls_count, call_duration, y total_call_duration son muy altos, indicando que a medida que aumenta el número de llamadas, también aumentan la duración de las llamadas y el tiempo total de llamadas.
# 
# Correlaciones negativas con avg_efficiency: Aunque la correlación entre calls_count y avg_efficiency es moderadamente negativa, sugiere que un mayor número de llamadas podría estar asociado con una eficiencia promedio más baja.

# In[44]:


# Listas para almacenar los coeficientes y nombres de las variables
efficient_coefs = []
inefficient_coefs = []
var_names = []

# Calcular los coeficientes de correlación y clasificarlos
for i in range(len(vars)):
    for j in range(i + 1, len(vars)):
        col1 = vars[i]
        col2 = vars[j]
        
        # Coeficiente para operadores eficientes
        coef_efficient, _ = spearmanr(efficient[col1], efficient[col2])
        efficient_coefs.append(coef_efficient)

        # Coeficiente para operadores ineficientes
        coef_inefficient, _ = spearmanr(inefficient[col1], inefficient[col2])
        inefficient_coefs.append(coef_inefficient)

        # Agregar nombres de las variables
        var_names.append(f"{col1} vs {col2}")

# Crear la gráfica
plt.figure(figsize=(12, 6))
x = range(len(var_names))

# Graficar coeficientes
plt.bar(x, efficient_coefs, width=0.4, label="Eficientes", align="center", alpha=0.7)
plt.bar([p + 0.4 for p in x], inefficient_coefs, width=0.4, label="Ineficientes", align="center", alpha=0.7)

# Añadir etiquetas y título
plt.xticks([p + 0.2 for p in x], var_names, rotation=45, ha="right")
plt.axhline(0, color="grey", lw=0.8)
plt.ylabel("Coeficiente de Correlación de Spearman")
plt.title("Comparación de Coeficientes de Correlación: Eficientes vs Ineficientes")
plt.legend()
plt.tight_layout()

# Mostrar la gráfica
plt.show()


# El detalle clave es el tiempo de espera entre las llamadas, aunque los operadores eficientes cuentan con un número de llamadas, si tiempo de espera/tiempo muerto es menor que los operadores ineficientes. Lo podemos ver en diferentes correlaciones.



# ## Conclusiones y recomendaciones

# ### Conclusiones
# Waiting Time:
# Un valor t de -14.11 significa que los operadores eficientes tienen un menor tiempo de espera en comparación con los ineficientes. Esto indica que los operadores eficientes logran atender más rápido a los clientes.
# 
# Is Missed Call:  
# Un valor t de -31.59 indica que los operadores eficientes tienen menos llamadas perdidas en comparación con los ineficientes. Esto sugiere que son más efectivos en su trabajo y logran resolver más llamadas exitosamente.
# 
# Calls Count:  
# Un valor t de -14.35 sugiere que los operadores eficientes realizan menos llamadas en comparación con los ineficientes. Esto puede ser un indicador de que están siendo más eficientes en su tiempo de atención, resolviendo problemas sin necesidad de tantas interacciones.
# 
# Clasificamos correctamente los operadores, puede que unos valores umbrales sean aún debatibles y pueden ser ajustados para estar permisibles, pero con estos umbrales clasificamos a 739 operadores como eficientes y 290 como ineficientes.

# ### Recomendaciones
# Capacitación Continua: Se recomienda implementar programas de capacitación y desarrollo continuo para todos los operadores, enfocándose en las mejores prácticas observadas en los operadores eficientes. Esto podría ayudar a elevar el rendimiento de los operadores ineficientes.
# 
# Monitoreo y Retrolimentación: Establecer un sistema de monitoreo y retroalimentación regular para evaluar el desempeño de los operadores. Esto podría incluir revisiones periódicas del tiempo de espera y la cantidad de llamadas perdidas, permitiendo ajustes en tiempo real.
# 
# Mejora de Procesos: Analizar los procesos de atención al cliente para identificar cuellos de botella que puedan estar contribuyendo a mayores tiempos de espera y pérdidas de llamadas. Implementar mejoras en los procesos podría beneficiar a todos los operadores.
# 
# Reconocimiento y Motivación: Implementar un sistema de reconocimiento para los operadores que consistentemente demuestren un alto rendimiento. Esto podría mejorar la moral del equipo y motivar a otros operadores a alcanzar estándares similares.
# 
# Análisis Continuo: Realizar análisis continuos del rendimiento de los operadores y ajustar las estrategias según sea necesario. La recolección y análisis de datos deben ser parte integral de la gestión del rendimiento en la atención al cliente.