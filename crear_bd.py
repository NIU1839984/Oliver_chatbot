import sqlite3

print("1. Creando base de datos SQLite...")
# Esto creará un archivo llamado 'clientes.db' en tu carpeta
conexion = sqlite3.connect("clientes.db")
cursor = conexion.cursor()

print("2. Construyendo la tabla de perfiles...")
# Creamos la tabla con las 3 columnas que necesitamos
cursor.execute('''
CREATE TABLE IF NOT EXISTS usuarios (
    telefono TEXT PRIMARY KEY,
    nombre TEXT,
    privilegio TEXT
)
''')

# Vaciamos la tabla por si ejecutamos el script más de una vez
cursor.execute('DELETE FROM usuarios')

print("3. Insertando los clientes de prueba...")
# Simulamos los perfiles que pide la guía técnica
cursor.execute("INSERT INTO usuarios VALUES ('600000000', 'Adri', 'total')")
cursor.execute("INSERT INTO usuarios VALUES ('611111111', 'Cliente Casual', 'limitado')")

# Guardamos los cambios y cerramos
conexion.commit()
conexion.close()

print("¡ÉXITO! Base de datos 'clientes.db' generada correctamente.")