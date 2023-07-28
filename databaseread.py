import psycopg2
from psycopg2 import Error
from dotenv import dotenv_values
import pandas as pd

env_vars = dotenv_values()

def dbread():

    try:
        connection = psycopg2.connect(
            user=env_vars.get('PG_DB_USERNAME'),
            password=env_vars.get('PG_DB_PASSWORD'),
            host=env_vars.get('PG_DB_SERVER'),
            port=env_vars.get('PG_DB_PORT'),
            database=env_vars.get('PG_DB_DATABASE')
        )
        cursor = connection.cursor()
        
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL:", error)
    
    try:
    
        cursor.execute("""SELECT asset, TO_CHAR(datalogged, 'YYYY-MM-DD') as datalogged, 
                          battery FROM mind4machine.textracklocationcheck
                          group by TO_CHAR(datalogged, 'YYYY-MM-DD'),asset,battery order by datalogged asc;""")
        temp = cursor.fetchall()
        df = pd.DataFrame(temp)
        mapping = {df.columns[0]:'yabby_kod', df.columns[1]:'datalogged', df.columns[2]:'battery'}
        df = df.rename(columns=mapping)
        return df
    
    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()