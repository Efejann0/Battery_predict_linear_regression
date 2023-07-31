import databaseread as  dbc
import battery_pred as  batprd

def main():
    data = dbc.dbread()
    batprd.battery_predict(data)
    
if __name__ == "__main__":
    main()
 