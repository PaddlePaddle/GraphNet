work under /GraphNet/

mkdir -p sqlite/logs

# migrate database
# Use default database path
python ./sqlite/init_db.py

# Specify custom database path
python ./sqlite/init_db.py --db_path sqlite/GraphNet.db


# Add data to database
bash ./sqlite/graphsample_insert.sh | tee "sqlite/logs/insert_$(date +'%Y-%m-%d-%H%M%S').log"
