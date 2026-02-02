work under /GraphNet/

mkdir -p sqlite/logs

# migrate database
python ./sqlite/init_db.py

# Add data to database
bash ./sqlite/graphsample_insert.sh | tee "sqlite/logs/insert_$(date +'%Y-%m-%d-%H%M%S').log"
