work under /GraphNet/

mkdir -p sqlite/logs
# Initialize database (create tables if not exist)
python ./sqlite/orm_models.py init | tee "sqlite/logs/init_$(date +'%Y-%m-%d-%H%M%S').log"

# Drop all tables
python ./sqlite/orm_models.py drop | tee "sqlite/logs/drop_$(date +'%Y-%m-%d-%H%M%S').log"

# Add data to database
bash ./sqlite/graphsample_insert.sh | tee "sqlite/logs/insert_$(date +'%Y-%m-%d-%H%M%S').log"