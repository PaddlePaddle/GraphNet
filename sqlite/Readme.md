work under /GraphNet/

# Initialize database (create tables if not exist)
python ./sqlite/orm_models.py init

# Drop all tables
python ./sqlite/orm_models.py drop

# Add data to database
bash ./sqlite/graphsample_insert.sh