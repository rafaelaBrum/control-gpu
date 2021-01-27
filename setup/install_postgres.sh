MSG='''
\n\n
###########################################################################################################################################\n
\t\n\n
\t INSTALL POSTGRES
\t\n\n
###########################################################################################################################################\n
'''

# this is needed to install the communication between Python and postgres
sudo apt install -y libpq-dev

# installing docker first

# shellcheck disable=SC2086
echo $MSG

# shellcheck disable=SC2086
mkdir -p $HOME/docker/volumes/postgres

# shellcheck disable=SC2086
docker run -e POSTGRES_PASSWORD=rafaela123 -d -p 5432:5432 -v $HOME/docker/volumes/postgres:/var/lib/postgresql/data --name pg-gpu-docker postgres -N 1500  -B 1096MB

cat clean_db.sql | docker exec -i pg-gpu-docker psql -U postgres

echo "Note: Before the control execution, to create the database, execute client.py recreate_db "

