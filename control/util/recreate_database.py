from sqlalchemy import create_engine
from control.repository.postgres_objects import Base
from sqlalchemy.orm import sessionmaker

from control.config.database_config import DataBaseConfig

import logging


class RecreateDatabase:

    @staticmethod
    def execute():
        conf = DataBaseConfig()

        logging.info('Recreating database {}...'.format(conf.database_name))

        DATABASE_URI = 'postgres+psycopg2://postgres:{}@{}:5432/{}'.format(
            conf.password,
            conf.host,
            conf.database_name
        )

        print(DATABASE_URI)

        engine = create_engine(DATABASE_URI)

        Session = sessionmaker(bind=engine)
        s = Session()
        s.close_all()

        s.commit()
        print("call drop all")
        try:
            Base.metadata.drop_all(engine)
        except Exception as e:
            print(e)
        print("call create_all")
        Base.metadata.create_all(engine)
        print("Close")
        s.close()
