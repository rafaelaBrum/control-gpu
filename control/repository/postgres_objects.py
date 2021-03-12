from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey, ForeignKeyConstraint, TIMESTAMP,Interval
from sqlalchemy.orm import relationship

Base = declarative_base()


class Task(Base):
    __tablename__ = 'task'
    task_id = Column(Integer, primary_key=True)
    task_name = Column(String)
    command = Column(String)

    executions = relationship('Execution', backref='task', lazy='dynamic')

    def __repr__(self):
        return "<Task(task_id='{}' command='{}'>".format(self.task_id, self.task_name, self.command)


class InstanceType(Base):
    __tablename__ = 'instance_type'
    type = Column(String, primary_key=True)
    provider = Column(String)

    instances = relationship('Instance', backref='instance_type', lazy='dynamic')

    def __repr__(self):
        return "<InstanceType(type='{}'>".format(self.type)


class Instance(Base):
    __tablename__ = 'instance'
    id = Column(String, primary_key=True)
    type = Column(String, ForeignKey('instance_type.type'))
    region = Column(String)
    zone = Column(String)
    ebs_volume = Column(String)
    market = Column(String)
    price = Column(Float)

    execution = relationship('Execution', backref='instance', lazy='dynamic')

    def __repr__(self):
        return "<Instance(instance_id='{}', type='{}' region='{}', zone='{}', market='{}', price='{}')>" \
            .format(self.id, self.type, self.region, self.zone, self.market, self.price)


class Execution(Base):
    __tablename__ = 'execution'
    execution_id = Column(Integer, primary_key=True)
    task_id = Column(Integer, primary_key=True)
    instance_id = Column(String, ForeignKey('instance.id'), primary_key=True)
    timestamp = Column(TIMESTAMP, primary_key=True)

    status = Column(String)

    __table_args__ = (ForeignKeyConstraint(['task_id', ], [Task.task_id]),
                      {})

    def __repr__(self):
        return "Execution: <task_id='{}', instance_id='{}', timestamp='{}', status='{}'>".format(
            self.task_id,
            self.instance_id,
            self.timestamp,
            self.status)


class InstanceStatistic(Base):
    __tablename__ = 'instance_statistic'
    instance_id = Column(String, ForeignKey('instance.id'), primary_key=True)
    deploy_overhead = Column(Float)
    termination_overhead = Column(Float)
    uptime = Column(Float)

    def __repr__(self):
        return "InstanceStatistic: <instance_id='{}', " \
               "Deploy Overhead='{}' Termination_overhead='{}'>".format(self.instance_id,
                                                                        self.deploy_overhead,
                                                                        self.termination_overhead)


class Statistic(Base):
    __tablename__ = 'statistic'
    execution_id = Column(Integer, primary_key=True)
    task_id = Column(Integer, primary_key=True)
    start = Column(TIMESTAMP, primary_key=True)
    end = Column(TIMESTAMP)
    deadline = Column(Interval)
    cost = Column(Float)
    status = Column(String)
