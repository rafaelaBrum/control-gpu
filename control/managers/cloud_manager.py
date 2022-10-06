# Each supported cloud have to implement the follows methods


class CloudManager:
    # VM STATE
    PENDING = 'pending'
    RUNNING = 'running'
    STOPPING = 'stopping'
    STOPPED = 'stopped'
    SHUTTING_DOWN = 'shutting-down'
    TERMINATED = 'terminated'
    HIBERNATING = 'hibernating'
    HIBERNATED = 'stop - hibernate'

    ERROR = 'error'
    ABORT = 'abort'

    # QUEUE STATE
    IDLE = 'idle'
    WORKING = 'working'  # == Busy
    FAILED = 'failed'    # == Failed

    # MARKET
    ON_DEMAND = 'on-demand'
    PREEMPTIBLE = 'preemptible'

    # BURSTABLE = 'burstable'

    # PROVIDERs
    AWS = 'aws'
    EC2 = 'ec2'
    GCLOUD = 'gcloud'
    GCP = 'gcp'
    CLOUDLAB = 'CloudLab'

    # FILE SYSTEM

    S3 = 's3'
    EBS = 'ebs'
    EFS = 'efs'

    # [START CREATE_INSTANCE]

    # Create an on-demand instance
    # InputConfig: Instance type (str)
    # Return: instance_id (str) or None if instance was not created
    def create_on_demand_instance(self, instance_type, image_id, zone=''):
        pass

    # Create a preemptible_instance
    # InputConfig: Instance type (str)
    # Return: instance_id (str) or None if instance was not created
    def create_preemptible_instance(self, instance_type, image_id, max_price, zone=''):
        pass

    # Create a storage volume
    # InputConfig: Volume size (int), volume_name (str)
    # Return: volume id (str) or None if volume was not created
    def create_volume(self, size, volume_name='', zone=''):
        pass

    # Attach a volume to an instance
    # InputConfig: instance_id (str), volume_id (str)
    # Return: True if the volume was attached with success or False otherwise
    def attach_volume(self, instance_id, volume_id, zone=''):
        pass

    # Delete a storage volume
    # InputConfig: Volume id (str)
    # Return: True if volume was delete or false otherwise
    def delete_volume(self, volume_id, zone=''):
        pass

    # [TERMINATE_INSTANCE]

    # Terminate an instance
    # InputConfig: Instance_id (str)
    # Output: Operate State (Success: True, Error: False)
    def terminate_instance(self, instance_id, wait=True, zone=''):
        pass

    # [GET_INFO]

    # Return the current status of the VM
    # InputConfig: Instance_id (str)
    # Output: instance_status (str) in lower case or None if Instance not found
    def get_instance_status(self, instance_id, zone=''):
        pass

    # Return all instances_id
    # InputConfig: Filter (dict)
    # Output: instances_id (List)
    def list_instances_id(self, list_filter=None, zone=''):
        """
                  # Filter Format #

                      search_filter = {
                          'status' :  [ list of VALID STATES]
                      }
                  Valid State:  PENDING, RUNNING, STOPPING, STOPPED
                                SHUTTING_DOWN, TERMINATED, HIBERNATED
           """
        pass

    # Return the current public IP of the VM
    # InputConfig: instance_id
    # Output: ip(str) or None if there is no IP associated to the VM
    def get_public_instance_ip(self, instance_id, zone=''):
        pass

    # Return the current private IP of the VM
    # InputConfig: instance_id
    # Output: ip(str) or None if there is no IP associated to the VM
    def get_private_instance_ip(self, instance_id, zone=''):
        pass

    # Return the current CPU credit of a burstable instance
    # InputConfig: Instance_id(str)
    # Output: cpu_credits (int) or None if the instance is not burstable
    # def get_cpu_credits(self, instance_id, zone=''):
    #     pass
