#enviar algo para a VM:
gcloud compute scp PATH_TO_FILE  --zone INSTANCE_ZONE INSTANCE_NAME:~
#acessar a VM:
gcloud compute ssh --zone INSTANCE_ZONE INSTANCE_NAME

#nome usuário: sa_109649273287045369425 (o número é o client_id da conta de servico)

#SuperLink command (one in the server VM)
flower-superlink --control-api-address 0.0.0.0:9093 --serverappio-api-address 0.0.0.0:9091 --fleet-api-address 0.0.0.0:9092 --insecure

#SuperNode command (one per client)
flower-supernode --insecure --superlink IP_SERVER:9092 --clientappio-api-address 0.0.0.0:9094 --max-retries 1

#Execution command - in the server VM
flwr run . local-deployment --stream


#PreScheduling execution
flwr run . local-deployment --stream --run-config "length-parameters=10000 file='teste.json'"

#Quitting a screen
screen -S screen_name -X quit